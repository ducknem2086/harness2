"""Worker for the decision service's harness jobs: claim, improve, post the outcome.

    harness2-decision-worker --decision-url http://localhost:3000 [--once]
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

from harness2 import config
from harness2.core.proposer import Proposer
from harness2.decisions.client import DecisionClient, LeaseLost, ServiceError
from harness2.decisions.improve import improve
from harness2.decisions.library import ExperienceLibrary
from harness2.decisions.proposer import make_proposer

log = logging.getLogger("harness2.decision_worker")

# One proposer session may run up to 30 minutes against a lease of about the same length,
# so a background heartbeat keeps the lease between `improve`'s own per-step heartbeats.
HEARTBEAT_SECONDS = 300.0
_KEEPER_JOIN_SECONDS = 5.0


class _LeaseKeeper:
    """Heartbeats a job from a background thread while the `with` block runs.

    `lost` is set when the service says the lease is gone. A `ServiceError` is ignored:
    the next per-step heartbeat or evaluate call surfaces a real outage.
    """

    def __init__(self, client: DecisionClient, job_id: str, interval: float):
        self.lost = threading.Event()
        self._client = client
        self._job_id = job_id
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lease-keeper", daemon=True)

    def __enter__(self) -> "_LeaseKeeper":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=_KEEPER_JOIN_SECONDS)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._client.heartbeat(self._job_id)
            except LeaseLost:
                self.lost.set()
                return
            except ServiceError:
                continue


def run_once(
    client: DecisionClient,
    proposer: Proposer,
    library: ExperienceLibrary,
    k: int,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> str:
    """Claim and process at most one job.

    Returns "idle" (nothing to do), "succeeded", "no-improvement", "failed" or
    "lease-lost" (the job is gone; nothing was posted). A `ServiceError` from `claim`
    propagates: the caller decides how to wait.
    """
    job = client.claim()
    if job is None:
        return "idle"
    log.info("job %s: claimed", job.id)
    error: Exception | None = None
    outcome = None
    with _LeaseKeeper(client, job.id, heartbeat_seconds) as keeper:
        try:
            outcome = improve(job, client, proposer, library, k=k)
        except LeaseLost:
            return _lease_lost(job.id)
        except Exception as exc:
            error = exc
    if keeper.lost.is_set():
        return _lease_lost(job.id)
    try:
        if error is not None:
            return _fail(client, job.id, error)
        if outcome.kind == "succeeded":
            client.succeed(job.id, outcome.questions, outcome.report)
        else:
            client.no_improvement(job.id, outcome.report)
    except LeaseLost:
        return _lease_lost(job.id)
    except ServiceError as exc:
        log.warning("job %s: could not post the %s result: %s", job.id, outcome.kind, exc)
        return "failed"
    log.info("job %s: %s", job.id, outcome.kind)
    return outcome.kind


def _fail(client: DecisionClient, job_id: str, error: Exception) -> str:
    reason = f"{type(error).__name__}: {error}"
    log.warning("job %s: failed: %s", job_id, reason)
    try:
        client.fail(job_id, reason)
    except ServiceError as exc:
        log.warning("job %s: could not report the failure: %s", job_id, exc)
    return "failed"


def _lease_lost(job_id: str) -> str:
    log.warning("job %s: lease lost; dropping the job without a result", job_id)
    return "lease-lost"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness2-decision-worker",
        description="Improve the routing text of workflows by claiming jobs from the decision service.",
    )
    parser.add_argument("--decision-url", default="http://localhost:3000")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    parser.add_argument("--k", type=int, default=3, help="independent candidates per job")
    parser.add_argument(
        "--model", default="", help="Claude model for proposer sessions (empty: the CLI default)"
    )
    parser.add_argument("--library-dir", type=Path, default=config.RUNS / "decisions" / "library")
    parser.add_argument("--audit-dir", type=Path, default=config.RUNS / "decisions" / "audit")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true", help="process at most one job, then exit")
    return parser


def _configure_logging() -> None:
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging()
    client = DecisionClient(args.decision_url, args.worker_id)
    proposer = make_proposer(args.model, args.audit_dir)
    library = ExperienceLibrary(args.library_dir)
    log.info("worker %s polling %s", args.worker_id, args.decision_url)
    try:
        while True:
            try:
                outcome = run_once(client, proposer, library, args.k)
            except ServiceError as exc:
                log.warning("decision service unavailable: %s", exc)
                if args.once:
                    return 1
                outcome = "idle"
            if args.once:
                print(outcome)
                return 0
            if outcome == "idle":
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
