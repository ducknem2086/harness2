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
from harness2.decisions.client import (
    SCORING_TIMEOUT,
    DecisionClient,
    Job,
    LeaseLost,
    ServiceError,
)
from harness2.decisions.improve import Outcome, improve, record
from harness2.decisions.library import ExperienceLibrary
from harness2.decisions.proposer import make_proposer

log = logging.getLogger("harness2.decision_worker")

# One proposer session may run up to 30 minutes against a lease of about the same length,
# so a background heartbeat keeps the lease between `improve`'s own per-step heartbeats.
HEARTBEAT_SECONDS = 300.0
_KEEPER_JOIN_SECONDS = 5.0
# What the service answers when its own re-score finds the winner no better than the baseline.
NOT_AN_IMPROVEMENT = "no-improvement-as-success"


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
    if error is not None:
        return _fail(client, job.id, error)
    return _post_result(client, library, job, outcome)


def _post_result(client: DecisionClient, library: ExperienceLibrary, job: Job, outcome: Outcome) -> str:
    """Send the outcome, then record its lesson under what the service accepted."""
    kind = outcome.kind
    try:
        try:
            _send(client, job.id, outcome)
        except ServiceError as exc:
            if kind != "succeeded" or exc.code != NOT_AN_IMPROVEMENT:
                raise
            # The service re-scored the winner and it did not beat the baseline after all.
            kind = "no-improvement"
            report = {**outcome.report, "serviceRejection": str(exc)}
            client.no_improvement(job.id, report)
    except LeaseLost:
        return _lease_lost(job.id)
    except ServiceError as exc:
        log.warning("job %s: the service did not take the %s result: %s", job.id, kind, exc)
        return _fail(client, job.id, exc)
    try:
        record(library, job, outcome, kind)
    except OSError as exc:
        log.warning("job %s: could not record the lesson: %s", job.id, exc)
    log.info("job %s: %s", job.id, kind)
    return kind


def _send(client: DecisionClient, job_id: str, outcome: Outcome) -> None:
    if outcome.kind == "succeeded":
        client.succeed(job_id, outcome.questions, outcome.report)
    else:
        client.no_improvement(job_id, outcome.report)


def _fail(client: DecisionClient, job_id: str, error: Exception) -> str:
    reason = f"{type(error).__name__}: {error}"
    log.warning("job %s: failed: %s", job_id, reason)
    try:
        client.fail(job_id, reason)
    except LeaseLost:
        return _lease_lost(job_id)
    except ServiceError as exc:
        log.warning("job %s: could not report the failure: %s", job_id, exc)
    return "failed"


def _lease_lost(job_id: str) -> str:
    log.warning("job %s: lease lost; dropping the job without a result", job_id)
    return "lease-lost"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness2-decision-worker",
        description="Improve the routing text of workflows by claiming jobs from the decision service.",
    )
    parser.add_argument("--decision-url", default="http://localhost:3000")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    parser.add_argument(
        "--k", type=_positive_int, default=3, help="independent candidates per job (at least 1)"
    )
    parser.add_argument(
        "--model", default="", help="Claude model for proposer sessions (empty: the CLI default)"
    )
    parser.add_argument("--library-dir", type=Path, default=config.RUNS / "decisions" / "library")
    parser.add_argument("--audit-dir", type=Path, default=config.RUNS / "decisions" / "audit")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument(
        "--http-timeout",
        type=_positive_float,
        default=SCORING_TIMEOUT,
        help="seconds to wait for the service to score candidates (evaluate and the final result)",
    )
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
    client = DecisionClient(args.decision_url, args.worker_id, scoring_timeout=args.http_timeout)
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
