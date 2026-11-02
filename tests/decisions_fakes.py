"""In-process fake of the decision service's harness-job API, for tests."""

from __future__ import annotations

import copy
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

LEASE_UNTIL = "2099-01-01T00:00:00.000Z"


def payload(**overrides) -> dict:
    """A claim payload shaped like spec §7.1. Keyword overrides replace top-level keys."""
    base = {
        "jobId": "job-placeholder",  # the fake replaces this with the real job id
        "workflowKey": "wf-1",
        "baseVersion": 3,
        "routingQuestion": "route",
        "providers": ["claude", "codex"],
        "decisions": {
            "input": "{{input}}",
            "questions": [
                {
                    "name": "route",
                    "type": "choice",
                    "instructions": "Pick the provider best suited to the task.",
                    "choices": [
                        {"value": "claude", "description": "Careful reasoning and edits."},
                        {"value": "codex", "description": "Fast code generation."},
                    ],
                },
                {"name": "confidence", "type": "number", "instructions": "How sure are you, 0 to 1."},
            ],
        },
        "testCases": [
            {
                "id": "tc-1",
                "input": "Refactor this module and explain the trade-offs.",
                "expected": "claude",
                "assertion": {"operator": "equals", "expected_value": "claude"},
                "origin": "user",
            },
            {
                "id": "tc-2",
                "input": "Generate a quick script.",
                "expected": None,  # `in` assertions carry a list instead of one expected value
                "assertion": {"operator": "in", "expected_value": ["codex", "claude"]},
                "origin": "self-route",
            },
        ],
        "baseline": {
            "accuracy": 0.5,
            "results": [
                {"testCaseId": "tc-1", "actual": "codex", "pass": False},
                {"testCaseId": "tc-2", "actual": "codex", "pass": True},
            ],
        },
        "reports": [
            {
                "decisionId": "d-1",
                "kind": "user",
                "input": "Refactor this module and explain the trade-offs.",
                "answered": "codex",
                "correct": "claude",
                "note": "needs careful reasoning",
            }
        ],
        "stats": {"decisions": 12, "reports": 1},
        "dropped": {"reports": 0},
    }
    base.update(overrides)
    return base


def _passes(assertion: dict, actual: str) -> bool:
    operator = assertion.get("operator")
    expected = assertion.get("expected_value")
    if operator == "equals":
        return actual == expected
    if operator == "in":
        return actual in expected
    raise ValueError(f"unsupported assertion operator: {operator!r}")


def _default_answers(questions: list[dict], test_case: dict) -> str:
    """Answer correctly: the expected value, or the first allowed one for `in`."""
    expected = test_case["assertion"]["expected_value"]
    return expected[0] if isinstance(expected, list) else expected


class FakeDecisionService:
    """Context manager serving the harness-job endpoints on an ephemeral port.

    `answers(questions, test_case)` decides each test case's `actual`.
    `results` records the body of every accepted `result` call.
    `invalid_detail` (when set) makes `evaluate` answer 400 candidate-invalid.
    `server_error` makes every call answer 500.
    """

    def __init__(self) -> None:
        self.answers: Callable[[list[dict], dict], str] = _default_answers
        self.results: list[dict] = []
        self.invalid_detail: str | None = None
        self.server_error = False
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._payloads: dict[str, dict] = {}
        self._claimed_by: dict[str, str] = {}
        self._lost: set[str] = set()
        self._next_id = 1
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def queue(self, job_payload: dict) -> str:
        with self._lock:
            job_id = f"job-{self._next_id}"
            self._next_id += 1
            stored = copy.deepcopy(job_payload)
            stored["jobId"] = job_id
            self._payloads[job_id] = stored
            self._queue.append(job_id)
            return job_id

    def lose_lease(self, job_id: str) -> None:
        with self._lock:
            self._lost.add(job_id)

    def start(self) -> "FakeDecisionService":
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # keep test output quiet
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    return self._send(400, {"error": "bad-json"})
                status, response = service._handle(self.path, body)
                self._send(status, response)

            def _send(self, status: int, response: dict | None):
                data = b"" if response is None else json.dumps(response).encode("utf-8")
                self.send_response(status)
                if data:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "FakeDecisionService":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- request handling -------------------------------------------------

    def _handle(self, path: str, body: dict) -> tuple[int, dict | None]:
        if self.server_error:
            return 500, {"error": "boom"}
        worker = body.get("worker")
        with self._lock:
            if path == "/api/harness/jobs/claim":
                if not self._queue:
                    return 204, None
                job_id = self._queue.pop(0)
                self._claimed_by[job_id] = worker
                return 200, {
                    "jobId": job_id,
                    "leaseUntil": LEASE_UNTIL,
                    "payload": copy.deepcopy(self._payloads[job_id]),
                }
            if path == "/api/harness/evaluate":
                job_id = body.get("jobId")
                if not self._owns(job_id, worker):
                    return 409, {"error": "lease-lost"}
                if self.invalid_detail is not None:
                    return 400, {"error": "candidate-invalid", "detail": self.invalid_detail}
                return 200, self._evaluate(job_id, body.get("questions", []))
            match = re.fullmatch(r"/api/harness/jobs/([^/]+)/(heartbeat|result)", path)
            if match:
                job_id, action = match.groups()
                if not self._owns(job_id, worker):
                    return 409, {"error": "lease-lost"}
                if action == "heartbeat":
                    return 200, {"leaseUntil": LEASE_UNTIL}
                return 200, self._record_result(job_id, body)
        return 404, {"error": "not-found"}

    def _owns(self, job_id: str | None, worker: str | None) -> bool:
        return (
            job_id in self._claimed_by
            and self._claimed_by[job_id] == worker
            and job_id not in self._lost
        )

    def _evaluate(self, job_id: str, questions: list[dict]) -> dict:
        cases = self._payloads[job_id]["testCases"]
        results = []
        for case in cases:
            actual = self.answers(questions, case)
            results.append(
                {"testCaseId": case["id"], "actual": actual, "pass": _passes(case["assertion"], actual)}
            )
        accuracy = sum(1 for r in results if r["pass"]) / len(results) if results else 0.0
        return {"accuracy": accuracy, "results": results}

    def _record_result(self, job_id: str, body: dict) -> dict:
        self.results.append(copy.deepcopy(body))
        outcome = body.get("outcome")
        if outcome == "succeeded":
            return {"status": "succeeded", "version": self._payloads[job_id]["baseVersion"] + 1}
        return {"status": outcome}
