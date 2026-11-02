"""In-process fake of the decision service's harness-job API, for tests."""

from __future__ import annotations

import copy
import json
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from harness2.core.spec import ChannelSpec
from harness2.decisions.proposer import LocalClaudeProposer

LEASE_UNTIL = "2099-01-01T00:00:00.000Z"

# What JavaScript's trim() removes; the real service compares text against its trimmed self.
_JS_SPACE = "\t\n\v\f\r " + "".join(
    chr(code)
    for code in (0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF)
)
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_]{1,64}$")
_LIST_KEY = {"predicate": None, "choice": "choices", "score": "levels"}
_LABEL_KEY = {"choices": "value", "levels": "label"}


def _text_problem(value: object, *, single_line: bool, required: bool) -> str | None:
    """Why the real service refuses a text field (scenarioTransfer.ts), or None."""
    if not isinstance(value, str):
        return "must be a string"
    if "\r" in value or "\x00" in value:
        return "must not contain CR or NUL"
    if value != value.strip(_JS_SPACE):
        return "must not start or end with whitespace"
    if required and not value:
        return "must not be empty"
    if single_line and "\n" in value:
        return "must be a single line"
    return None


def _question_problem(question: object) -> str | None:
    """Structural rules of the service's `questionSchema(false)`."""
    if not isinstance(question, dict):
        return "must be an object"
    kind = question.get("type")
    if kind not in _LIST_KEY:
        return "type: must be predicate, choice or score"
    list_key = _LIST_KEY[kind]
    allowed = {"name", "type", "instructions"} | ({list_key} if list_key else set())
    if set(question) - allowed:
        return f"unexpected keys {sorted(set(question) - allowed)}"
    name = question.get("name")
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        return "name: invalid identifier"
    problem = _text_problem(question.get("instructions"), single_line=False, required=True)
    if problem:
        return f"instructions: {problem}"
    if list_key is None:
        return None
    items = question.get(list_key)
    if not isinstance(items, list) or len(items) < 2:
        return f"{list_key}: need at least 2"
    label = _LABEL_KEY[list_key]
    for position, item in enumerate(items):
        at = f"{list_key}.{position}"
        if not isinstance(item, dict) or label not in item or set(item) - {label, "description"}:
            return f"{at}: must have {label} and optionally description"
        problem = _text_problem(item[label], single_line=True, required=True)
        if problem:
            return f"{at}.{label}: {problem}"
        problem = _text_problem(item.get("description", ""), single_line=True, required=False)
        if problem:
            return f"{at}.description: {problem}"
    return None


def check_candidate(base_questions: list[dict], routing: str, raw: object) -> str | None:
    """Mirror of the service's `checkCandidate`: None when acceptable, else the detail."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= 400:
        return "questions: need between 1 and 400 questions"
    # The service ignores thresholds, so a worker's copy may omit them.
    questions = [
        {k: v for k, v in q.items() if k != "threshold"} if isinstance(q, dict) else q for q in raw
    ]
    for index, question in enumerate(questions):
        problem = _question_problem(question)
        if problem:
            return f"{index}.{problem}"
    routing_questions = [q for q in questions if q["name"] == routing]
    if len(routing_questions) != 1:
        return f"need exactly one routing question {routing!r}"
    if routing_questions[0]["type"] != "choice":
        return "the routing question must stay type choice"
    values = [c["value"] for c in routing_questions[0]["choices"]]
    base_routing = next(q for q in base_questions if q.get("name") == routing)
    if len(set(values)) != len(values) or sorted(values) != sorted(
        c["value"] for c in base_routing["choices"]
    ):
        return "the routing question's choice values must stay the same"

    def others(question_list: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in q.items() if k != "threshold"}
            for q in question_list
            if q.get("name") != routing
        ]

    if others(questions) != others(base_questions):
        return "the other questions must stay the same"
    return None


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
                {
                    "name": "confidence",
                    "type": "score",
                    "instructions": "How sure are you.",
                    "levels": [
                        {"label": "low", "description": "Mostly guessing."},
                        {"label": "high", "description": ""},
                    ],
                },
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
    `evaluate` and a `succeeded` result check the candidate like the service's
    `checkCandidate` (trimmed text, single-line descriptions, same choice values, other
    questions unchanged) and answer 400 candidate-invalid; a `succeeded` result is also
    scored again and answers 400 no-improvement-as-success unless it beats the baseline.
    `invalid_detail` (when set) makes both answer 400 candidate-invalid regardless.
    `engine_error` makes both answer 502 engine-error after the checks.
    `server_error` makes every call answer 500.
    """

    def __init__(self) -> None:
        self.answers: Callable[[list[dict], dict], str] = _default_answers
        self.results: list[dict] = []
        self.invalid_detail: str | None = None
        self.server_error = False
        self.engine_error = False
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
                rejected = self._reject(job_id, body.get("questions"))
                if rejected is not None:
                    return rejected
                return 200, self._evaluate(job_id, body.get("questions", []))
            match = re.fullmatch(r"/api/harness/jobs/([^/]+)/(heartbeat|result)", path)
            if match:
                job_id, action = match.groups()
                if not self._owns(job_id, worker):
                    return 409, {"error": "lease-lost"}
                if action == "heartbeat":
                    return 200, {"leaseUntil": LEASE_UNTIL}
                if body.get("outcome") == "succeeded":
                    rejected = self._reject(job_id, body.get("questions"))
                    if rejected is not None:
                        return rejected
                    # Like the service, score the winner again before publishing it.
                    scored = self._evaluate(job_id, body["questions"])
                    if not scored["accuracy"] > self._payloads[job_id]["baseline"]["accuracy"]:
                        return 400, {"error": "no-improvement-as-success"}
                return 200, self._record_result(job_id, body)
        return 404, {"error": "not-found"}

    def _reject(self, job_id: str, questions: object) -> tuple[int, dict] | None:
        """The 400 or 502 the service would answer for these questions, else None."""
        if self.invalid_detail is not None:
            return 400, {"error": "candidate-invalid", "detail": self.invalid_detail}
        payload = self._payloads[job_id]
        detail = check_candidate(payload["decisions"]["questions"], payload["routingQuestion"], questions)
        if detail is not None:
            return 400, {"error": "candidate-invalid", "detail": detail}
        if self.engine_error:
            return 502, {"error": "engine-error", "detail": "the engine is down"}
        return None

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


class FakeProposer(LocalClaudeProposer):
    """A proposer whose session edits files instead of running the Claude CLI.

    `edit` maps a context-relative key (e.g. "surface/routing.md") to the text the
    session leaves there; every session does the same. `edits` instead gives one
    entry per session, in call order: a dict like `edit`, or an exception to raise
    from that session (the core wraps it in a `ProposerError`). `seen` records the
    bytes of every context file any session found and `sessions` the same per
    session; `labels` lists the label of every call; `events` (when given) gets a
    "session" entry whenever a session starts. `drop_surface=True` simulates a
    session that removed `surface/`. A single attempt per session, so a failing
    session neither retries nor consumes the next entry of `edits`.
    """

    def __init__(
        self,
        edit: dict[str, str] | None = None,
        audit_dir: Path | None = None,
        drop_surface: bool = False,
        edits: list[dict[str, str] | Exception] | None = None,
        events: list[str] | None = None,
    ) -> None:
        super().__init__(ChannelSpec("claude_cli", ""), audit_dir=audit_dir, attempts=1)
        self.edit = edit or {}
        self.edits = edits
        self.drop_surface = drop_surface
        self.events = events
        self.seen: dict[str, bytes] = {}
        self.sessions: list[dict[str, bytes]] = []
        self.labels: list[str] = []

    def call(self, mission, context_dir, label, **kwargs):
        self.labels.append(label)
        return super().call(mission, context_dir, label, **kwargs)

    def _claude_cli(self, cwd: Path, mission: str, system: str, wall: int):
        if self.events is not None:
            self.events.append("session")
        context = cwd / "context"
        session: dict[str, bytes] = {}
        for path in context.rglob("*"):
            if path.is_file():
                session[path.relative_to(context).as_posix()] = path.read_bytes()
        self.seen.update(session)
        self.sessions.append(session)
        edit = self.edit
        if self.edits is not None:
            index = len(self.sessions) - 1
            if index >= len(self.edits):
                raise AssertionError(f"unexpected session {index + 1}: only {len(self.edits)} edits given")
            edit = self.edits[index]
            if isinstance(edit, Exception):
                raise edit
        if self.drop_surface:
            shutil.rmtree(context / "surface", ignore_errors=True)
        for key, text in edit.items():
            target = context / key
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        return "ok", 0.0, ""
