"""HTTP client for the decision service's harness-job API (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    id: str
    lease_until: str
    payload: dict


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    results: list[dict]  # {"testCaseId", "actual", "pass"}


class LeaseLost(Exception):
    """The service no longer considers this worker the owner of the job."""


class CandidateInvalid(Exception):
    """The service rejected a candidate question set."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class ServiceError(Exception):
    """Network failure, 5xx, or a response the client cannot interpret."""


class DecisionClient:
    def __init__(self, base_url: str, worker: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.worker = worker
        self.timeout = timeout

    def claim(self) -> Job | None:
        status, body = self._post("/api/harness/jobs/claim", {"worker": self.worker})
        if status == 204:
            return None
        try:
            return Job(id=body["jobId"], lease_until=body["leaseUntil"], payload=body["payload"])
        except (KeyError, TypeError) as exc:
            raise ServiceError(f"malformed claim response: {body!r}") from exc

    def heartbeat(self, job_id: str) -> str:
        _, body = self._post(f"/api/harness/jobs/{job_id}/heartbeat", {"worker": self.worker})
        return self._field(body, "leaseUntil")

    def evaluate(self, job_id: str, questions: list[dict]) -> Evaluation:
        _, body = self._post(
            "/api/harness/evaluate",
            {"jobId": job_id, "worker": self.worker, "questions": questions},
        )
        return Evaluation(
            accuracy=float(self._field(body, "accuracy")),
            results=list(self._field(body, "results")),
        )

    def succeed(self, job_id: str, questions: list[dict], report: dict) -> int:
        _, body = self._post(
            f"/api/harness/jobs/{job_id}/result",
            {"worker": self.worker, "outcome": "succeeded", "questions": questions, "report": report},
        )
        return int(self._field(body, "version"))

    def no_improvement(self, job_id: str, report: dict) -> None:
        self._post(
            f"/api/harness/jobs/{job_id}/result",
            {"worker": self.worker, "outcome": "no-improvement", "report": report},
        )

    def fail(self, job_id: str, reason: str) -> None:
        self._post(
            f"/api/harness/jobs/{job_id}/result",
            {"worker": self.worker, "outcome": "failed", "reason": reason},
        )

    @staticmethod
    def _field(body: dict, key: str):
        try:
            return body[key]
        except (KeyError, TypeError) as exc:
            raise ServiceError(f"response is missing {key!r}: {body!r}") from exc

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, self._decode(response.read())
        except urllib.error.HTTPError as exc:
            try:
                error_body = self._decode(exc.read())
            finally:
                exc.close()
            self._raise_for_error(exc.code, error_body)
        except (urllib.error.URLError, OSError) as exc:  # includes timeouts
            raise ServiceError(f"{path}: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _decode(raw: bytes) -> dict:
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except ValueError as exc:
            raise ServiceError(f"response is not JSON: {raw[:200]!r}") from exc
        return decoded if isinstance(decoded, dict) else {"value": decoded}

    @staticmethod
    def _raise_for_error(status: int, body: dict) -> None:
        error = body.get("error")
        if status == 409 and error == "lease-lost":
            raise LeaseLost(error)
        if status == 400 and error == "candidate-invalid":
            raise CandidateInvalid(str(body.get("detail", "")))
        raise ServiceError(f"HTTP {status}: {body!r}")
