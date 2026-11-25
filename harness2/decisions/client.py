"""HTTP client for the decision service's harness-job API (stdlib only)."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

# Scoring runs every test case through a single-worker engine and can take many minutes.
SCORING_TIMEOUT = 1800.0
# The service rejects a failure reason longer than 4000 characters.
MAX_REASON = 3900


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
    """Network failure, 5xx, or a response the client cannot interpret.

    When the service answered with an error, `status` and `code` (its `error` field)
    say what it was; both are None for a network failure.
    """

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class EngineError(ServiceError):
    """The service could not score a candidate (502 engine-error); the service itself is up."""

    def __init__(self, message: str, *, status: int | None = 502):
        super().__init__(message, status=status, code="engine-error")


class DecisionClient:
    def __init__(
        self,
        base_url: str,
        worker: str,
        timeout: float = 120.0,
        scoring_timeout: float = SCORING_TIMEOUT,
    ):
        """`timeout` covers the quick calls; `evaluate` and `succeed` score every test case
        one after another on the service, so they get `scoring_timeout`."""
        self.base_url = base_url.rstrip("/")
        self.worker = worker
        self.timeout = timeout
        self.scoring_timeout = scoring_timeout

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
            timeout=self.scoring_timeout,
        )
        return Evaluation(
            accuracy=float(self._field(body, "accuracy")),
            results=list(self._field(body, "results")),
        )

    def succeed(self, job_id: str, questions: list[dict], report: dict) -> int:
        _, body = self._post(
            f"/api/harness/jobs/{job_id}/result",
            {"worker": self.worker, "outcome": "succeeded", "questions": questions, "report": report},
            timeout=self.scoring_timeout,
        )
        return int(self._field(body, "version"))

    def no_improvement(self, job_id: str, report: dict) -> None:
        self._post(
            f"/api/harness/jobs/{job_id}/result",
            {"worker": self.worker, "outcome": "no-improvement", "report": report},
        )

    def fail(self, job_id: str, reason: str, report: dict | None = None) -> None:
        body = {"worker": self.worker, "outcome": "failed", "reason": reason[:MAX_REASON]}
        if report is not None:
            body["report"] = report
        self._post(
            f"/api/harness/jobs/{job_id}/result",
            body,
        )

    @staticmethod
    def _field(body: dict, key: str):
        try:
            return body[key]
        except (KeyError, TypeError) as exc:
            raise ServiceError(f"response is missing {key!r}: {body!r}") from exc

    def _post(self, path: str, body: dict, timeout: float | None = None) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                return response.status, self._decode(response.read())
        except urllib.error.HTTPError as exc:
            try:
                error_body = self._decode(exc.read())
            except (http.client.HTTPException, OSError) as read_error:
                raise ServiceError(
                    f"{path}: HTTP {exc.code}, body cut short: {read_error!r}"
                ) from read_error
            finally:
                exc.close()
            self._raise_for_error(exc.code, error_body)
        # URLError and timeouts are OSErrors; a response cut short is an HTTPException.
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
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
        if status == 502 and error == "engine-error":
            raise EngineError(f"engine-error: {body.get('detail', '')}")
        raise ServiceError(
            f"HTTP {status}: {body!r}", status=status, code=error if isinstance(error, str) else None
        )
