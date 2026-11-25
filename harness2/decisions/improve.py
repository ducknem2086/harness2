"""One harness job: propose routing text candidates, probe them, compose the best two.

`improve` never posts a result or records a lesson; the caller (the worker) does
both from the returned `Outcome`, once the service has accepted it. `LeaseLost` and
`ServiceError` from the client propagate, except an `EngineError` while scoring one
candidate, which only fails that candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from harness2.core.proposer import Proposer, ProposerError
from harness2.decisions.client import (
    CandidateInvalid,
    DecisionClient,
    EngineError,
    Evaluation,
    Job,
)
from harness2.decisions.library import ExperienceLibrary
from harness2.decisions.proposer import MISSION, SURFACE_PREFIX, propose, proposer_calls
from harness2.decisions.surface import apply_files, changed, context_files, editable_files

NO_NOTES = "Chưa có bài học nào từ các lần trước.\n"

COMPOSE_MISSION = (
    MISSION
    + """
This is a compose session. Two earlier attempts at the same task are under
`context/brief/candidate_a/` and `context/brief/candidate_b/`: each holds the
text it wrote (`routing.md`, `choices/<value>.md`) and `results.md`, which says
which test cases it got right and wrong. The files under `context/surface/` are
still the original text. Write into `context/surface/` one version that keeps the
strengths of both attempts: keep what made each case pass, and do not copy a
change that caused a case to fail.
"""
)


@dataclass
class Outcome:
    kind: Literal["succeeded", "no-improvement"]
    questions: list[dict] | None
    report: dict
    lesson: dict  # what `record` adds to the library once the service has taken the result


@dataclass
class _Candidate:
    label: str
    files: dict[str, str] | None = None
    questions: list[dict] | None = None
    evaluation: Evaluation | None = None
    error: str | None = None
    changes: tuple[str, ...] = ()

    @property
    def accuracy(self) -> float | None:
        return self.evaluation.accuracy if self.evaluation is not None else None


def improve(
    job: Job,
    client: DecisionClient,
    proposer: Proposer,
    library: ExperienceLibrary,
    k: int = 3,
) -> Outcome:
    payload = job.payload
    key = payload["workflowKey"]
    routing = payload["routingQuestion"]
    questions = payload["decisions"]["questions"]
    baseline = float(payload["baseline"]["accuracy"])
    originals = editable_files(payload)
    notes = library.notes(key) or NO_NOTES
    briefing = context_files(payload, None, notes)
    call_start = len(proposer._calls)
    candidates: list[_Candidate] = []

    def attempt(label: str, mission: str, extra: dict[str, str]) -> _Candidate:
        candidate = _Candidate(label)
        candidates.append(candidate)
        client.heartbeat(job.id)
        try:
            candidate.files = propose(proposer, {**originals, **briefing, **extra}, mission, label)
            candidate.questions = apply_files(questions, routing, candidate.files)
        except (ProposerError, ValueError) as exc:
            candidate.error = str(exc)
            return candidate
        candidate.changes = tuple(changed(questions, candidate.questions, routing))
        if not candidate.changes:
            # Re-scoring the original text could "beat" the baseline on noise alone.
            candidate.error = "the session made no change to the routing text"
            return candidate
        client.heartbeat(job.id)
        try:
            candidate.evaluation = client.evaluate(job.id, candidate.questions)
        except CandidateInvalid as exc:
            candidate.error = f"rejected by the decision service: {exc.detail}"
        except EngineError as exc:  # the service could not score this one; the others may still work
            candidate.error = f"the decision service could not score it: {exc}"
        return candidate

    try:
        for i in range(k):
            attempt(f"{job.id}-p{i}", MISSION, {})

        graded = sorted((c for c in candidates if c.evaluation is not None), key=_rank)
        if len(graded) >= 2:
            first, second = graded[0], graded[1]
            extra = {
                **_candidate_files("brief/candidate_a/", first, payload["testCases"]),
                **_candidate_files("brief/candidate_b/", second, payload["testCases"]),
            }
            attempt(f"{job.id}-compose", COMPOSE_MISSION, extra)
    except Exception as exc:
        if candidates:
            candidates[-1].error = f"{type(exc).__name__}: {exc}"
        best = min((c for c in candidates if c.evaluation is not None), key=_rank, default=None)
        exc.report = _report(payload, baseline, candidates, best)
        exc.report["proposerCalls"] = proposer_calls(proposer, call_start)
        raise

    best = min((c for c in candidates if c.evaluation is not None), key=_rank, default=None)
    report = _report(payload, baseline, candidates, best)
    report["proposerCalls"] = proposer_calls(proposer, call_start)
    improved = best is not None and best.accuracy > baseline
    lesson = {
        "jobId": job.id,
        "baseVersion": payload.get("baseVersion"),
        "baselineAccuracy": baseline,
        "bestAccuracy": report["bestAccuracy"],
        "changes": report["changes"],
        "summary": report["summary"],
    }
    if improved:
        return Outcome("succeeded", best.questions, report, lesson)
    return Outcome("no-improvement", None, report, lesson)


def record(library: ExperienceLibrary, job: Job, outcome: Outcome, accepted: str | None = None) -> None:
    """Add the job's lesson to the library, under the outcome the service accepted.

    The caller records only once the service has taken the result, so the library never
    claims a version that was not created. `accepted` overrides `outcome.kind` when the
    service ended the job differently from what `improve` proposed.
    """
    kind = accepted or outcome.kind
    entry = {**outcome.lesson, "outcome": kind}
    if kind != outcome.kind:
        entry["summary"] += f" The service ended the job as {kind}, not {outcome.kind}."
    library.record(job.payload["workflowKey"], entry)


def _rank(candidate: _Candidate) -> tuple[float, int]:
    """Sort key: higher accuracy first, then fewer changed parts. Stable for ties."""
    return (-candidate.accuracy, len(candidate.changes))


def _candidate_files(prefix: str, candidate: _Candidate, test_cases: list[dict]) -> dict[str, str]:
    files = {
        prefix + key.removeprefix(SURFACE_PREFIX): text for key, text in (candidate.files or {}).items()
    }
    files[prefix + "results.md"] = _results(candidate, test_cases)
    return files


def _results(candidate: _Candidate, test_cases: list[dict]) -> str:
    evaluation = candidate.evaluation
    by_id = {r["testCaseId"]: r for r in evaluation.results}
    lines = [f"# Kết quả của {candidate.label}: {evaluation.accuracy:.0%} test case đạt", ""]
    for case in test_cases:
        result = by_id.get(case["id"])
        if result is None:
            continue
        verdict = "ĐẠT" if result.get("pass") else "SAI"
        lines.append(f"- {case['id']}: {verdict}, trả lời {result.get('actual')}")
        lines.append(f"  Input: {case['input']}")
    return "\n".join(lines) + "\n"


def _report(
    payload: dict,
    baseline: float,
    candidates: list[_Candidate],
    best: _Candidate | None,
) -> dict:
    base_failing = [r["testCaseId"] for r in payload["baseline"]["results"] if not r.get("pass")]
    if best is None:
        fixed: list[str] = []
        still_failing = base_failing
        best_accuracy = baseline
        changes: list[str] = []
        summary = f"Không ứng viên nào chấm được; giữ nguyên baseline {baseline:.0%}."
    else:
        failing_now = {r["testCaseId"] for r in best.evaluation.results if not r.get("pass")}
        fixed = [t for t in base_failing if t not in failing_now]
        still_failing = [r["testCaseId"] for r in best.evaluation.results if not r.get("pass")]
        best_accuracy = best.accuracy
        changes = list(best.changes)
        verdict = "vượt baseline" if best_accuracy > baseline else "không vượt baseline"
        summary = (
            f"Ứng viên tốt nhất {best.label} đạt {best_accuracy:.0%} ({verdict} {baseline:.0%}); "
            f"sửa {', '.join(fixed) if fixed else 'không ca nào'}; "
            f"đã đổi {', '.join(changes)}."
        )
    return {
        "input": payload,
        "baselineAccuracy": baseline,
        "bestAccuracy": best_accuracy,
        "candidates": [
            {
                "label": c.label, "accuracy": c.accuracy, "error": c.error,
                "files": c.files, "questions": c.questions,
                "results": c.evaluation.results if c.evaluation else None,
                "changes": list(c.changes),
            } for c in candidates
        ],
        "fixed": fixed,
        "stillFailing": still_failing,
        "changes": changes,
        "summary": summary,
    }
