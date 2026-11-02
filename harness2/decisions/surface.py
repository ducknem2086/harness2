"""The editable routing surface of a decision profile, and checks on candidates.

The proposer edits plain files. Only the routing question's `instructions` and
each of its choices' `description` are exposed; everything else in the question
set (names, types, choice values, other questions) is never writable.
File keys always use forward slashes, whatever the OS.
"""

from __future__ import annotations

import copy
import re

from harness2.decisions.client import Evaluation

ROUTING_KEY = "surface/routing.md"
CHOICES_PREFIX = "surface/choices/"
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._-]+$")
# What JavaScript's String.prototype.trim removes, which is how the service checks the text.
_JS_WHITESPACE = "\t\n\v\f\r " + "".join(
    chr(code)
    for code in (0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF)
)


def editable_files(payload: dict) -> dict[str, str]:
    """`surface/routing.md` and one `surface/choices/<value>.md` per choice."""
    routing = _routing_question(payload["decisions"]["questions"], payload["routingQuestion"])
    files = {ROUTING_KEY: routing.get("instructions", "")}
    for choice in routing.get("choices", []):
        value = choice["value"]
        if not _safe_filename(value):
            raise ValueError(f"choice value {value!r} cannot be used as a file name")
        files[f"{CHOICES_PREFIX}{value}.md"] = choice.get("description", "")
    return files


def context_files(payload: dict, evaluation: Evaluation | None, notes: str) -> dict[str, str]:
    """Read-only briefing files for the proposer."""
    questions = payload["decisions"]["questions"]
    results = evaluation.results if evaluation is not None else payload["baseline"]["results"]
    return {
        "brief/failures.md": _failures(payload["testCases"], results),
        "brief/reports.md": _reports(payload.get("reports", [])),
        "brief/other_questions.md": _other_questions(questions, payload["routingQuestion"]),
        "brief/experience.md": notes,
    }


def apply_files(questions: list[dict], routing: str, files: dict[str, str]) -> list[dict]:
    """Copy of `questions` with the edited routing text applied.

    Raises ValueError listing every problem found, so a bad candidate can be
    rejected (and reported) before it is evaluated.
    """
    result = copy.deepcopy(questions)
    question = _routing_question(result, routing)
    choices = {choice["value"]: choice for choice in question.get("choices", [])}
    problems: list[str] = []

    expected_keys = {ROUTING_KEY} | {f"{CHOICES_PREFIX}{value}.md" for value in choices}
    for key in sorted(expected_keys - files.keys()):
        problems.append(f"missing file {key}")
    for key in sorted(k for k in files if k.startswith("surface/") and k not in expected_keys):
        if key.startswith(CHOICES_PREFIX):
            problems.append(f"unknown choice file {key} (choice values cannot be added or renamed)")
        else:
            problems.append(f"unknown file {key}")
    originals = {ROUTING_KEY: question.get("instructions", "")}
    originals.update({f"{CHOICES_PREFIX}{v}.md": c.get("description", "") for v, c in choices.items()})
    texts: dict[str, str] = {}
    for key in sorted(expected_keys & files.keys()):
        text = files[key]
        if text == originals[key]:
            texts[key] = text  # untouched text is always accepted, even if it would not be valid as an edit
            continue
        if not isinstance(text, str):
            problems.append(f"{key} is not text")
            continue
        # The service requires text with no leading or trailing whitespace.
        text = text.strip(_JS_WHITESPACE)
        if not text:
            problems.append(f"{key} is empty")
        elif "\r" in text or "\x00" in text:
            problems.append(f"{key} contains a CR or NUL character")
        elif key.startswith(CHOICES_PREFIX) and "\n" in text:
            problems.append(f"{key} must be a single line (a choice description cannot contain a newline)")
        texts[key] = text
    if problems:
        raise ValueError("; ".join(problems))

    question["instructions"] = texts[ROUTING_KEY]
    for value, choice in choices.items():
        choice["description"] = texts[f"{CHOICES_PREFIX}{value}.md"]
    return result


def changed(before: list[dict], after: list[dict], routing: str) -> list[str]:
    """Short descriptions of what differs on the routing surface."""
    old = _routing_question(before, routing)
    new = _routing_question(after, routing)
    notes = []
    if old.get("instructions") != new.get("instructions"):
        notes.append("routing instructions")
    old_choices = {c["value"]: c for c in old.get("choices", [])}
    for choice in new.get("choices", []):
        previous = old_choices.get(choice["value"])
        if previous is None or previous.get("description") != choice.get("description"):
            notes.append(f"choice {choice['value']} description")
    return notes


def _routing_question(questions: list[dict], routing: str) -> dict:
    for question in questions:
        if question.get("name") == routing:
            return question
    raise ValueError(f"routing question {routing!r} not found")


def _safe_filename(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(_SAFE_VALUE.match(value))
        and not value.endswith(".")  # Windows strips trailing dots
        and value not in (".", "..")
    )


def _failures(test_cases: list[dict], results: list[dict]) -> str:
    by_id = {r["testCaseId"]: r for r in results}
    blocks = []
    for case in test_cases:
        result = by_id.get(case["id"])
        if result is not None and result.get("pass"):
            continue
        actual = result.get("actual") if result is not None else None
        blocks.append(
            "\n".join(
                [
                    f"## {case['id']}",
                    "Input:",
                    case["input"],
                    f"Hiện đang trả lời: {actual if actual is not None else '(chưa có kết quả)'}",
                    f"Đáp án đúng: {_describe_assertion(case)}",
                ]
            )
        )
    if not blocks:
        return "Không có test case nào sai.\n"
    return "# Test case đang sai\n\n" + "\n\n".join(blocks) + "\n"


def _describe_assertion(case: dict) -> str:
    assertion = case.get("assertion") or {}
    operator = assertion.get("operator")
    value = assertion.get("expected_value")
    if operator == "in" and isinstance(value, list):
        return "phải là một trong " + ", ".join(str(v) for v in value)
    if operator == "equals":
        return f"phải là {value}"
    if operator is None:
        return f"phải là {case.get('expected')}"
    return f"{operator} {value}"


def _reports(reports: list[dict]) -> str:
    if not reports:
        return "Chưa có báo cáo nào.\n"
    blocks = []
    for report in reports:
        lines = [
            f"## {report.get('decisionId', '?')} ({report.get('kind', '?')})",
            "Input:",
            str(report.get("input", "")),
            f"Đã trả lời: {report.get('answered')}",
            f"Đáng lẽ là: {report.get('correct')}",
        ]
        if report.get("note"):
            lines.append(f"Ghi chú: {report['note']}")
        blocks.append("\n".join(lines))
    return "# Báo cáo từ người dùng và self-route\n\n" + "\n\n".join(blocks) + "\n"


def _other_questions(questions: list[dict], routing: str) -> str:
    blocks = []
    for question in questions:
        if question.get("name") == routing:
            continue
        lines = [f"## {question.get('name')} ({question.get('type')})", question.get("instructions", "")]
        for choice in question.get("choices", []):
            lines.append(f"- {choice['value']}: {choice.get('description', '')}")
        blocks.append("\n".join(lines))
    if not blocks:
        return "Không có câu hỏi nào khác.\n"
    return "# Các câu khác (chỉ để đọc, không sửa)\n\n" + "\n\n".join(blocks) + "\n"
