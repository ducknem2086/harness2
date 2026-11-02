"""Per-workflow experience library: what earlier improvement jobs learned.

One JSON line per finished job, in `<root>/<slug>.jsonl`. The proposer reads the
most recent lessons as `brief/experience.md` so it does not repeat attempts that
already failed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class ExperienceLibrary:
    def __init__(self, root: Path):
        self.root = Path(root)

    def notes(self, workflow_key: str, limit: int = 10) -> str:
        """The most recent lessons for the workflow as text, newest first ("" when none)."""
        path = self._path(workflow_key)
        if limit <= 0 or not path.is_file():
            return ""
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a torn or hand-edited line must not hide the others
            if isinstance(entry, dict):
                entries.append(entry)
        return "".join(_format(entry) for entry in reversed(entries[-limit:]))

    def record(self, workflow_key: str, entry: dict) -> None:
        path = self._path(workflow_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _path(self, workflow_key: str) -> Path:
        slug = _UNSAFE.sub("", workflow_key) or "_"
        return self.root / f"{slug}.jsonl"


def _format(entry: dict) -> str:
    lines = [
        f"## {entry.get('jobId', '?')} (từ phiên bản {entry.get('baseVersion', '?')}): "
        f"{entry.get('outcome', '?')}, {_percent(entry.get('baselineAccuracy'))} -> "
        f"{_percent(entry.get('bestAccuracy'))}"
    ]
    changes = entry.get("changes")
    if changes:
        lines.append("Đã sửa: " + ", ".join(str(c) for c in changes))
    if entry.get("summary"):
        lines.append(str(entry["summary"]))
    return "\n".join(lines) + "\n\n"


def _percent(value: object) -> str:
    return f"{value:.0%}" if isinstance(value, (int, float)) else "?"
