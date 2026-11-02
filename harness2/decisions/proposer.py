"""Proposer sessions for the decision worker, on the locally logged-in Claude CLI.

The core `Proposer` forces Vertex AI for the `claude_cli` channel. The decision
worker instead uses whatever login the Claude CLI already has on this machine,
so `LocalClaudeProposer` never sets a Vertex variable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from harness2.core.proposer import _DROP_ENV, SYSTEM_ARG_MAX, Proposer, ProposerError, _parse_json
from harness2.core.spec import ChannelSpec

_VERTEX_ENV = (
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION",
)

SURFACE_PREFIX = "surface/"
_HARVEST = "context/surface"

MISSION = """\
You are improving how a workflow router picks a provider for a task.

The router reads a routing question and one description per choice, then answers
with one of the choices. Your job is to rewrite that text so the router picks the
right provider for the cases listed in `context/brief/failures.md` without
breaking the cases it already gets right.

Rules:
- Edit only the files under `context/surface/`. `surface/routing.md` is the
  routing question's instructions; `surface/choices/<value>.md` is the description
  of that choice.
- Read everything under `context/brief/` first: `failures.md` lists the cases to
  fix and the cases that must keep passing, `reports.md` lists user corrections,
  `other_questions.md` shows the other questions asked in the same step, and
  `experience.md` holds notes from earlier attempts.
- Do not create, rename or delete any file. Do not touch anything outside
  `context/surface/`.
- Write plain text only. Keep each file a description of when to pick that
  provider, not a list of the test inputs.
- Each file must not start or end with blank space or a blank line (it is
  trimmed before use). Every `surface/choices/<value>.md` must be a single line:
  no line breaks inside it. `surface/routing.md` may span several lines.
- When you are done, reply with one short paragraph saying what you changed and why.
"""


class LocalClaudeProposer(Proposer):
    """Runs the Claude CLI with the local login: no Vertex, model only when given."""

    def __post_init__(self) -> None:
        """The model is optional here; an empty one means the CLI's own default."""

    def env(self) -> dict[str, str]:
        skip = (*_DROP_ENV, *_VERTEX_ENV)
        env = {k: v for k, v in os.environ.items() if k not in skip}
        if self.spec.model.strip():
            env["ANTHROPIC_MODEL"] = self.spec.model
        return env

    def _claude_cli(self, cwd: Path, mission: str, system: str, wall: int):
        """The core session, except that the CLI's output is decoded as UTF-8.

        The core runs `subprocess.run(text=True)`, which decodes with the locale. On a
        Windows code page a Vietnamese reply then fails to decode and the session
        comes back empty.
        """
        cmd = [
            shutil.which("claude") or "claude",
            "--dangerously-skip-permissions",
            "-p",
            "--output-format",
            "json",
            "--effort",
            self.spec.effort,
        ]
        stdin = mission
        if system:
            if len(system) > SYSTEM_ARG_MAX:
                stdin = f"# SYSTEM INSTRUCTIONS\n\n{system}\n\n# END\n\n{mission}"
                cmd += [
                    "--append-system-prompt",
                    "Follow the SYSTEM INSTRUCTIONS block at the top of the message.",
                ]
            else:
                cmd += ["--append-system-prompt", system]
        p = subprocess.run(
            cmd,
            input=stdin,
            cwd=cwd,
            env=self.env(),
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=wall,
        )
        payload = _parse_json(p.stdout)
        text = str(payload.get("result") or payload.get("text") or "")
        err = ""
        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            err = f"CLI error: {payload.get('subtype')} {text[:200]}"
            text = ""
        elif p.returncode != 0 and not text.strip():
            err = f"rc={p.returncode}: {(p.stderr or '')[-300:]}"
        return text, float(payload.get("total_cost_usd") or 0.0), err


def make_proposer(
    model: str, audit_dir: Path, effort: str = "high", timeout: int = 1800
) -> Proposer:
    spec = ChannelSpec("claude_cli", model or "", effort=effort, timeout=timeout)
    return LocalClaudeProposer(spec, audit_dir=audit_dir)


def propose(proposer: Proposer, files: dict[str, str], mission: str, label: str) -> dict[str, str]:
    """Run one session over `files`; return the `surface/` files it left behind.

    Context files are written as raw UTF-8 with no newline translation. Harvested
    files are read as UTF-8 and their line endings normalised to LF (CRLF and a
    lone CR both become LF), because `apply_files` rejects carriage returns and a
    CLI on Windows may write CRLF.
    """
    with tempfile.TemporaryDirectory(prefix="decisions-context-") as tmp:
        context = Path(tmp)
        for key, text in files.items():
            path = context / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        reply = proposer.call(mission, context, label, harvest=_HARVEST)
    workspace = reply.workspace
    if workspace is None or not workspace.is_dir():
        raise ProposerError(f"[{label}] the session left no surface/ directory", attempts=reply.attempts)
    edited: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        key = SURFACE_PREFIX + path.relative_to(workspace).as_posix()
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProposerError(f"[{label}] {key} is not valid UTF-8: {exc}", attempts=reply.attempts) from exc
        edited[key] = text.replace("\r\n", "\n").replace("\r", "\n")
    return edited
