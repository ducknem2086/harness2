"""Proposer sessions for the decision worker, on locally authenticated CLIs.

The core `Proposer` forces Vertex AI for the `claude_cli` channel. The decision
worker instead uses whatever login the Claude CLI already has on this machine,
so `LocalClaudeProposer` never sets a Vertex variable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness2.core.proposer import SYSTEM_ARG_MAX, Proposer, ProposerError, _DROP_ENV, _parse_json
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
        p = self._run_process(
            cmd,
            input=stdin,
            cwd=cwd,
            env=self.env(),
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=wall,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = _parse_json(p.stdout)
        text = str(payload.get("result") or payload.get("text") or "")
        err = ""
        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            err = f"CLI error: {payload.get('subtype')} {text}"
            text = ""
        elif p.returncode != 0 and not text.strip():
            err = f"rc={p.returncode}: {p.stderr or ''}"
        return text, float(payload.get("total_cost_usd") or 0.0), err


@dataclass
class LocalCodexProposer(Proposer):
    """Use the Codex CLI's local login/configuration without handling credentials."""

    codex_home: Path | None = None

    def __post_init__(self) -> None:
        """An empty model uses the CLI's configured default."""

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.codex_home is not None:
            env["CODEX_HOME"] = str(self.codex_home.expanduser().resolve())
        elif env.get("CODEX_HOME"):
            # The session runs in a temporary directory; preserve relative caller paths.
            env["CODEX_HOME"] = str(Path(env["CODEX_HOME"]).expanduser().resolve())
        return env

    def _session(self, cwd: Path, mission: str, system: str, wall: int):
        cli = shutil.which("codex") or "codex"
        command = [cli]
        if Path(cli).suffix.lower() in (".cmd", ".bat", ".ps1"):
            # Execute npm's JS entry directly: batch wrappers would parse model text
            # through cmd.exe. Native Codex binaries need no launcher.
            entry = Path(cli).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
            if not entry.is_file():
                raise FileNotFoundError(
                    "Codex launcher requires the npm package or a native codex executable"
                )
            command = [shutil.which("node") or "node", str(entry)]
        answer = cwd / "codex-answer.md"
        command += [
            "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
            "--json", "--output-last-message", str(answer),
            "-c", 'approval_policy="never"',
        ]
        if self.spec.model.strip():
            command += ["--model", self.spec.model]
        if self.spec.effort:
            command += ["-c", f"model_reasoning_effort={json.dumps(self.spec.effort)}"]
        command.append("-")
        result = self._run_process(
            command,
            input=(f"# SYSTEM INSTRUCTIONS\n\n{system}\n\n# END\n\n" if system else "") + mission,
            cwd=cwd, env=self.env(), encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=wall,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            error = result.stderr or result.stdout or "Codex session failed"
            return "", 0.0, f"rc={result.returncode}: {error}"
        text = answer.read_text(encoding="utf-8", errors="replace") if answer.is_file() else ""
        fallback = ""
        for line in (result.stdout or "").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") in ("error", "turn.failed"):
                return "", 0.0, json.dumps(event, ensure_ascii=False)
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                fallback = str(item.get("text") or "") or fallback
        text = text or fallback
        error = "" if text.strip() else f"Codex returned no final answer: {result.stderr or ''}"
        return text, 0.0, error


def make_proposer(
    model: str, audit_dir: Path, effort: str = "high", timeout: int = 1800,
    *, proposer: str = "claude", codex_home: Path | str | None = None,
) -> Proposer:
    if proposer == "codex":
        spec = ChannelSpec("codex_cli", model or "", effort=effort, timeout=timeout)
        return LocalCodexProposer(
            spec, audit_dir=audit_dir, codex_home=Path(codex_home) if codex_home else None
        )
    if proposer != "claude":
        raise ValueError(f"Unknown proposer {proposer!r}; choose claude or codex")
    spec = ChannelSpec("claude_cli", model or "", effort=effort, timeout=timeout)
    return LocalClaudeProposer(spec, audit_dir=audit_dir)


def proposer_calls(proposer: Proposer, start: int = 0) -> list[dict]:
    """JSON-safe reports with full attempts and links to durable audit inputs/outputs."""
    calls = []
    for reply in proposer._calls[start:]:
        artifacts = {}
        if reply.audit:
            stem = str(reply.audit).removesuffix(".response.md")
            artifacts = {
                key: f"{stem}.{suffix}" for key, suffix in {
                    "mission": "mission.md", "system": "system.md", "context": "context.json",
                    "response": "response.md", "meta": "meta.json", "attempts": "attempts.json",
                }.items() if Path(f"{stem}.{suffix}").is_file()
            }
        calls.append({
            "label": reply.label, "channel": reply.channel, "model": proposer.spec.model,
            "ok": reply.ok, "text": reply.text, "error": reply.error,
            "attempts": reply.attempts, "attemptDetails": reply.attempt_details,
            "cost_usd": reply.cost_usd, "wall_s": reply.wall_s, "cached": reply.cached,
            "audit": str(reply.audit) if reply.audit else None,
            "workspace": str(reply.workspace) if reply.workspace else None, "artifacts": artifacts,
        })
    return calls


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
