"""Codex worker configuration and complete proposer failure reports, without inference."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from harness2.cli import decision_worker
from harness2.core.proposer import ProposerError
from harness2.decisions.client import DecisionClient, ServiceError
from harness2.decisions.improve import improve
from harness2.decisions.library import ExperienceLibrary
from harness2.decisions.proposer import make_proposer, propose
from tests.decisions_fakes import FakeDecisionService, FakeProposer
from tests.test_decisions import STRONG, brainstorm_aware, improve_payload


class CodexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="codex-decisions-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_codex_session_edits_surface_with_utf8_and_selected_login_home(self):
        seen = {}

        def fake_run(command, **kwargs):
            seen.update(command=command, **kwargs)
            workspace = Path(kwargs["cwd"])
            (workspace / "context/surface/routing.md").write_text("Đã sửa tuyến", encoding="utf-8")
            Path(command[command.index("--output-last-message") + 1]).write_text("Đã sửa.", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout='{"type":"turn.completed"}\n', stderr="full diagnostics")

        with patch.dict(os.environ, {"CODEX_HOME": "ignored", "OPENAI_BASE_URL": "inherited-custom-endpoint"}), patch("harness2.decisions.proposer.subprocess.run", fake_run):
            proposer = make_proposer("codex-test", self.tmp / "audit", proposer="codex", codex_home=self.tmp / "login")
            edited = propose(proposer, {"surface/routing.md": "cũ"}, "Sửa tuyến", "codex-job")
        self.assertEqual(edited, {"surface/routing.md": "Đã sửa tuyến"})
        self.assertEqual(proposer.spec.kind, "codex_cli")
        self.assertEqual(seen["input"], "Sửa tuyến")
        self.assertEqual(seen["encoding"], "utf-8")
        self.assertEqual(seen["env"]["CODEX_HOME"], str(self.tmp / "login"))
        self.assertEqual(seen["env"]["OPENAI_BASE_URL"], "inherited-custom-endpoint")
        self.assertEqual(seen["command"][seen["command"].index("--model") + 1], "codex-test")
        self.assertEqual(seen["command"][seen["command"].index("--sandbox") + 1], "workspace-write")
        self.assertIn('approval_policy="never"', seen["command"])
        self.assertEqual(seen["command"][-1], "-")
        self.assertTrue(proposer._calls[0].audit.is_file())

    def test_codex_default_model_uses_inherited_home_and_keeps_local_login_untouched(self):
        home = self.tmp / "login"
        home.mkdir()
        auth = home / "auth.json"
        auth.write_bytes(b"do not read or change this file")
        seen = {}

        def fake_run(command, **kwargs):
            seen.update(command=command, **kwargs)
            Path(command[command.index("--output-last-message") + 1]).write_text("done", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.dict(os.environ, {"CODEX_HOME": str(home)}), patch("harness2.decisions.proposer.subprocess.run", fake_run):
            proposer = make_proposer("", self.tmp / "audit", proposer="codex")
            propose(proposer, {"surface/routing.md": "original"}, "mission", "default-model")
        self.assertNotIn("--model", seen["command"])
        self.assertEqual(seen["env"]["CODEX_HOME"], str(home))
        self.assertEqual(auth.read_bytes(), b"do not read or change this file")

    def test_codex_nonzero_exit_rejects_partial_answer_and_preserves_full_process_error(self):
        diagnostic = "failure " + "x" * 7000

        def fake_run(command, **kwargs):
            Path(command[command.index("--output-last-message") + 1]).write_text("partial", encoding="utf-8")
            return subprocess.CompletedProcess(command, 2, stdout="all stdout", stderr=diagnostic)

        proposer = make_proposer("", self.tmp / "audit", proposer="codex")
        proposer.attempts = 1
        with patch("harness2.decisions.proposer.subprocess.run", fake_run), self.assertRaises(ProposerError) as caught:
            propose(proposer, {"surface/routing.md": "original"}, "mission", "broken")
        self.assertIn(diagnostic, str(caught.exception))
        call = proposer._calls[-1]
        attempts = json.loads(next((self.tmp / "audit").glob("*.attempts.json")).read_text(encoding="utf-8"))
        self.assertEqual(attempts[0]["stderr"], diagnostic)
        self.assertEqual(attempts[0]["stdout"], "all stdout")
        self.assertEqual(attempts[0]["returncode"], 2)
        self.assertFalse(call.ok)

    def test_cli_passes_selected_provider_and_home_to_the_real_proposer(self):
        captured = []

        def idle(_client, proposer, _library, _k):
            captured.append(proposer)
            return "idle"

        with patch.object(decision_worker, "run_once", idle), contextlib.redirect_stdout(io.StringIO()):
            result = decision_worker.main(["--once", "--proposer", "codex", "--codex-home", str(self.tmp), "--audit-dir", str(self.tmp / "audit"), "--library-dir", str(self.tmp / "library")])
        self.assertEqual(result, 0)
        self.assertEqual(captured[0].spec.kind, "codex_cli")
        self.assertEqual(captured[0].env()["CODEX_HOME"], str(self.tmp))

    def test_unknown_provider_is_rejected_without_starting_a_session(self):
        with self.assertRaises(ValueError):
            make_proposer("", self.tmp, proposer="unknown")

    def test_timeout_retains_partial_utf8_process_output(self):
        proposer = make_proposer("", self.tmp / "audit", proposer="codex", timeout=1)
        proposer.attempts = 1
        timeout = subprocess.TimeoutExpired("codex", 1, output="đã bắt đầu".encode(), stderr=b"timed out")
        with patch("harness2.decisions.proposer.subprocess.run", side_effect=timeout), self.assertRaises(ProposerError):
            propose(proposer, {"surface/routing.md": "original"}, "mission", "timeout")
        attempts = json.loads(next((self.tmp / "audit").glob("*.attempts.json")).read_text(encoding="utf-8"))
        self.assertEqual(attempts[0]["stdout"], "đã bắt đầu")
        self.assertEqual(attempts[0]["stderr"], "timed out")
        self.assertIn("timeout", attempts[0]["error"])

    def test_json_stream_fallback_selects_final_completed_answer(self):
        proposer = make_proposer("", self.tmp / "audit", proposer="codex")
        stream = "\n".join(json.dumps(event) for event in [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Starting work"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Final answer"}},
            {"type": "turn.completed", "usage": {}},
        ])
        with patch("harness2.decisions.proposer.subprocess.run", return_value=subprocess.CompletedProcess("codex", 0, stdout=stream, stderr="")):
            propose(proposer, {"surface/routing.md": "original"}, "mission", "fallback")
        self.assertEqual(proposer._calls[-1].text, "Final answer")

    def test_windows_npm_launcher_runs_a_local_fake_without_shell_expansion(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for the npm launcher integration test")
        prefix = self.tmp / "npm prefix"
        entry = prefix / "node_modules/@openai/codex/bin/codex.js"
        entry.parent.mkdir(parents=True)
        entry.write_text("""const fs = require('node:fs');
const args = process.argv.slice(2);
fs.writeFileSync('context/surface/routing.md', 'Đã sửa bằng CLI');
fs.writeFileSync(args[args.indexOf('--output-last-message') + 1], JSON.stringify(args));
console.log(JSON.stringify({type:'turn.completed'}));
""", encoding="utf-8")
        which = shutil.which
        model = 'model with spaces & $() | "quotes"'
        with patch("harness2.decisions.proposer.shutil.which", side_effect=lambda name: str(prefix / "codex.cmd") if name == "codex" else which(name)):
            proposer = make_proposer(model, self.tmp / "audit", proposer="codex")
            result = propose(proposer, {"surface/routing.md": "original"}, "mission", "npm")
        self.assertEqual(result["surface/routing.md"], "Đã sửa bằng CLI")
        args = json.loads(proposer._calls[-1].text)
        self.assertEqual(args[args.index("--model") + 1], model)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="decision-reports-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.service = FakeDecisionService().start()
        self.addCleanup(self.service.stop)
        self.service.answers = brainstorm_aware
        self.client = DecisionClient(self.service.url, "report-worker")
        self.library = ExperienceLibrary(self.tmp / "library")

    def test_retry_audit_keeps_each_attempt_and_complete_context(self):
        proposer = FakeProposer(edits=[RuntimeError("first error"), STRONG], audit_dir=self.tmp / "audit")
        proposer.attempts = 2
        with patch("harness2.core.proposer.time.sleep"):
            propose(proposer, {"surface/routing.md": "original", "brief/failures.md": "all input"}, "mission", "retry")
        call = proposer._calls[-1]
        self.assertEqual(len(call.attempt_details), 2)
        self.assertIn("first error", call.attempt_details[0]["error"])
        self.assertEqual(call.attempt_details[1]["text"], "ok")
        inputs = json.loads(next((self.tmp / "audit").glob("*.context.json")).read_text(encoding="utf-8"))
        self.assertEqual(inputs["brief/failures.md"], "all input")

    def test_failed_attempt_is_persisted_before_retry_starts(self):
        proposer = FakeProposer(edits=[RuntimeError("first error"), STRONG], audit_dir=self.tmp / "audit")
        proposer.attempts = 2
        observed = []
        original = proposer._claude_cli

        def session(*args):
            if proposer.sessions:
                files = list((self.tmp / "audit").glob("*.attempts.json"))
                observed.extend(json.loads(files[0].read_text(encoding="utf-8")) if files else [])
            return original(*args)

        with patch.object(proposer, "_claude_cli", session), patch("harness2.core.proposer.time.sleep"):
            propose(proposer, {"surface/routing.md": "original"}, "mission", "durable")
        self.assertEqual(len(observed), 1)
        self.assertIn("first error", observed[0]["error"])

    def test_concurrent_calls_do_not_mix_process_diagnostics(self):
        proposer = make_proposer("model", self.tmp / "audit")
        barrier = threading.Barrier(2)

        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=kwargs["input"], stderr="")

        def session(cwd, mission, system, wall):
            result = proposer._run_process(["fake"], input=mission)
            barrier.wait(timeout=5)
            return result.stdout, 0.0, ""

        with patch.object(proposer, "_session", session), patch("harness2.core.proposer.subprocess.run", run), patch("harness2.core.proposer._SEM", None), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(proposer.call, "first reply", None, "one")
            second = pool.submit(proposer.call, "second reply", None, "two")
            replies = [first.result(), second.result()]
        for reply in replies:
            self.assertEqual(reply.attempt_details[0]["stdout"], reply.text)

    def test_successful_report_includes_failed_attempts_and_candidate_files(self):
        self.service.queue(improve_payload())
        job = self.client.claim()
        proposer = FakeProposer(edits=[RuntimeError("full error"), STRONG], audit_dir=self.tmp / "audit")
        outcome = improve(job, self.client, proposer, self.library, k=2)
        self.assertEqual(outcome.kind, "succeeded")
        self.assertEqual(len(outcome.report["proposerCalls"]), 2)
        call = outcome.report["proposerCalls"][0]
        self.assertIn("full error", call["error"])
        self.assertTrue(Path(call["artifacts"]["context"]).is_file())
        self.assertEqual(outcome.report["candidates"][1]["files"], STRONG | {"surface/routing.md": job.payload["decisions"]["questions"][0]["instructions"], "surface/choices/codex.md": job.payload["decisions"]["questions"][0]["choices"][1]["description"]})
        self.assertTrue(outcome.report["candidates"][1]["results"])

    def test_failed_result_round_trip_keeps_report_and_legacy_call(self):
        self.service.queue(improve_payload())
        job = self.client.claim()
        report = {"error": {"message": "x" * 7000}, "proposerCalls": [{"attempts": 2}]}
        self.client.fail(job.id, "reason", report)
        self.assertEqual(self.service.results[-1]["report"], report)
        self.service.queue(improve_payload())
        self.client.fail(self.client.claim().id, "legacy")
        self.assertNotIn("report", self.service.results[-1])

    def test_scoring_outage_retains_partial_candidates_and_proposer_audits(self):
        self.service.queue(improve_payload())
        proposer = FakeProposer(edits=[STRONG], audit_dir=self.tmp / "audit")
        with patch.object(self.client, "evaluate", side_effect=ServiceError("score unavailable")), self.assertLogs(decision_worker.log, level="WARNING"):
            result = decision_worker.run_once(self.client, proposer, self.library, k=1)
        self.assertEqual(result, "failed")
        report = self.service.results[-1]["report"]
        self.assertIn("score unavailable", report["error"]["message"])
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(report["proposerCalls"][0]["text"], "ok")
        self.assertEqual(report["input"]["workflowKey"], "wf-1")

    def test_result_post_failure_retains_full_completed_report(self):
        self.service.queue(improve_payload())
        proposer = FakeProposer(edits=[STRONG], audit_dir=self.tmp / "audit")
        with patch.object(self.client, "succeed", side_effect=ServiceError("save unavailable")), self.assertLogs(decision_worker.log, level="WARNING"):
            result = decision_worker.run_once(self.client, proposer, self.library, k=1)
        self.assertEqual(result, "failed")
        report = self.service.results[-1]["report"]
        self.assertEqual(report["bestAccuracy"], 1.0)
        self.assertEqual(report["proposerCalls"][0]["text"], "ok")
        self.assertEqual(report["candidates"][0]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
