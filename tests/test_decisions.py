"""Tests for the decision worker: service client, routing surface, proposer sessions."""

import contextlib
import copy
import io
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.cli import decision_worker
from harness2.core.proposer import ProposerError
from harness2.decisions.client import (
    CandidateInvalid,
    DecisionClient,
    Evaluation,
    LeaseLost,
    ServiceError,
)
from harness2.decisions.improve import improve
from harness2.decisions.library import ExperienceLibrary
from harness2.decisions.proposer import MISSION, make_proposer, propose
from harness2.decisions.surface import apply_files, changed, context_files, editable_files
from tests.decisions_fakes import FakeDecisionService, FakeProposer, payload


class ClientTests(unittest.TestCase):
    def test_claim_returns_none_when_no_job(self):
        with FakeDecisionService() as svc:
            self.assertIsNone(DecisionClient(svc.url, "w1").claim())

    def test_claim_evaluate_and_succeed_round_trip(self):
        with FakeDecisionService() as svc:
            job_id = svc.queue(payload())
            client = DecisionClient(svc.url, "w1")
            job = client.claim()
            self.assertEqual(job.id, job_id)
            self.assertTrue(job.lease_until)
            questions = job.payload["decisions"]["questions"]
            evaluation = client.evaluate(job.id, questions)
            self.assertEqual(len(evaluation.results), len(job.payload["testCases"]))
            version = client.succeed(job.id, questions, {"summary": "x"})
            self.assertEqual(version, job.payload["baseVersion"] + 1)
            self.assertEqual(svc.results[-1]["outcome"], "succeeded")
            self.assertEqual(svc.results[-1]["worker"], "w1")
            self.assertEqual(svc.results[-1]["questions"], questions)

    def test_claimed_job_is_not_claimed_twice(self):
        with FakeDecisionService() as svc:
            svc.queue(payload())
            self.assertIsNotNone(DecisionClient(svc.url, "w1").claim())
            self.assertIsNone(DecisionClient(svc.url, "w2").claim())

    def test_heartbeat_returns_lease_until(self):
        with FakeDecisionService() as svc:
            svc.queue(payload())
            client = DecisionClient(svc.url, "w1")
            job = client.claim()
            self.assertTrue(client.heartbeat(job.id))

    def test_lost_lease_raises(self):
        with FakeDecisionService() as svc:
            svc.queue(payload())
            client = DecisionClient(svc.url, "w1")
            job = client.claim()
            svc.lose_lease(job.id)
            with self.assertRaises(LeaseLost):
                client.heartbeat(job.id)
            with self.assertRaises(LeaseLost):
                client.evaluate(job.id, job.payload["decisions"]["questions"])
            with self.assertRaises(LeaseLost):
                client.fail(job.id, "late")
            self.assertEqual(svc.results, [])

    def test_candidate_invalid_carries_detail(self):
        with FakeDecisionService() as svc:
            svc.queue(payload())
            client = DecisionClient(svc.url, "w1")
            job = client.claim()
            svc.invalid_detail = "choice values changed"
            with self.assertRaises(CandidateInvalid) as caught:
                client.evaluate(job.id, job.payload["decisions"]["questions"])
            self.assertEqual(caught.exception.detail, "choice values changed")

    def test_fake_scores_equals_and_in_assertions(self):
        with FakeDecisionService() as svc:
            body = payload()
            svc.queue(body)
            client = DecisionClient(svc.url, "w1")
            job = client.claim()
            questions = job.payload["decisions"]["questions"]
            # Answers the first (equals) case correctly, the second (in) wrongly.
            svc.answers = lambda qs, case: (
                case["assertion"]["expected_value"]
                if case["assertion"]["operator"] == "equals"
                else "nowhere"
            )
            evaluation = client.evaluate(job.id, questions)
            by_id = {r["testCaseId"]: r["pass"] for r in evaluation.results}
            self.assertEqual(by_id, {"tc-1": True, "tc-2": False})
            self.assertEqual(evaluation.accuracy, 0.5)
            # Answers the `in` case with one of the allowed values.
            svc.answers = lambda qs, case: (
                case["assertion"]["expected_value"][-1]
                if case["assertion"]["operator"] == "in"
                else "nowhere"
            )
            evaluation = client.evaluate(job.id, questions)
            by_id = {r["testCaseId"]: r["pass"] for r in evaluation.results}
            self.assertEqual(by_id, {"tc-1": False, "tc-2": True})

    def test_no_improvement_and_fail_post_results(self):
        with FakeDecisionService() as svc:
            svc.queue(payload())
            svc.queue(payload())
            client = DecisionClient(svc.url, "w1")
            first = client.claim()
            client.no_improvement(first.id, {"summary": "none"})
            second = client.claim()
            client.fail(second.id, "proposer crashed")
            self.assertEqual(svc.results[0]["outcome"], "no-improvement")
            self.assertEqual(svc.results[0]["report"], {"summary": "none"})
            self.assertNotIn("questions", svc.results[0])
            self.assertEqual(svc.results[1]["outcome"], "failed")
            self.assertEqual(svc.results[1]["reason"], "proposer crashed")

    def test_server_error_raises_service_error(self):
        with FakeDecisionService() as svc:
            svc.server_error = True
            with self.assertRaises(ServiceError):
                DecisionClient(svc.url, "w1").claim()

    def test_unreachable_service_raises_service_error(self):
        svc = FakeDecisionService()
        svc.start()
        url = svc.url
        svc.stop()
        with self.assertRaises(ServiceError):
            DecisionClient(url, "w1", timeout=2).claim()

    def test_response_cut_short_raises_service_error(self):
        # The service dies mid-response: urllib raises http.client.IncompleteRead, which is
        # an HTTPException rather than an OSError.
        for status in (200, 500):
            with self.subTest(status=status), TruncatingServer(status) as url:
                with self.assertRaises(ServiceError):
                    DecisionClient(url, "w1", timeout=5).claim()


class TruncatingServer:
    """Answers one request with a Content-Length it does not honour, then hangs up."""

    def __init__(self, status: int):
        self.status = status
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.url = f"http://127.0.0.1:{self.listener.getsockname()[1]}"

    def __enter__(self) -> str:
        threading.Thread(target=self._serve, daemon=True).start()
        return self.url

    def __exit__(self, *exc_info) -> None:
        self.listener.close()

    def _serve(self) -> None:
        try:
            conn, _ = self.listener.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            conn.recv(65536)
            conn.sendall(
                f"HTTP/1.1 {self.status} X\r\nContent-Type: application/json\r\n"
                f"Content-Length: 200\r\n\r\n".encode() + b'{"jobId": "j'
            )


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.p = payload()
        self.routing = self.p["routingQuestion"]
        self.questions = self.p["decisions"]["questions"]

    def test_editable_files_expose_only_routing_text(self):
        self.assertEqual(
            editable_files(self.p),
            {
                "surface/routing.md": "Pick the provider best suited to the task.",
                "surface/choices/claude.md": "Careful reasoning and edits.",
                "surface/choices/codex.md": "Fast code generation.",
            },
        )

    def test_round_trip_without_edits_returns_equal_questions(self):
        files = editable_files(self.p)
        self.assertEqual(apply_files(self.questions, self.routing, files), self.questions)

    def test_apply_files_updates_only_routing_text(self):
        files = editable_files(self.p) | {
            "surface/routing.md": "Route by task type.",
            "surface/choices/claude.md": "Phân tích, brainstorm, viết spec.",
        }
        out = apply_files(self.questions, self.routing, files)
        routing = next(q for q in out if q["name"] == self.routing)
        self.assertEqual(routing["instructions"], "Route by task type.")
        self.assertEqual(
            next(c for c in routing["choices"] if c["value"] == "claude")["description"],
            "Phân tích, brainstorm, viết spec.",
        )
        self.assertEqual(
            [q for q in out if q["name"] != self.routing],
            [q for q in self.questions if q["name"] != self.routing],
        )

    def test_apply_files_does_not_mutate_input(self):
        before = copy.deepcopy(self.questions)
        apply_files(self.questions, self.routing, editable_files(self.p) | {"surface/routing.md": "new"})
        self.assertEqual(self.questions, before)

    def test_apply_files_rejects_value_changes(self):
        files = editable_files(self.p)
        files["surface/choices/pi.md"] = "x"
        with self.assertRaisesRegex(ValueError, "pi"):
            apply_files(self.questions, self.routing, files)

    def test_apply_files_lists_every_problem(self):
        files = editable_files(self.p)
        del files["surface/routing.md"]
        del files["surface/choices/codex.md"]
        files["surface/choices/pi.md"] = "x"
        files["surface/choices/claude.md"] = ""
        with self.assertRaises(ValueError) as caught:
            apply_files(self.questions, self.routing, files)
        message = str(caught.exception)
        for part in ("routing.md", "codex", "pi", "claude"):
            self.assertIn(part, message)

    def test_apply_files_rejects_cr_nul_and_blank(self):
        for bad in ("a" + chr(13) + chr(10) + "b", "a" + chr(0) + "b", "   " + chr(10)):
            with self.subTest(bad=bad):
                files = editable_files(self.p) | {"surface/routing.md": bad}
                with self.assertRaisesRegex(ValueError, "routing.md"):
                    apply_files(self.questions, self.routing, files)

    def test_unedited_invalid_originals_are_accepted(self):
        for original in ("", "  ", "a" + chr(13) + chr(10) + "b", "a" + chr(0) + "b"):
            with self.subTest(original=original):
                p = payload()
                question = p["decisions"]["questions"][0]
                question["instructions"] = original
                question["choices"][0]["description"] = original
                files = editable_files(p)
                self.assertEqual(
                    apply_files(p["decisions"]["questions"], p["routingQuestion"], files),
                    p["decisions"]["questions"],
                )

    def test_editing_an_empty_original_is_still_checked(self):
        p = payload()
        p["decisions"]["questions"][0]["choices"][0]["description"] = ""
        files = editable_files(p)
        files["surface/choices/claude.md"] = " "
        with self.assertRaisesRegex(ValueError, "claude"):
            apply_files(p["decisions"]["questions"], p["routingQuestion"], files)
        files["surface/choices/claude.md"] = "Now described."
        out = apply_files(p["decisions"]["questions"], p["routingQuestion"], files)
        self.assertEqual(out[0]["choices"][0]["description"], "Now described.")

    def test_blanking_a_non_empty_description_is_rejected(self):
        files = editable_files(self.p) | {"surface/choices/codex.md": ""}
        with self.assertRaisesRegex(ValueError, "codex"):
            apply_files(self.questions, self.routing, files)

    def test_editable_files_rejects_unsafe_choice_value(self):
        for value in ("../x", "a/b", "a" + chr(92) + "b", "..", "C:x", "a b", "x."):
            with self.subTest(value=value):
                p = payload()
                p["decisions"]["questions"][0]["choices"][0]["value"] = value
                with self.assertRaisesRegex(ValueError, "choice"):
                    editable_files(p)

    def test_context_describes_in_assertions(self):
        p = payload(testCases=[{"id": "t1", "input": "…", "expected": None,
                                "assertion": {"operator": "in", "expected_value": ["codex", "antigravity"]}}])
        text = context_files(p, None, "")["brief/failures.md"]
        self.assertIn("phải là một trong codex, antigravity", text)

    def test_context_failures_use_baseline_without_evaluation(self):
        failures = context_files(self.p, None, "")["brief/failures.md"]
        self.assertIn("Refactor this module", failures)
        self.assertIn("phải là claude", failures)
        self.assertNotIn("Generate a quick script", failures)  # tc-2 passes in the baseline

    def test_context_failures_use_evaluation_when_given(self):
        evaluation = Evaluation(
            accuracy=0.5,
            results=[
                {"testCaseId": "tc-1", "actual": "claude", "pass": True},
                {"testCaseId": "tc-2", "actual": "pi", "pass": False},
            ],
        )
        failures = context_files(self.p, evaluation, "")["brief/failures.md"]
        self.assertNotIn("Refactor this module", failures)
        self.assertIn("Generate a quick script", failures)
        self.assertIn("đang trả lời: pi", failures)
        self.assertIn("phải là một trong codex, claude", failures)

    def test_context_other_files(self):
        files = context_files(self.p, None, "tried X, did not help")
        self.assertEqual(files["brief/experience.md"], "tried X, did not help")
        self.assertIn("needs careful reasoning", files["brief/reports.md"])
        self.assertIn("confidence", files["brief/other_questions.md"])
        self.assertNotIn("Fast code generation", files["brief/other_questions.md"])
        for key in files:
            self.assertNotIn(chr(92), key)

    def test_changed_lists_each_edit(self):
        files = editable_files(self.p) | {
            "surface/routing.md": "new routing",
            "surface/choices/codex.md": "new codex",
        }
        after = apply_files(self.questions, self.routing, files)
        self.assertEqual(
            changed(self.questions, after, self.routing),
            ["routing instructions", "choice codex description"],
        )
        self.assertEqual(changed(self.questions, self.questions, self.routing), [])


class ProposerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="decisions-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_local_env_has_no_vertex_variables(self):
        leaked = {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "p",
            "CLOUD_ML_REGION": "r",
            "ANTHROPIC_API_KEY": "k",
        }
        with patch.dict(os.environ, leaked):
            os.environ.pop("ANTHROPIC_MODEL", None)
            env = make_proposer("", self.tmp).env()
        for name in (*leaked, "ANTHROPIC_MODEL"):
            self.assertNotIn(name, env)

    def test_model_override_sets_anthropic_model(self):
        env = make_proposer("claude-opus-5-5", self.tmp).env()
        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-opus-5-5")
        self.assertNotIn("CLAUDE_CODE_USE_VERTEX", env)

    def test_make_proposer_uses_claude_cli_channel_settings(self):
        proposer = make_proposer("", self.tmp, effort="low", timeout=60)
        self.assertEqual(proposer.spec.kind, "claude_cli")
        self.assertEqual((proposer.spec.effort, proposer.spec.timeout), ("low", 60))
        self.assertEqual(proposer.audit_dir, self.tmp)

    def test_propose_returns_edited_surface_files(self):
        proposer = FakeProposer(edit={"surface/routing.md": "mới"}, audit_dir=self.tmp)
        out = propose(proposer, {"surface/routing.md": "cũ", "brief/failures.md": "x"}, MISSION, "c1")
        self.assertEqual(out["surface/routing.md"], "mới")
        self.assertNotIn("brief/failures.md", out)

    def test_propose_gives_the_session_every_context_file(self):
        proposer = FakeProposer(edit={}, audit_dir=self.tmp)
        files = {"surface/choices/claude.md": "a", "brief/failures.md": "b"}
        out = propose(proposer, files, MISSION, "c1")
        self.assertEqual(set(proposer.seen), set(files))
        self.assertEqual(out, {"surface/choices/claude.md": "a"})

    def test_propose_keeps_newlines_as_written(self):
        text = "line one\nline two\n\nline four\n"
        proposer = FakeProposer(edit={"surface/routing.md": text}, audit_dir=self.tmp)
        files = {"surface/routing.md": text, "brief/failures.md": text}
        out = propose(proposer, files, MISSION, "c1")
        self.assertEqual(out["surface/routing.md"], text)
        self.assertNotIn("\r", out["surface/routing.md"])
        self.assertEqual(set(proposer.seen), set(files))
        for data in proposer.seen.values():
            self.assertNotIn(b"\r", data)

    def test_propose_normalises_harvested_line_endings(self):
        raw = "a\r\nb\rc\n"
        edit = {"surface/routing.md": raw, "surface/choices/claude.md": raw}
        proposer = FakeProposer(edit=edit, audit_dir=self.tmp)
        out = propose(proposer, {"surface/routing.md": "x"}, MISSION, "c1")
        self.assertEqual(out["surface/routing.md"], "a\nb\nc\n")
        self.assertEqual(out["surface/choices/claude.md"], "a\nb\nc\n")

    def test_propose_without_workspace_raises(self):
        proposer = FakeProposer(edit={}, audit_dir=self.tmp, drop_surface=True)
        with self.assertRaises(ProposerError):
            propose(proposer, {"surface/routing.md": "x"}, MISSION, "c1")

    def test_mission_names_the_rules(self):
        for needle in ("context/surface/", "context/brief/", "failures.md"):
            self.assertIn(needle, MISSION)


BRAINSTORM_CASE = {
    "id": "t-brainstorm",
    "input": "Brainstorm names for the new product.",
    "expected": "claude",
    "assertion": {"operator": "equals", "expected_value": "claude"},
    "origin": "user",
}


def brainstorm_aware(questions: list[dict], test_case: dict) -> str:
    """Router stand-in: it gets the brainstorm case right once claude's description mentions it."""
    claude = next(c for c in questions[0]["choices"] if c["value"] == "claude")
    if test_case["id"] == "t-brainstorm" and "brainstorm" not in claude["description"].lower():
        return "codex"
    expected = test_case["assertion"]["expected_value"]
    return expected[0] if isinstance(expected, list) else expected


def improve_payload() -> dict:
    base = payload()
    return payload(
        testCases=[*base["testCases"], BRAINSTORM_CASE],
        baseline={
            "accuracy": 2 / 3,
            "results": [
                {"testCaseId": "tc-1", "actual": "claude", "pass": True},
                {"testCaseId": "tc-2", "actual": "codex", "pass": True},
                {"testCaseId": "t-brainstorm", "actual": "codex", "pass": False},
            ],
        },
    )


WEAK = {"surface/routing.md": "Pick the provider best suited to the task, with care."}
STRONG = {"surface/choices/claude.md": "Careful reasoning, edits and brainstorm sessions."}
STRONG_AND_MORE = {
    "surface/routing.md": "Choose the provider that fits the task.",
    "surface/choices/claude.md": "Careful reasoning, edits and brainstorm sessions.",
}


class RecordingClient(DecisionClient):
    """A client that logs heartbeat and evaluate calls into a shared event list."""

    def __init__(self, *args, events: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events

    def heartbeat(self, job_id):
        self.events.append("heartbeat")
        return super().heartbeat(job_id)

    def evaluate(self, job_id, questions):
        self.events.append("evaluate")
        return super().evaluate(job_id, questions)


class ImproveTests(unittest.TestCase):
    def setUp(self):
        self.audit = Path(tempfile.mkdtemp(prefix="decisions-audit-"))
        self.addCleanup(shutil.rmtree, self.audit, ignore_errors=True)
        self.library = ExperienceLibrary(Path(tempfile.mkdtemp(prefix="decisions-library-")))
        self.addCleanup(shutil.rmtree, self.library.root, ignore_errors=True)
        self.events: list[str] = []
        self.svc = FakeDecisionService().start()
        self.addCleanup(self.svc.stop)
        self.svc.answers = brainstorm_aware
        self.client = RecordingClient(self.svc.url, "w1", events=self.events)

    def claim(self, job_payload: dict | None = None):
        self.svc.queue(job_payload or improve_payload())
        return self.client.claim()

    def proposer(self, edits: list) -> FakeProposer:
        return FakeProposer(edits=edits, audit_dir=self.audit, events=self.events)

    def test_best_candidate_above_baseline_succeeds(self):
        job = self.claim()
        proposer = self.proposer([WEAK, STRONG, WEAK, STRONG_AND_MORE])
        outcome = improve(job, self.client, proposer, self.library, k=3)
        self.assertEqual(outcome.kind, "succeeded")
        report = outcome.report
        self.assertGreater(report["bestAccuracy"], report["baselineAccuracy"])
        self.assertEqual(report["bestAccuracy"], 1.0)
        self.assertIn("t-brainstorm", report["fixed"])
        self.assertEqual(report["stillFailing"], [])
        # compose ties with the strong candidate on accuracy and changes more, so the strong one wins
        self.assertEqual(report["changes"], ["choice claude description"])
        self.assertEqual(
            [c["label"] for c in report["candidates"]],
            [f"{job.id}-p0", f"{job.id}-p1", f"{job.id}-p2", f"{job.id}-compose"],
        )
        self.assertEqual(proposer.labels, [c["label"] for c in report["candidates"]])
        question = outcome.questions[0]
        self.assertEqual(question["choices"][0]["description"], STRONG["surface/choices/claude.md"])
        self.assertEqual(question["instructions"], job.payload["decisions"]["questions"][0]["instructions"])
        self.assertEqual(self.svc.results, [])  # improve only decides; the worker posts

    def test_compose_session_gets_the_two_best_candidates_and_their_results(self):
        job = self.claim()
        proposer = self.proposer([WEAK, STRONG, WEAK, STRONG_AND_MORE])
        improve(job, self.client, proposer, self.library, k=3)
        compose = proposer.sessions[3]
        self.assertEqual(
            compose["brief/candidate_a/choices/claude.md"].decode(), STRONG["surface/choices/claude.md"]
        )
        self.assertEqual(compose["brief/candidate_b/routing.md"].decode(), WEAK["surface/routing.md"])
        self.assertIn("ĐẠT", compose["brief/candidate_a/results.md"].decode())
        results_b = compose["brief/candidate_b/results.md"].decode()
        self.assertIn("SAI", results_b)
        self.assertIn("t-brainstorm", results_b)
        # the editable surface the compose session starts from is the original text
        self.assertEqual(compose["surface/choices/claude.md"].decode(), "Careful reasoning and edits.")

    def test_no_candidate_beats_baseline(self):
        job = self.claim()
        outcome = improve(job, self.client, self.proposer([WEAK, WEAK, WEAK, WEAK]), self.library, k=3)
        self.assertEqual(outcome.kind, "no-improvement")
        self.assertIsNone(outcome.questions)
        self.assertEqual(outcome.report["bestAccuracy"], outcome.report["baselineAccuracy"])
        self.assertEqual(outcome.report["stillFailing"], ["t-brainstorm"])

    def test_proposer_failure_skips_candidate(self):
        job = self.claim()
        proposer = self.proposer([ProposerError("x"), STRONG, WEAK, WEAK])
        outcome = improve(job, self.client, proposer, self.library, k=3)
        self.assertEqual(outcome.kind, "succeeded")
        first = outcome.report["candidates"][0]
        self.assertIsNone(first["accuracy"])
        self.assertIn("x", first["error"])
        self.assertIsNone(outcome.report["candidates"][1]["error"])

    def test_candidate_that_adds_a_choice_is_not_evaluated(self):
        job = self.claim()
        bad = {"surface/choices/gemini.md": "A new provider."}
        outcome = improve(job, self.client, self.proposer([bad, bad]), self.library, k=2)
        self.assertEqual(outcome.kind, "no-improvement")
        self.assertNotIn("evaluate", self.events)
        for candidate in outcome.report["candidates"]:
            self.assertIn("gemini", candidate["error"])

    def test_candidate_rejected_by_the_service_is_reported(self):
        job = self.claim()
        self.svc.invalid_detail = "choice values changed"
        outcome = improve(job, self.client, self.proposer([STRONG, STRONG]), self.library, k=2)
        self.assertEqual(outcome.kind, "no-improvement")
        self.assertIn("choice values changed", outcome.report["candidates"][0]["error"])
        self.assertEqual(len(outcome.report["candidates"]), 2)  # nothing graded, so no compose

    def test_unchanged_candidate_is_not_scored_against_the_baseline(self):
        job = self.claim()
        outcome = improve(job, self.client, self.proposer([{}, {}]), self.library, k=2)
        self.assertEqual(outcome.kind, "no-improvement")
        self.assertNotIn("evaluate", self.events)
        self.assertIn("no change", outcome.report["candidates"][0]["error"])

    def test_tie_prefers_the_candidate_with_fewer_changes(self):
        job = self.claim()
        proposer = self.proposer([STRONG_AND_MORE, STRONG, WEAK])
        outcome = improve(job, self.client, proposer, self.library, k=2)
        self.assertEqual(outcome.kind, "succeeded")
        self.assertEqual(outcome.report["changes"], ["choice claude description"])

    def test_heartbeat_runs_before_each_session_and_evaluation(self):
        job = self.claim()
        self.events.clear()
        improve(job, self.client, self.proposer([STRONG, WEAK, WEAK]), self.library, k=2)
        # two proposals and one compose, each: heartbeat, session, heartbeat, evaluate
        self.assertEqual(self.events, ["heartbeat", "session", "heartbeat", "evaluate"] * 3)

    def test_lease_loss_propagates(self):
        job = self.claim()
        self.svc.lose_lease(job.id)
        with self.assertRaises(LeaseLost):
            improve(job, self.client, self.proposer([STRONG, STRONG, STRONG]), self.library, k=3)
        self.assertEqual(self.svc.results, [])
        self.assertEqual(self.library.notes("wf-1"), "")

    def test_lease_loss_during_a_session_propagates(self):
        job = self.claim()
        proposer = self.proposer([STRONG, STRONG])
        original = proposer._claude_cli

        def lose_lease_mid_session(cwd, mission, system, wall):
            self.svc.lose_lease(job.id)
            return original(cwd, mission, system, wall)

        proposer._claude_cli = lose_lease_mid_session
        with self.assertRaises(LeaseLost):
            improve(job, self.client, proposer, self.library, k=2)
        self.assertEqual(self.library.notes("wf-1"), "")

    def test_library_notes_feed_the_next_job(self):
        first = improve(self.claim(), self.client, self.proposer([STRONG, WEAK, WEAK]), self.library, k=2)
        summary = first.report["summary"]
        self.assertIn(summary, self.library.notes("wf-1"))
        proposer = self.proposer([WEAK, WEAK, WEAK])
        improve(self.claim(), self.client, proposer, self.library, k=2)
        self.assertIn(summary, proposer.sessions[0]["brief/experience.md"].decode())

    def test_first_job_still_gets_an_experience_file(self):
        proposer = self.proposer([WEAK, WEAK, WEAK])
        improve(self.claim(), self.client, proposer, self.library, k=2)
        self.assertTrue(proposer.sessions[0]["brief/experience.md"].decode().strip())

    def test_job_is_recorded_in_the_library(self):
        job = self.claim()
        improve(job, self.client, self.proposer([WEAK, WEAK, WEAK]), self.library, k=2)
        notes = self.library.notes("wf-1")
        self.assertIn(job.id, notes)
        self.assertIn("no-improvement", notes)


class ExperienceLibraryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="decisions-library-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.library = ExperienceLibrary(self.root / "nested")

    def entry(self, n: int) -> dict:
        return {
            "jobId": f"job-{n}",
            "baseVersion": n,
            "baselineAccuracy": 0.5,
            "bestAccuracy": 0.75,
            "outcome": "succeeded",
            "changes": ["routing instructions"],
            "summary": f"lesson number {n}",
        }

    def test_notes_are_empty_without_history(self):
        self.assertEqual(self.library.notes("wf"), "")

    def test_notes_list_newest_first_up_to_the_limit(self):
        for n in range(1, 5):
            self.library.record("wf", self.entry(n))
        notes = self.library.notes("wf", limit=2)
        self.assertNotIn("lesson number 2", notes)
        self.assertLess(notes.index("lesson number 4"), notes.index("lesson number 3"))
        self.assertEqual(self.library.notes("wf"), self.library.notes("wf", limit=10))

    def test_workflows_do_not_share_notes(self):
        self.library.record("wf-a", self.entry(1))
        self.assertEqual(self.library.notes("wf-b"), "")

    def test_record_appends_one_json_line_in_a_slugged_file(self):
        self.library.record("a/b:c ../d", self.entry(1))
        self.library.record("a/b:c ../d", self.entry(2))
        files = list((self.root / "nested").iterdir())
        self.assertEqual([f.name for f in files], ["abc..d.jsonl"])
        lines = files[0].read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["jobId"] for line in lines], ["job-1", "job-2"])

    def test_non_ascii_text_round_trips(self):
        entry = self.entry(1)
        entry["summary"] = "Đã sửa mô tả của claude"
        self.library.record("wf", entry)
        self.assertIn(entry["summary"], self.library.notes("wf"))

    def test_a_torn_line_does_not_hide_the_others(self):
        self.library.record("wf", self.entry(1))
        path = next((self.root / "nested").iterdir())
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"jobId": "job-2", "summ\n')
        self.library.record("wf", self.entry(3))
        notes = self.library.notes("wf")
        self.assertIn("lesson number 1", notes)
        self.assertIn("lesson number 3", notes)

    def test_key_with_no_safe_characters_still_gets_a_file(self):
        self.library.record("///", self.entry(1))
        self.assertIn("lesson number 1", self.library.notes("///"))


class WorkerClient(DecisionClient):
    """Records which thread heartbeats and which results are posted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heartbeat_threads: list[threading.Thread] = []
        self.posted: list[str] = []
        self.thread_heartbeat_error: Exception | None = None

    def heartbeat(self, job_id):
        thread = threading.current_thread()
        self.heartbeat_threads.append(thread)
        if self.thread_heartbeat_error is not None and thread is not threading.main_thread():
            raise self.thread_heartbeat_error
        return super().heartbeat(job_id)

    def succeed(self, job_id, questions, report):
        self.posted.append("succeed")
        return super().succeed(job_id, questions, report)

    def no_improvement(self, job_id, report):
        self.posted.append("no_improvement")
        return super().no_improvement(job_id, report)

    def fail(self, job_id, reason):
        self.posted.append("fail")
        return super().fail(job_id, reason)


class RaisingProposer(FakeProposer):
    """A proposer that breaks outside the session, so the core cannot wrap the error."""

    def call(self, mission, context_dir, label, **kwargs):
        raise RuntimeError("proposer exploded")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.audit = Path(tempfile.mkdtemp(prefix="decisions-audit-"))
        self.addCleanup(shutil.rmtree, self.audit, ignore_errors=True)
        self.library = ExperienceLibrary(Path(tempfile.mkdtemp(prefix="decisions-library-")))
        self.addCleanup(shutil.rmtree, self.library.root, ignore_errors=True)
        self.svc = FakeDecisionService().start()
        self.addCleanup(self.svc.stop)
        self.svc.answers = brainstorm_aware
        self.client = WorkerClient(self.svc.url, "w1")

    def proposer(self, edits: list, cls=FakeProposer) -> FakeProposer:
        return cls(edits=edits, audit_dir=self.audit)

    def run_once(self, proposer, k=3, **kwargs) -> str:
        return decision_worker.run_once(self.client, proposer, self.library, k, **kwargs)

    def test_run_once_is_idle_without_jobs(self):
        proposer = self.proposer([])
        self.assertEqual(self.run_once(proposer), "idle")
        self.assertEqual(proposer.sessions, [])
        self.assertEqual(self.client.posted, [])

    def test_run_once_posts_succeeded_with_questions(self):
        self.svc.queue(improve_payload())
        proposer = self.proposer([WEAK, STRONG, WEAK, STRONG_AND_MORE])
        self.assertEqual(self.run_once(proposer), "succeeded")
        result = self.svc.results[-1]
        self.assertEqual(result["outcome"], "succeeded")
        self.assertEqual(result["worker"], "w1")
        claude = next(c for c in result["questions"][0]["choices"] if c["value"] == "claude")
        self.assertIn("brainstorm", claude["description"])
        self.assertIn("summary", result["report"])

    def test_run_once_posts_no_improvement_without_questions(self):
        self.svc.queue(improve_payload())
        self.assertEqual(self.run_once(self.proposer([WEAK, WEAK, WEAK])), "no-improvement")
        result = self.svc.results[-1]
        self.assertEqual(result["outcome"], "no-improvement")
        self.assertNotIn("questions", result)

    def test_run_once_posts_failed_on_unexpected_error(self):
        self.svc.queue(improve_payload())
        with self.assertLogs(decision_worker.log, level="WARNING"):
            outcome = self.run_once(self.proposer([WEAK], RaisingProposer))
        self.assertEqual(outcome, "failed")
        result = self.svc.results[-1]
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(result["reason"], "RuntimeError: proposer exploded")

    def test_lease_loss_posts_nothing(self):
        job_id = self.svc.queue(improve_payload())
        proposer = self.proposer([STRONG, STRONG, STRONG])
        original = proposer._claude_cli

        def lose_lease_mid_session(cwd, mission, system, wall):
            self.svc.lose_lease(job_id)
            return original(cwd, mission, system, wall)

        proposer._claude_cli = lose_lease_mid_session
        with self.assertLogs(decision_worker.log, level="WARNING"):
            self.assertEqual(self.run_once(proposer), "lease-lost")
        self.assertEqual(self.svc.results, [])
        self.assertEqual(self.client.posted, [])

    def test_worker_survives_service_outage(self):
        self.svc.queue(improve_payload())
        proposer = self.proposer([STRONG, STRONG, STRONG])
        original = proposer._claude_cli

        def die_mid_session(cwd, mission, system, wall):
            self.svc.stop()
            return original(cwd, mission, system, wall)

        proposer._claude_cli = die_mid_session
        with self.assertLogs(decision_worker.log, level="WARNING"):
            self.assertEqual(self.run_once(proposer), "failed")
        self.assertEqual(self.client.posted, ["fail"])  # tried to tell the dead service
        with FakeDecisionService() as fresh:
            client = DecisionClient(fresh.url, "w1")
            outcome = decision_worker.run_once(client, self.proposer([]), self.library, 3)
        self.assertEqual(outcome, "idle")

    def sleepy(self, proposer: FakeProposer, before=None, seconds: float = 0.4) -> None:
        original = proposer._claude_cli

        def slow_session(cwd, mission, system, wall):
            if before is not None:
                before()
            time.sleep(seconds)
            return original(cwd, mission, system, wall)

        proposer._claude_cli = slow_session

    def test_heartbeat_thread_keeps_the_lease_during_a_long_session(self):
        self.svc.queue(improve_payload())
        proposer = self.proposer([WEAK])
        self.sleepy(proposer)
        self.assertEqual(self.run_once(proposer, k=1, heartbeat_seconds=0.05), "no-improvement")
        background = [t for t in self.client.heartbeat_threads if t is not threading.main_thread()]
        self.assertGreaterEqual(len(background), 3)
        self.assertEqual([t for t in threading.enumerate() if t in background and t.is_alive()], [])

    def test_heartbeat_thread_sees_lease_loss_and_nothing_is_posted(self):
        job_id = self.svc.queue(improve_payload())
        proposer = self.proposer([{}])  # no edit: improve finishes without another service call
        self.sleepy(proposer, before=lambda: self.svc.lose_lease(job_id))
        with self.assertLogs(decision_worker.log, level="WARNING"):
            self.assertEqual(self.run_once(proposer, k=1, heartbeat_seconds=0.05), "lease-lost")
        self.assertEqual(self.client.posted, [])
        self.assertEqual(self.svc.results, [])

    def test_heartbeat_thread_outage_is_ignored(self):
        self.svc.queue(improve_payload())
        self.client.thread_heartbeat_error = ServiceError("blip")
        proposer = self.proposer([WEAK])
        self.sleepy(proposer)
        self.assertEqual(self.run_once(proposer, k=1, heartbeat_seconds=0.05), "no-improvement")
        background = [t for t in self.client.heartbeat_threads if t is not threading.main_thread()]
        self.assertGreaterEqual(len(background), 2)  # it kept beating after the first error

    def test_heartbeat_thread_stops_when_the_job_fails(self):
        self.svc.queue(improve_payload())
        with self.assertLogs(decision_worker.log, level="WARNING"):
            self.run_once(self.proposer([WEAK], RaisingProposer), heartbeat_seconds=0.05)
        self.assertEqual([t for t in threading.enumerate() if t.name == "lease-keeper"], [])


class WorkerMainTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(decision_worker, "_configure_logging")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="decisions-main-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def argv(self, url: str, *extra: str) -> list[str]:
        return [
            "--decision-url", url,
            "--library-dir", str(self.tmp / "library"),
            "--audit-dir", str(self.tmp / "audit"),
            *extra,
        ]  # fmt: skip

    def test_defaults(self):
        args = decision_worker.build_parser().parse_args([])
        self.assertEqual(args.decision_url, "http://localhost:3000")
        self.assertRegex(args.worker_id, rf"-{os.getpid()}$")
        self.assertEqual((args.k, args.model, args.poll_seconds, args.once), (3, "", 30, False))
        self.assertEqual(args.library_dir, config.RUNS / "decisions" / "library")
        self.assertEqual(args.audit_dir, config.RUNS / "decisions" / "audit")

    def test_once_exits_zero_when_idle(self):
        with FakeDecisionService() as svc, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(decision_worker.main(self.argv(svc.url, "--once")), 0)
        self.assertEqual(out.getvalue().strip(), "idle")

    def test_once_exits_one_when_the_service_is_down(self):
        svc = FakeDecisionService().start()
        url = svc.url
        svc.stop()
        with self.assertLogs(decision_worker.log, level="WARNING"):
            self.assertEqual(decision_worker.main(self.argv(url, "--once")), 1)

    def test_loop_sleeps_when_idle_and_ctrl_c_exits_zero(self):
        outcomes = ["idle", "succeeded", KeyboardInterrupt()]
        with (
            patch.object(decision_worker, "run_once", side_effect=outcomes) as run_once,
            patch.object(decision_worker.time, "sleep") as sleep,
        ):
            code = decision_worker.main(self.argv("http://127.0.0.1:9", "--poll-seconds", "7"))
        self.assertEqual(code, 0)
        self.assertEqual(run_once.call_count, 3)
        sleep.assert_called_once_with(7)  # only the idle round waits

    def test_loop_keeps_going_when_claim_fails(self):
        outcomes = [ServiceError("down"), "idle", KeyboardInterrupt()]
        with (
            patch.object(decision_worker, "run_once", side_effect=outcomes),
            patch.object(decision_worker.time, "sleep") as sleep,
            self.assertLogs(decision_worker.log, level="WARNING"),
        ):
            code = decision_worker.main(self.argv("http://127.0.0.1:9", "--poll-seconds", "5"))
        self.assertEqual(code, 0)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
