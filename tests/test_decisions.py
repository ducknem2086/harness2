"""Tests for the decision service client (harness jobs API)."""

import copy
import unittest

from harness2.decisions.client import (
    CandidateInvalid,
    DecisionClient,
    Evaluation,
    LeaseLost,
    ServiceError,
)
from harness2.decisions.surface import apply_files, changed, context_files, editable_files
from tests.decisions_fakes import FakeDecisionService, payload


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


if __name__ == "__main__":
    unittest.main()
