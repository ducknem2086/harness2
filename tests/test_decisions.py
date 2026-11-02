"""Tests for the decision service client (harness jobs API)."""

import unittest

from harness2.decisions.client import (
    CandidateInvalid,
    DecisionClient,
    LeaseLost,
    ServiceError,
)
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


if __name__ == "__main__":
    unittest.main()
