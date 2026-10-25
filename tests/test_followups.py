"""Named demonstratives and genuine followups against the bundled evidence."""
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.runtime import RAGRuntime


ROOT = Path(__file__).resolve().parents[1]
HPA_DOCUMENT = "kubernetes_pod_autoscaling.html"
CRON_DOCUMENT = "kubernetes_cronjobs.docx"
QUEUE_DOCUMENT = "kubernetes_parallel_work_queue.txt"


class DemonstrativeFollowupTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(ROOT, runtime=RAGRuntime(ROOT, mode="local")))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def ask(self, question, thread):
        response = self.client.post("/query", json={"q": question, "thread_id": thread})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def assert_document_evidence(self, answer, document, *, exclusive=True):
        self.assertEqual(answer["status"], "complete")
        self.assertTrue(answer["sources"])
        self.assertTrue(answer["citations"])
        files = [citation["filename"] for citation in answer["citations"]]
        self.assertIn(document, files)
        if exclusive:
            self.assertTrue(all(filename == document for filename in files))
        self.assertTrue(all(citation["excerpt"] in answer["sources"] for citation in answer["citations"]))

    def test_fresh_questions_with_named_demonstrative_subjects_retrieve_evidence(self):
        for index, (question, document) in enumerate((
            ("How does this HPA work?", HPA_DOCUMENT),
            ("How does that CronJob work?", CRON_DOCUMENT),
            ("How do these CronJobs work?", CRON_DOCUMENT),
            ("How do those HPAs work?", HPA_DOCUMENT),
        )):
            with self.subTest(question=question):
                answer = self.ask(question, f"fresh-{index}")
                self.assert_document_evidence(answer, document)
                self.assertFalse(any("Resolved the follow-up" in step for step in answer["thought_process"]))

    def test_explicit_subject_changes_do_not_inherit_the_previous_topic(self):
        self.assert_document_evidence(self.ask("What is a Kubernetes CronJob?", "topic-change"), CRON_DOCUMENT, exclusive=False)
        hpa = self.ask("How does this HPA use CPU metrics?", "topic-change")
        self.assert_document_evidence(hpa, HPA_DOCUMENT)
        self.assertIn("CPU", "\n".join(hpa["sources"]))
        self.assertFalse(any("Resolved the follow-up" in step for step in hpa["thought_process"]))
        cron = self.ask("How does that CronJob work?", "topic-change")
        self.assert_document_evidence(cron, CRON_DOCUMENT)
        self.assertFalse(any("Resolved the follow-up" in step for step in cron["thought_process"]))

    def test_work_queue_is_an_explicit_subject_instead_of_a_generic_work_predicate(self):
        for index, question in enumerate(("How does this work queue work?", "How do these work queues work?")):
            with self.subTest(question=question):
                fresh = self.ask(question, f"queue-fresh-{index}")
                self.assert_document_evidence(fresh, QUEUE_DOCUMENT)
                thread = f"queue-after-hpa-{index}"
                self.ask("Explain HPA in simple words", thread)
                changed = self.ask(question, thread)
                self.assert_document_evidence(changed, QUEUE_DOCUMENT)
                self.assertEqual(changed["sources"], fresh["sources"])
                self.assertFalse(any("Resolved the follow-up" in step for step in changed["thought_process"]))

    def test_standalone_demonstratives_and_pronouns_still_require_and_use_context(self):
        for index, question in enumerate(("How does this work?", "How does it work?", "How do these work?", "How do they work?")):
            with self.subTest(question=question):
                fresh = self.ask(question, f"missing-{index}")
                self.assertEqual(fresh["status"], "no_answer")
                self.assertEqual(fresh["citations"], [])
                self.assertIn("name the subject", fresh["answer"])
                thread = f"context-{index}"
                anchor = self.ask("What is a Kubernetes CronJob?", thread)
                followup = self.ask(question, thread)
                self.assert_document_evidence(followup, CRON_DOCUMENT, exclusive=False)
                self.assertEqual(followup["sources"], anchor["sources"])
                self.assertIn("Resolved the follow-up", followup["thought_process"][0])


if __name__ == "__main__":
    unittest.main()
