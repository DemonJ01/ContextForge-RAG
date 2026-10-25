"""Targeted wording regressions using the bundled, unchanged source documents."""

from pathlib import Path
import unittest

from rag.local_rag import LocalRAG
from rag.runtime import RAGRuntime


ROOT = Path(__file__).resolve().parents[1]
CRON_DOCUMENT = "kubernetes_cronjobs.docx"
AUTOSCALING_DOCUMENT = "kubernetes_pod_autoscaling.html"
WRAPPER_ANCHORS = (
    "Can you walk me through how CronJobs work?",
    "Please give me a quick overview of Kubernetes Jobs.",
    "What does HPA stand for?",
)


class QueryWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = LocalRAG(ROOT)

    def assert_top_evidence(self, question, filename, *facts, opening=None, scope="reference"):
        result = self.rag.query(question, scope=scope, top_k=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(len(result["citations"]), 1)
        citation = result["citations"][0]
        self.assertEqual(citation["filename"], filename)
        excerpt = result["sources"][0]
        self.assertEqual(excerpt, citation["excerpt"])
        document = self.rag.document(citation["document_id"])
        self.assertIn(excerpt, document["content"])
        self.assertEqual(citation["source_type"], document["source_type"])
        self.assertIn(excerpt + " [1]", result["answer"])
        if scope == "reference":
            self.assertEqual(citation["source_type"], "true")
        if opening is not None:
            self.assertTrue(excerpt.startswith(opening), excerpt)
        for fact in facts:
            self.assertIn(fact, excerpt)
        return result

    def assert_no_evidence(self, question, scope="all"):
        result = self.rag.query(question, scope=scope)
        self.assertEqual(result["status"], "no_answer")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citations"], [])

    def test_spaced_cron_job_names_find_the_definition(self):
        for question in ("What is a Cron Job?", "Describe Kubernetes Cron Jobs"):
            with self.subTest(question=question):
                self.assert_top_evidence(
                    question, CRON_DOCUMENT, "thin wrapper around Jobs",
                    opening="What Are CronJobs?",
                )

    def test_expanded_autoscaling_names_lead_with_their_definitions(self):
        for name, abbreviation, fact in (
            ("horizontal", "HPA", "automatically scales the number of pods"),
            ("vertical", "VPA", "adjusts the CPU and memory resource requests and limits"),
        ):
            with self.subTest(name=name):
                self.assert_top_evidence(
                    f"What is {name} pod autoscaling?", AUTOSCALING_DOCUMENT, fact,
                    opening=f"What is {abbreviation}?",
                )

    def test_conversational_overview_requests_keep_the_named_subject(self):
        self.assert_top_evidence(
            "Can you walk me through how CronJobs work?", CRON_DOCUMENT,
            "thin wrapper around Jobs",
        )
        self.assert_top_evidence(
            "Please give me a quick overview of Kubernetes Jobs.",
            "kubernetes_job_monitoring.docx", "A Kubernetes Job is", "successfully terminate",
        )

    def test_abbreviation_and_layman_wording_find_the_hpa_definition(self):
        for question in ("What does HPA stand for?", "Explain HPA in layman terms"):
            with self.subTest(question=question):
                self.assert_top_evidence(
                    question, AUTOSCALING_DOCUMENT, "Horizontal Pod Autoscaling (HPA)",
                    "automatically scales the number of pods", opening="What is HPA?",
                )

    def test_retry_count_question_returns_the_documented_example(self):
        self.assert_top_evidence(
            "How many times will a failed Job retry?", "kubernetes_job_monitoring.docx",
            "backoffLimit", "retry up to 4 times",
        )

    def test_timezone_wording_returns_the_actual_api_field_example(self):
        for question in ("How do I set a CronJob timezone?", "Explain CronJob timeZone"):
            with self.subTest(question=question):
                self.assert_top_evidence(
                    question, CRON_DOCUMENT, 'timeZone: "America/New_York"',
                )

    def test_scheduler_question_returns_its_workload_placement_evidence(self):
        self.assert_top_evidence(
            "What does the Kubernetes scheduler do?", "kubernetes_architecture.pptx",
            "evaluates workload", "matching resource",
        )

    def test_presentation_wording_does_not_hide_unsupported_content(self):
        for question in (
            "Please give me a quick overview of the Kubernetes default password.",
            "Can you walk me through how CronJobs teleport?",
            "Explain HPA quantum entanglement in layman terms",
            "Explain CronJob teleportPolicy",
        ):
            with self.subTest(question=question):
                self.assert_no_evidence(question)

    def test_wording_normalization_preserves_source_scope(self):
        for question in ("What is a Cron Job?", "What is horizontal pod autoscaling?"):
            with self.subTest(question=question):
                self.assert_no_evidence(question, scope="noisy")
        self.assert_top_evidence(
            "Explain CronJob timeZone", CRON_DOCUMENT,
            'timeZone: "America/New_York"', scope="all",
        )
        question = "Intel 5 level paging EPT"
        result = self.rag.query(question, scope="noisy", top_k=1)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["citations"])
        for citation in result["citations"]:
            self.assertEqual(citation["source_type"], "noisy")
            self.assertIn(citation["excerpt"], self.rag.document(citation["document_id"])["content"])
        self.assert_no_evidence(question, scope="reference")


class WrapperFollowupTests(unittest.TestCase):
    def setUp(self):
        self.runtime = RAGRuntime(ROOT, mode="local")

    def ask(self, question, thread, scope="reference"):
        return self.runtime.query(question, thread_id=thread, scope=scope, top_k=1)

    def assert_no_evidence(self, result):
        self.assertEqual(result["status"], "no_answer")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citations"], [])

    def test_wrapper_anchors_preserve_evidence_on_pronoun_followups(self):
        for index, question in enumerate(WRAPPER_ANCHORS):
            with self.subTest(question=question):
                thread = f"wrapper-followup-{index}"
                anchor = self.ask(question, thread)
                self.assertEqual(anchor["status"], "complete")
                self.assertTrue(anchor["sources"])
                followup = self.ask("How does it work?", thread)
                self.assertEqual(followup["status"], "complete")
                self.assertEqual(followup["sources"], anchor["sources"])
                self.assertEqual(
                    [citation["id"] for citation in followup["citations"]],
                    [citation["id"] for citation in anchor["citations"]],
                )

    def test_explicit_unsupported_topic_clears_wrapper_context(self):
        for index, question in enumerate(WRAPPER_ANCHORS):
            with self.subTest(question=question):
                thread = f"wrapper-topic-change-{index}"
                anchor = self.ask(question, thread)
                self.assertEqual(anchor["status"], "complete")
                self.assert_no_evidence(self.ask("Explain quantum entanglement", thread))
                followup = self.ask("How does it work?", thread)
                self.assert_no_evidence(followup)
                self.assertIn("name the subject", followup["answer"])

    def test_wrapper_context_stays_in_its_thread_and_search_scope(self):
        for index, question in enumerate(WRAPPER_ANCHORS):
            with self.subTest(question=question):
                thread = f"wrapper-scope-{index}"
                anchor = self.ask(question, thread)
                self.assertEqual(anchor["status"], "complete")
                for scope in ("all", "noisy"):
                    self.assert_no_evidence(self.ask("How does it work?", thread, scope))
                self.assert_no_evidence(self.ask("How does it work?", f"other-{thread}"))
                followup = self.ask("How does it work?", thread)
                self.assertEqual(followup["status"], "complete")
                self.assertEqual(followup["sources"], anchor["sources"])


if __name__ == "__main__":
    unittest.main()
