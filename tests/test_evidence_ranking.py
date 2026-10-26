"""Source relationships must survive ranking and identifier normalization."""

from pathlib import Path
import unittest

from rag.local_rag import LocalRAG


ROOT = Path(__file__).resolve().parents[1]


class EvidenceRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = LocalRAG(ROOT)

    def first_excerpt(self, question, filename, *, scope="reference"):
        result = self.rag.query(question, scope=scope, top_k=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["citations"]), 1)
        citation = result["citations"][0]
        self.assertEqual(citation["filename"], filename)
        self.assertEqual(result["sources"], [citation["excerpt"]])
        self.assertIn(citation["excerpt"], self.rag.document(citation["document_id"])["content"])
        return citation["excerpt"]

    def test_restart_policy_value_leads_with_the_actual_prohibition(self):
        for field in ("restartPolicy", "RESTARTPOLICY", "restart policy"):
            with self.subTest(field=field):
                excerpt = self.first_excerpt(
                    f"Can I use {field} Always for a Job?", "kubernetes_cronjobs.docx")
                self.assertIn("Restart Policy Requirements", excerpt)
                self.assertIn("Always: Not allowed", excerpt)
                self.assertNotIn("imagePullPolicy: Always", excerpt)

    def test_a_value_from_another_field_is_not_evidence_for_the_requested_field(self):
        for question in (
            "Can I use restartPolicy Forbid for a Job?",
            "Can I use restartPolicy Replace for a Job?",
            "Explain CronJob concurrencyPolicy Never",
            "Explain CronJob concurrencyPolicy OnFailure",
            "Explain CronJob teleportPolicy Always",
        ):
            with self.subTest(question=question):
                result = self.rag.query(question, scope="all")
                self.assertEqual(result["status"], "no_answer")
                self.assertEqual(result["sources"], [])
                self.assertEqual(result["citations"], [])

    def test_documented_field_values_still_retrieve_their_own_assignments(self):
        excerpt = self.first_excerpt(
            "Explain CronJob concurrencyPolicy Forbid", "kubernetes_cronjobs.docx")
        self.assertIn("concurrencyPolicy: Forbid", excerpt)
        excerpt = self.first_excerpt(
            "Can I use restartPolicy Never for a Job?", "kubernetes_job_monitoring.docx")
        self.assertIn("restartPolicy: Never", excerpt)

    def test_dash_definition_beats_incidental_security_yaml(self):
        excerpt = self.first_excerpt("What is a Kubernetes Secret?", "kubernetes_architecture.pptx")
        self.assertTrue(excerpt.startswith("Secret -"))
        self.assertIn("stored encoded as base64", excerpt)

    def test_named_component_heading_beats_incidental_workflow_yaml(self):
        for question in (
            "What does the Kubernetes kubelet do?",
            "How does the Kubernetes kubelet work?",
        ):
            with self.subTest(question=question):
                excerpt = self.first_excerpt(question, "kubernetes_architecture.pptx")
                self.assertTrue(excerpt.startswith("kubelet\nActs as"))
                self.assertIn("managing pod lifecycle on its host", excerpt)

    def test_narrow_how_question_starts_with_the_named_concept_definition(self):
        excerpt = self.first_excerpt("How do Kubernetes CronJobs work?", "kubernetes_cronjobs.docx")
        self.assertTrue(excerpt.startswith("What Are CronJobs?"))
        self.assertIn("thin wrapper around Jobs", excerpt)
        self.assertNotIn("Introduction", excerpt)

    def test_comparison_connector_does_not_become_a_field_value(self):
        excerpt = self.first_excerpt("Explain ConfigMap and Secret", "kubernetes_architecture.pptx")
        self.assertIn("ConfigMap - Externalized data", excerpt)
        self.assertIn("Secret - Functionally identical", excerpt)

    def test_definition_and_field_relationships_stay_within_search_scope(self):
        for question in (
            "Can I use restartPolicy Always for a Job?",
            "What is a Kubernetes Secret?",
            "How do Kubernetes CronJobs work?",
            "How do Kubernetes CronJobs teleport?",
        ):
            with self.subTest(question=question):
                result = self.rag.query(question, scope="noisy")
                self.assertEqual(result["status"], "no_answer")
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
