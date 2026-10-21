"""Real-corpus checks for the credential-free retrieval mode."""

from pathlib import Path
import json
import re
import unittest

from rag.local_rag import LocalRAG, _MAX_CHUNK_CHARS


ROOT = Path(__file__).resolve().parents[1]


class LocalRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = LocalRAG(ROOT)

    def test_indexes_all_documents_and_scopes(self):
        stats = self.rag.health()
        self.assertEqual(stats["document_count"], 11)
        self.assertEqual(stats["reference_document_count"], 6)
        self.assertGreater(stats["chunk_count"], 80)
        self.assertEqual(len(self.rag.documents("reference")), 6)
        self.assertEqual(len(self.rag.documents("noisy")), 5)
        self.assertEqual(len(self.rag.documents()), 11)
        self.assertTrue(all("content" not in d for d in self.rag.documents()))

    def test_stable_document_and_chunk_ids(self):
        other = LocalRAG(ROOT)
        self.assertEqual(self.rag.documents(), other.documents())
        first = self.rag.query("How do Kubernetes CronJobs schedule jobs?")
        second = other.query("How do Kubernetes CronJobs schedule jobs?")
        self.assertEqual(first["citations"], second["citations"])
        self.assertTrue(all(len(d["id"]) == 16 for d in self.rag.documents()))

    def test_oversized_chunks_are_split_and_content_is_preserved(self):
        self.assertTrue(all(0 < len(chunk.text) <= _MAX_CHUNK_CHARS for chunk in self.rag._chunks))
        for chunk in self.rag._chunks:
            self.assertIn(chunk.text, self.rag.document(chunk.document_id)["content"])
        self.assertIsNone(self.rag.document("../../etc/passwd"))

    def test_domain_queries_find_the_expected_documents(self):
        cases = {
            "How do Kubernetes CronJobs schedule recurring jobs?": "kubernetes_cronjobs.docx",
            "What is the difference between HPA and VPA?": "kubernetes_pod_autoscaling.html",
            "How do I monitor Kubernetes job status and logs?": "kubernetes_job_monitoring.docx",
            "How does a Redis work queue process parallel tasks?": "kubernetes_parallel_work_queue.txt",
            "Show Databricks CLI commands to manage jobs": "databricks_job_management.html",
        }
        for query, filename in cases.items():
            with self.subTest(query=query):
                result = self.rag.query(query)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["citations"][0]["filename"], filename)

    def test_answers_quote_real_text_and_have_resolving_citations(self):
        result = self.rag.query("How do I monitor Kubernetes job logs?", top_k=3)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["citations"]), 3)
        self.assertEqual(result["sources"], [c["excerpt"] for c in result["citations"]])
        for index, citation in enumerate(result["citations"], 1):
            document = self.rag.document(citation["document_id"])
            self.assertIn(citation["excerpt"], document["content"])
            self.assertIn(citation["excerpt"] + f" [{index}]", result["answer"])
            self.assertEqual(citation["filename"], document["filename"])
            self.assertGreater(citation["score"], 0)

    def test_no_evidence_returns_no_answer_or_citations(self):
        for query in (
            "Explain quantum entanglement", "What is today's weather?",
            "quantum computing", "", "the and a",
            "Which Kubernetes version supports telepathy?",
            "How do Kubernetes jobs teleport pods?",
            "Which Kubernetes Job restart policy prevents quantum entanglement?",
            "What is the Kubernetes default password?",
        ):
            with self.subTest(query=query):
                result = self.rag.query(query, scope="all")
                self.assertEqual(result["status"], "no_answer")
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["sources"], [])

    def test_existing_golden_questions_retrieve_their_reference_topics(self):
        samples = json.loads((ROOT / "evaluation/rag_golden_dataset.json").read_text())["rag_samples"]
        expected = {
            "parallel_work_queue": "kubernetes_parallel_work_queue.txt",
            "pods_autoscale": "kubernetes_pod_autoscaling.html",
            "job_management": "databricks_job_management.html",
            "cronjobs": "kubernetes_cronjobs.docx",
            "monitor_job": "kubernetes_job_monitoring.docx",
        }
        self.assertEqual(len(samples), 15)
        for sample in samples:
            with self.subTest(sample=sample["id"]):
                result = self.rag.query(sample["question"])
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["citations"][0]["filename"], expected[sample["domain"]])

    def test_evidence_keeps_commands_and_definitions_from_golden_questions(self):
        samples = {s["id"]: s for s in json.loads(
            (ROOT / "evaluation/rag_golden_dataset.json").read_text())["rag_samples"]}
        facts = {
            1: ["redis-pod.yaml", "redis-service.yaml"],
            3: ["rpush job2", "lrange job2 0 -1"],
            4: ["number of pods", "adjusts the CPU and memory resource requests and limits"],
            5: ["https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml",
                "kubectl get pods -n kube-system | grep metrics-server"],
            6: ["kubectl top nodes", "kubectl top pods --all-namespaces"],
            9: ["one-off tasks", "Python, Java, Go, or R", "not currently available"],
        }
        for sample_id, expected_facts in facts.items():
            result = self.rag.query(samples[sample_id]["question"])
            evidence = "\n".join(result["sources"])
            for fact in expected_facts:
                with self.subTest(sample=sample_id, fact=fact):
                    self.assertIn(fact, evidence)
            self.assertTrue(all(len(c["excerpt"]) <= _MAX_CHUNK_CHARS for c in result["citations"]))

    def test_frontend_suggested_questions_work_with_the_local_corpus(self):
        html = (ROOT / "ui/web/index.html").read_text()
        prompts = re.findall(r'data-question="([^"]+)"', html)
        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = self.rag.query(prompt)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["citations"])

    def test_noisy_scope_and_result_limit(self):
        result = self.rag.query("Intel 5 level paging EPT", scope="all", top_k=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["source_type"], "noisy")
        self.assertLessEqual(len(self.rag.query("Kubernetes", top_k=100)["citations"]), 8)
        self.assertEqual(len(self.rag.query("Kubernetes", top_k=0)["citations"]), 1)
        with self.assertRaises(ValueError):
            self.rag.query("Kubernetes", scope="unknown")


if __name__ == "__main__":
    unittest.main()
