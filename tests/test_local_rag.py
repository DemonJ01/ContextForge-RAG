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

    def test_natural_wording_retrieves_the_actual_supporting_evidence(self):
        cases = {
            "How can I troubleshoot a failing CronJob?": (
                "kubernetes_cronjobs.docx", "kubectl describe cronjob <cronjob-name>"),
            "How do I debug failed CronJobs?": (
                "kubernetes_cronjobs.docx", "kubectl describe cronjob <cronjob-name>"),
            "How to run a job every day?": (
                "kubernetes_cronjobs.docx", 'schedule: "@daily"'),
            "How can I configure cron schedules?": (
                "kubernetes_cronjobs.docx", "Common Schedule Examples"),
            "What happens if my job fails?": (
                "kubernetes_job_monitoring.docx", "backoffLimit: 2"),
            "How does Redis handle work queues?": (
                "kubernetes_parallel_work_queue.txt", 'rpush job2 "apple"'),
            "How do I configure autoscaling?": (
                "kubernetes_pod_autoscaling.html", "kubectl"),
        }
        for query, (filename, fact) in cases.items():
            with self.subTest(query=query):
                result = self.rag.query(query)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["citations"][0]["filename"], filename)
                self.assertIn(fact, "\n".join(result["sources"]))

    def test_natural_api_field_and_plural_wording_has_source_evidence(self):
        cases = {
            "Explain CronJob concurrency policies": "concurrencyPolicy: Forbid",
            "How do I check Kubernetes job statuses?": "kubectl get jobs",
            "How do Kubernetes jobs handle failures?": "backoffLimit",
        }
        for query, fact in cases.items():
            with self.subTest(query=query):
                result = self.rag.query(query)
                self.assertEqual(result["status"], "complete")
                self.assertIn(fact, "\n".join(result["sources"]))

    def test_terminal_how_operation_verbs_preserve_named_subject_evidence(self):
        for question, filename in (
            ("How do CronJobs work?", "kubernetes_cronjobs.docx"),
            ("How does this work queue work?", "kubernetes_parallel_work_queue.txt"),
            ("How does HPA work?", "kubernetes_pod_autoscaling.html"),
        ):
            expected = self.rag.query(question)
            for verb in ("function", "operate"):
                alternative = question.removesuffix("work?") + verb + "?"
                with self.subTest(question=alternative):
                    result = self.rag.query(alternative)
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["citations"][0]["filename"], filename)
                    self.assertEqual(result["sources"], expected["sources"])
                    for citation in result["citations"]:
                        self.assertIn(citation["excerpt"], self.rag.document(citation["document_id"])["content"])

    def test_how_operation_verbs_also_work_for_genuine_contextual_followups(self):
        from rag.runtime import RAGRuntime
        runtime = RAGRuntime(ROOT, mode="local")
        named = runtime.query("How does this work queue function?", thread_id="named-queue")
        self.assertEqual(named["status"], "complete")
        self.assertTrue(all(c["filename"] == "kubernetes_parallel_work_queue.txt" for c in named["citations"]))
        for index, question in enumerate(("How does it function?", "How does this operate?")):
            with self.subTest(question=question):
                thread = f"how-operation-{index}"
                anchor = runtime.query("Explain HPA in simple words", thread_id=thread)
                result = runtime.query(question, thread_id=thread)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["sources"], anchor["sources"])
                self.assertTrue(all(c["filename"] == "kubernetes_pod_autoscaling.html" for c in result["citations"]))
                self.assertIn("Resolved the follow-up", result["thought_process"][0])

    def test_operation_predicates_do_not_hide_named_functions_or_unsupported_capabilities(self):
        for question in (
            "How do SQL functions operate?", "How does this SQL function work?",
            "How do I create a SQL function?", "How does a SQL function?",
            "How does teleport operate?", "How do Kubernetes Jobs teleport?",
        ):
            with self.subTest(question=question):
                result = self.rag.query(question, scope="all")
                self.assertEqual(result["status"], "no_answer")
                self.assertEqual(result["sources"], [])
                self.assertEqual(result["citations"], [])

    def test_unambiguous_topic_typos_retrieve_the_indexed_subject(self):
        cases = {
            "How do I monitor kubernets jobs?": "kubernetes_job_monitoring.docx",
            "What is a CronJab?": "kubernetes_cronjobs.docx",
        }
        for query, filename in cases.items():
            with self.subTest(query=query):
                result = self.rag.query(query)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["citations"][0]["filename"], filename)
                self.assertTrue(any("Matched" in step for step in result["thought_process"]))

    def test_presentation_wording_preserves_definitions_and_comparisons(self):
        for query in ("Explain HPA in simple words", "What is HPA in plain English?"):
            with self.subTest(query=query):
                result = self.rag.query(query)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["sources"][0].startswith("What is HPA?"))
                self.assertIn("automatically scales the number of pods", result["sources"][0])
        result = self.rag.query("What are Kubernetes Jobs in simple terms?")
        self.assertEqual(result["status"], "complete")
        self.assertIn("A Kubernetes Job is", result["sources"][0])
        result = self.rag.query("How do HPA and VPA differ?")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["citations"][0]["filename"], "kubernetes_pod_autoscaling.html")

    def test_definition_requests_lead_with_the_definition_when_only_one_excerpt_is_requested(self):
        cases = {
            "What is a Kubernetes CronJob?": ("What Are CronJobs?", "thin wrapper around Jobs"),
            "Describe Kubernetes CronJobs": ("What Are CronJobs?", "thin wrapper around Jobs"),
            "Explain HPA": ("What is HPA?", "automatically scales the number of pods"),
            "What are Kubernetes Jobs?": ("A Kubernetes Job is", "successfully terminate"),
        }
        for question, (opening, fact) in cases.items():
            with self.subTest(question=question):
                result = self.rag.query(question, top_k=1)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["sources"]), 1)
                self.assertTrue(result["sources"][0].startswith(opening))
                self.assertIn(fact, result["sources"][0])

    def test_comparison_does_not_treat_a_troubleshooting_symptom_as_a_definition(self):
        result = self.rag.query("What is the difference between Jobs and CronJobs?", top_k=1)
        self.assertEqual(result["status"], "complete")
        self.assertIn("A CronJob is a thin wrapper around Jobs", result["sources"][0])
        self.assertNotIn("Symptoms: CronJob exists but no Jobs are created", result["sources"][0])

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
            "How do I configure the Kubernetes default password?",
            "How do Kubernetes Jobs handle quantum entanglement?",
            "How can I troubleshoot a telepathic CronJob?",
            "How do I run a job with teleportation?",
            "What are the limitations of Kubernetes CronJobs?",
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
