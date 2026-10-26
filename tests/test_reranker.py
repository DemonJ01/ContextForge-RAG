"""Verify model selection without loading FlashRank or downloading model files."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


class RerankerModelTests(unittest.TestCase):
    def load_reranker(self, constructor):
        flashrank = ModuleType("flashrank")
        flashrank.Ranker = constructor
        flashrank.RerankRequest = SimpleNamespace
        logfire = ModuleType("logfire")
        for name in ("info", "warning", "error"):
            setattr(logfire, name, Mock())
        source = Path(__file__).resolve().parents[1] / "rag/services/retrieval/reranker.py"
        spec = importlib.util.spec_from_file_location("_reranker_under_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"flashrank": flashrank, "logfire": logfire}):
            spec.loader.exec_module(module)
        constructor.assert_not_called()
        return module

    def test_explicit_model_is_loaded_once_and_ranked_evidence_is_preserved(self):
        ranker = Mock()
        ranker.rerank.return_value = [
            {"id": 1, "text": "relevant evidence", "score": 0.9},
            {"id": 0, "text": "other evidence", "score": 0.1},
        ]
        constructor = Mock(return_value=ranker)
        module = self.load_reranker(constructor)
        result = module.rerank_documents("CronJobs", ["other evidence", "relevant evidence"], top_n=1)
        self.assertEqual(result, ["relevant evidence"])
        constructor.assert_called_once_with(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
        request = ranker.rerank.call_args.args[0]
        self.assertEqual(request.query, "CronJobs")
        self.assertEqual(request.passages, [{"id": 0, "text": "other evidence"}, {"id": 1, "text": "relevant evidence"}])
        self.assertIs(module._get_ranker(), ranker)
        constructor.assert_called_once()

    def test_alternate_cache_keeps_the_same_model(self):
        ranker = Mock()
        constructor = Mock(side_effect=[OSError("cache unavailable"), ranker])
        module = self.load_reranker(constructor)
        self.assertIs(module._get_ranker(), ranker)
        self.assertEqual(constructor.call_args_list, [
            call(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank"),
            call(model_name="ms-marco-TinyBERT-L-2-v2"),
        ])


if __name__ == "__main__":
    unittest.main()
