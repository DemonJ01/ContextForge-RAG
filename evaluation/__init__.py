"""Optional evaluation entry points, imported only when used."""
from importlib import import_module


_EXPORTS = {
    "run_pipeline": "evaluation.rag_pipeline",
    "load_golden_dataset": "evaluation.rag_pipeline",
    "run_guardrails_eval": "evaluation.guardrails_evaluation",
    "compute_guardrails_metrics": "evaluation.guardrails_evaluation",
    "run_all_metrics": "evaluation.rag_metrics",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
