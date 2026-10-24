"""Optional evaluation tracing; absent telemetry never prevents a local run."""
from contextlib import nullcontext
from functools import lru_cache
import logging
import os


@lru_cache(maxsize=1)
def _configured_tracer():
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        return None
    try:
        import logfire

        logfire.configure(token=token, service_name="contextforge-rag-evaluation")
        return logfire
    except Exception as exc:
        logging.getLogger(__name__).warning("Evaluation tracing unavailable: %s", exc)
        return None


class _Telemetry:
    def span(self, message, **attributes):
        tracer = _configured_tracer()
        return tracer.span(message, **attributes) if tracer else nullcontext()

    def _log(self, level, message, **attributes):
        tracer = _configured_tracer()
        if tracer:
            getattr(tracer, level)(message, **attributes)
        else:
            getattr(logging.getLogger("evaluation"), level)(message)

    def info(self, message, **attributes):
        self._log("info", message, **attributes)

    def error(self, message, **attributes):
        self._log("error", message, **attributes)


logfire = _Telemetry()
