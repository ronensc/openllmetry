"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection
from opentelemetry.instrumentation.langchain.config import Config
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.langchain.version import __version__


from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346", "langchain-core > 0.1.0")


class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        traceloopCallbackHandler = TraceloopCallbackHandler(tracer)
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(traceloopCallbackHandler),
        )

        wrap_function_wrapper(
            module="langchain_community.llms.openai",
            name="completion_with_retry",
            wrapper=_CompletionWithRetryWrapper(traceloopCallbackHandler),
        )

    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks", "BaseCallbackManager.__init__")
        unwrap("langchain_community.llms.openai", "completion_with_retry")


class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_manager: "TraceloopCallbackHandler"):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._callback_manager)):
                break
        else:
            instance.add_handler(self._callback_manager, True)


class _CompletionWithRetryWrapper:
    def __init__(self, callback_manager: "TraceloopCallbackHandler"):
        self._callback_manager = callback_manager

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ) -> None:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
        from opentelemetry.trace.propagation import set_span_in_context

        run_id = kwargs["run_manager"].run_id
        span_holder = self._callback_manager.spans[run_id]

        extra_headers = kwargs.get("extra_headers", {})

        ctx = set_span_in_context(span_holder.span)
        TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
        # TraceContextTextMapPropagator().inject(extra_headers, context=span_holder.context)

        kwargs["extra_headers"] = extra_headers
        return wrapped(*args, **kwargs)
