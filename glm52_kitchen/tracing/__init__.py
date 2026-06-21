"""GLM-5.2 tracing framework.

Phase 1 (MoE expert routing tracer) implementation.

Public API:

* :class:`schema.MoeRoutingRecord` – one ``(token, layer)`` routing event
* :class:`schema.RunMetadata` – run-level provenance (written to ``.meta.json``)
* :class:`writer.TraceWriter` – bounded async JSONL writer with backpressure
* :mod:`analyze` – aggregate traces into a Markdown report + summary JSON
* :mod:`compare` – compare multiple runs/models side-by-side (Story 8)
* :mod:`synth` – deterministic synthetic trace generator (tests + demo reports)

The C++ backend that captures real ``ffn_moe_topk`` / ``ffn_moe_weights`` tensors
lives in the vendored patched llama.cpp submodule at
``vendor/llama.cpp/examples/trace-moe/`` and is built as
``vendor/llama.cpp/build-metal/bin/llama-trace-moe``. It emits the same JSONL
schema this framework consumes.
"""

from .schema import (
    EVENT_ACTIVATION_SUMMARY,
    EVENT_MOE_ROUTING,
    DEFAULT_ACTIVATION_STEMS,
    DEFAULT_ACTIVATION_TOPK,
    PHASE_GENERATION,
    PHASE_PREFILL,
    SCHEMA_VERSION,
    SUPPORTED_BACKPRESSURE,
    SUPPORTED_LANGUAGES,
    ActivationSummaryRecord,
    MoeRoutingRecord,
    RunMetadata,
    TraceSchemaError,
)
from .writer import TraceWriter
from .analyze import (
    aggregate,
    build_summary,
    render_markdown,
    resolve_traces,
    write_report,
)
from .compare import compare, render_markdown as render_comparison, write_comparison
from .synth import generate_records, write_synth_trace, iter_suite
from .retrieval import (
    DEFAULT_K_STEM,
    DEFAULT_Q_STEM,
    DEFAULT_RETRIEVAL_TOPN,
    RetrievalAnalysis,
    analyze_retrieval,
    render_markdown as render_retrieval_markdown,
    to_summary_dict as retrieval_to_summary_dict,
)

__all__ = [
    # schema
    "SCHEMA_VERSION",
    "EVENT_MOE_ROUTING",
    "EVENT_ACTIVATION_SUMMARY",
    "DEFAULT_ACTIVATION_STEMS",
    "DEFAULT_ACTIVATION_TOPK",
    "PHASE_PREFILL",
    "PHASE_GENERATION",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_BACKPRESSURE",
    "MoeRoutingRecord",
    "ActivationSummaryRecord",
    "RunMetadata",
    "TraceSchemaError",
    # writer
    "TraceWriter",
    # analyze
    "aggregate",
    "build_summary",
    "render_markdown",
    "resolve_traces",
    "write_report",
    # compare
    "compare",
    "render_comparison",
    "write_comparison",
    # synth
    "generate_records",
    "write_synth_trace",
    "iter_suite",
    # retrieval (Phase 3 / Story 5 re-scoped to MLA activation patterns)
    "DEFAULT_Q_STEM",
    "DEFAULT_K_STEM",
    "DEFAULT_RETRIEVAL_TOPN",
    "RetrievalAnalysis",
    "analyze_retrieval",
    "render_retrieval_markdown",
    "retrieval_to_summary_dict",
]
