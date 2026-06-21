"""Trace schema for GLM-5.2 MoE expert-routing traces.

Defines the canonical JSONL record shapes emitted by the tracer backend and
consumed by :mod:`glm52_kitchen.tracing.analyze`.

Schema version 1 record (one per ``(token, layer)`` routing event)::

    {
      "schema_version": 1,
      "event": "moe_topk",
      "run_id": "2026-06-20-glm52-coding-en-001",
      "model": "GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest",
      "phase": "prefill",                # "prefill" | "generation"
      "token_index": 123,                # position within the phase
      "token_id": 1234,                  # optional, llama.cpp token id
      "token_text": "def",               # optional, detokenized piece
      "layer": 42,                       # MoE block index (0..n_layer_all-1)
      "experts": [17, 92, 104, 3],       # selected expert ids, len == n_expert_used
      "weights": [0.21, 0.17, 0.14, 0.12],
      "router_entropy": 1.92,            # optional
      "n_expert_used": 4,
      "n_expert": 256,                   # optional total expert count
      "task_label": "coding",
      "language": "en",                  # en|it|zh|es|fr|de|pt|<multi>
      "script": "Latin",                 # Latin|Han|mixed
      "prompt_family": "coding",         # coding|science-reading|...
      "test_id": "coding_01_iterative_merge_sort"  # optional, from smoke suite
    }

The companion metadata sidecar (``<trace.jsonl>.meta.json``) holds run-level
provenance: model path/hash, command line, prompt path/hash, thinking mode,
queue/backpressure counters, perf, etc. Heavy fields stay out of the JSONL so
the per-token stream stays compact and machine-parseable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SCHEMA_VERSION = 1

EVENT_MOE_ROUTING = "moe_topk"  # canonical routing event (experts + weights + entropy)
# Story 6 AC: distinct event type for bounded activation summaries. Carries
# top-K channels (by magnitude) + per-token norm/stat summaries for a named
# activation tensor at a given (layer, token_index).
EVENT_ACTIVATION_SUMMARY = "activation_summary"

PHASE_PREFILL = "prefill"
PHASE_GENERATION = "generation"

SUPPORTED_LANGUAGES = ("en", "it", "zh", "es", "fr", "de", "pt")
SUPPORTED_BACKPRESSURE = ("block", "drop", "sample")

# Story 6 AC: default activation tensor patterns emitted by the C++ tracer
# when --trace-activations is passed without an explicit pattern list.
# l_out-N is the per-layer inter-block residual: a single, interpretability-
# meaningful "what each layer contributes" signal. Other interesting stems
# include kqv_out-N / ffn_out-N / ffn_moe_out-N; users can override via
# --trace-activations kqv_out,ffn_moe_out,...
DEFAULT_ACTIVATION_STEMS = ("l_out",)

# Default top-K channels recorded per activation tensor (per token).
DEFAULT_ACTIVATION_TOPK = 10


class TraceSchemaError(ValueError):
    """Raised when a trace record violates the schema."""


@dataclass
class MoeRoutingRecord:
    """One ``(token, layer)`` MoE routing event."""

    run_id: str
    model: str
    phase: str
    token_index: int
    layer: int
    experts: list[int]
    weights: list[float]
    task_label: str
    language: str
    schema_version: int = SCHEMA_VERSION
    event: str = EVENT_MOE_ROUTING
    n_expert_used: int | None = None
    n_expert: int | None = None
    router_entropy: float | None = None
    token_id: int | None = None
    token_text: str | None = None
    script: str | None = None
    prompt_family: str | None = None
    test_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise TraceSchemaError(
                f"unsupported schema_version {self.schema_version}, expected {SCHEMA_VERSION}"
            )
        if self.event != EVENT_MOE_ROUTING:
            raise TraceSchemaError(
                f"unsupported event {self.event!r}, expected {EVENT_MOE_ROUTING!r}"
            )
        if self.phase not in (PHASE_PREFILL, PHASE_GENERATION):
            raise TraceSchemaError(
                f"phase must be {PHASE_PREFILL!r} or {PHASE_GENERATION!r}, got {self.phase!r}"
            )
        if self.token_index < 0:
            raise TraceSchemaError(f"token_index must be >= 0, got {self.token_index}")
        if self.layer < 0:
            raise TraceSchemaError(f"layer must be >= 0, got {self.layer}")
        if not isinstance(self.experts, list) or not self.experts:
            raise TraceSchemaError("experts must be a non-empty list of ints")
        if not isinstance(self.weights, list) or len(self.weights) != len(self.experts):
            raise TraceSchemaError(
                f"weights length {len(self.weights) if isinstance(self.weights, list) else 'NA'}"
                f" != experts length {len(self.experts)}"
            )
        if self.n_expert_used is not None and self.n_expert_used != len(self.experts):
            raise TraceSchemaError(
                f"n_expert_used {self.n_expert_used} != len(experts) {len(self.experts)}"
            )
        if self.n_expert_used is None:
            self.n_expert_used = len(self.experts)
        if not self.run_id or not self.model or not self.task_label or not self.language:
            raise TraceSchemaError(
                "run_id, model, task_label and language are required and must be non-empty"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop None values to keep the JSONL compact; analyzer treats absent as absent.
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoeRoutingRecord":
        # Tolerant of extra keys (future schema additions) — only consume what we know.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        # integers may arrive as numpy ints or strings; coerce defensively.
        for int_field in ("schema_version", "token_index", "layer", "token_id", "n_expert", "n_expert_used"):
            if int_field in kwargs and kwargs[int_field] is not None:
                try:
                    kwargs[int_field] = int(kwargs[int_field])
                except (TypeError, ValueError) as exc:
                    raise TraceSchemaError(f"{int_field} is not int-coercible: {exc}") from exc
        if "experts" in kwargs and kwargs["experts"] is not None:
            kwargs["experts"] = [int(e) for e in kwargs["experts"]]
        if "weights" in kwargs and kwargs["weights"] is not None:
            kwargs["weights"] = [float(w) for w in kwargs["weights"]]
        if "router_entropy" in kwargs and kwargs["router_entropy"] is not None:
            kwargs["router_entropy"] = float(kwargs["router_entropy"])
        return cls(**kwargs)


@dataclass
class ActivationSummaryRecord:
    """One bounded activation summary for a named tensor at ``(layer, token)``.

    Story 6 AC: not a full activation dump — only the top-K channels by
    magnitude plus per-token norm/stat summaries, so analysis of a 12-token/
    75-layer prefill doesn't blow up the JSONL to O(millions of floats).
    Emitted by the C++ tracer when ``--trace-activations <stems>`` is set
    (absent by default — see DEFAULT_ACTIVATION_STEMS).
    """

    run_id: str
    model: str
    phase: str
    token_index: int
    layer: int
    tensor_stem: str
    n_channels: int
    task_label: str
    language: str
    schema_version: int = SCHEMA_VERSION
    event: str = EVENT_ACTIVATION_SUMMARY
    # top-K channels by |magnitude|, descending: [[channel_idx, magnitude], ...]
    top_k_channels: list[list[float]] = field(default_factory=list)
    topk: int | None = None
    l2_norm: float | None = None
    mean: float | None = None
    std: float | None = None
    max_abs: float | None = None
    token_id: int | None = None
    script: str | None = None
    prompt_family: str | None = None
    test_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise TraceSchemaError(
                f"unsupported schema_version {self.schema_version}, expected {SCHEMA_VERSION}"
            )
        if self.event != EVENT_ACTIVATION_SUMMARY:
            raise TraceSchemaError(
                f"unsupported event {self.event!r}, expected {EVENT_ACTIVATION_SUMMARY!r}"
            )
        if self.phase not in (PHASE_PREFILL, PHASE_GENERATION):
            raise TraceSchemaError(
                f"phase must be {PHASE_PREFILL!r} or {PHASE_GENERATION!r}, got {self.phase!r}"
            )
        if self.token_index < 0:
            raise TraceSchemaError(f"token_index must be >= 0, got {self.token_index}")
        if self.layer < 0:
            raise TraceSchemaError(f"layer must be >= 0, got {self.layer}")
        if not self.tensor_stem:
            raise TraceSchemaError("tensor_stem is required and must be non-empty")
        if self.n_channels <= 0:
            raise TraceSchemaError(f"n_channels must be > 0, got {self.n_channels}")
        for pair in self.top_k_channels:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TraceSchemaError(
                    f"top_k_channels entries must be [channel_idx, magnitude] pairs, got {pair!r}"
                )
        if not self.run_id or not self.model or not self.task_label or not self.language:
            raise TraceSchemaError(
                "run_id, model, task_label and language are required and must be non-empty"
            )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationSummaryRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        for int_field in ("schema_version", "token_index", "layer", "token_id", "n_channels", "topk"):
            if int_field in kwargs and kwargs[int_field] is not None:
                try:
                    kwargs[int_field] = int(kwargs[int_field])
                except (TypeError, ValueError) as exc:
                    raise TraceSchemaError(f"{int_field} is not int-coercible: {exc}") from exc
        # top_k_channels: accept list-of-lists or list-of-tuples; coerce to list[list[float]]
        if "top_k_channels" in kwargs and kwargs["top_k_channels"] is not None:
            coerced: list[list[float]] = []
            for pair in kwargs["top_k_channels"]:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise TraceSchemaError(
                        f"top_k_channels entries must be [channel_idx, magnitude] pairs, got {pair!r}"
                    )
                coerced.append([int(pair[0]), float(pair[1])])
            kwargs["top_k_channels"] = coerced
        for float_field in ("l2_norm", "mean", "std", "max_abs"):
            if float_field in kwargs and kwargs[float_field] is not None:
                kwargs[float_field] = float(kwargs[float_field])
        return cls(**kwargs)


@dataclass
class RunMetadata:
    """Run-level provenance written to ``<trace>.meta.json``."""

    run_id: str
    model: str
    model_path: str
    command_line: str
    prompt_sha256: str
    task_label: str
    language: str
    script: str
    prompt_family: str
    thinking_mode: str  # "disabled" | "minimal" | "enabled"
    reasoning_effort: str  # "none" | "low" | "medium" | "high"
    max_new_tokens: int
    # prompt_path is optional: single-prompt mode (-p) writes the prompt file
    # path here, but batched mode (--trace-prompts) sources prompts from a
    # JSONL of PromptSpecs and has no single prompt file to point at.
    prompt_path: str | None = None
    # Story 9 AC: model sha256 prefix over the first 1 MiB of the model file
    # (cheap provenance, not a full-file fingerprint). The other reproducibility
    # fields below (model_size_bytes, started_at, ended_at, command_line,
    # prompt_sha256) already existed in the schema but were populated with
    # placeholders by the C++ writer until 2026-06-20.
    model_sha256_prefix: str | None = None
    # Story 9 AC: total size across all multi-shard parts (e.g. for GLM-5.2
    # this is 232 GB total; per-shard shard-1 size like 9.4 MiB looks tiny).
    # Absent for single-shard models or older sidecars.
    model_total_size_bytes: int | None = None
    phase_modes: dict[str, bool] = field(default_factory=lambda: {"prefill": True, "generation": True})
    trace_layers: str = "0..n_layer_all-1"
    trace_max_tokens: int | None = None
    backpressure: str = "sample"
    queue_size: int = 8192
    records_written: int = 0
    records_dropped: int = 0
    records_sampled: int = 0
    prompt_token_count: int | None = None
    gen_token_count: int | None = None
    perf_prompt_eval_per_sec: float | None = None
    perf_gen_per_sec: float | None = None
    # Story 8 AC: raw perf counters backing the per-sec rates above. Populated
    # by the C++ tracer via llama_perf_context(ctx); absent for old sidecars.
    perf_prompt_eval_ms: float | None = None
    perf_eval_ms: float | None = None
    perf_n_prompt_eval: int | None = None
    perf_n_eval: int | None = None
    # Story 8 AC: n_expert_total at the sidecar level (mirrors per-record
    # `n_expert` so the analyzer can label experts as #X of N even when
    # legacy records omit it).
    n_expert_total: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    wall_seconds: float | None = None
    cli_path: str | None = None
    cli_build: str | None = None
    model_size_bytes: int | None = None
    schema_version: int = SCHEMA_VERSION
    test_id: str | None = None
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None or k in (
            "records_written", "records_dropped", "records_sampled", "queue_size"
        )}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunMetadata":
        """Construct :class:`RunMetadata` from a sidecar JSON dict.

        Unknown keys are silently dropped (forward/backward-compat: new C++
        fields don't break old code, old sidecars with unknown-to-Python
        keys don't break new code). Required-from-schema-but-actually-optional
        fields like ``prompt_path`` default to None via the dataclass itself.
        """
        known = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


def iter_records(path: str) -> Iterable[MoeRoutingRecord | ActivationSummaryRecord]:
    """Stream a trace JSONL file as validated records.

    Yields :class:`MoeRoutingRecord` for ``event == moe_topk`` rows and
    :class:`ActivationSummaryRecord` for ``event == activation_summary`` rows.
    Blank lines and comment lines (starting with ``#``) are skipped. Future
    event types are skipped with a tolerance counter rather than raising,
    so adding new event types does not break the analyzer.
    """
    skipped_unknown = 0
    counts = {EVENT_MOE_ROUTING: 0, EVENT_ACTIVATION_SUMMARY: 0}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            import json

            obj = json.loads(line)
            ev = obj.get("event")
            if ev == EVENT_MOE_ROUTING:
                counts[EVENT_MOE_ROUTING] += 1
                yield MoeRoutingRecord.from_dict(obj)
            elif ev == EVENT_ACTIVATION_SUMMARY:
                counts[EVENT_ACTIVATION_SUMMARY] += 1
                yield ActivationSummaryRecord.from_dict(obj)
            else:
                skipped_unknown += 1
    if skipped_unknown:
        # surfaced via the import-time logger would be nicer, but keep this module
        # dependency-free; callers can wrap with their own accounting if needed.
        pass
