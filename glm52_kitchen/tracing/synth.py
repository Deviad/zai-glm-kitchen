"""Synthetic trace generator.

Emits canonical :data:`~glm52_kitchen.tracing.schema.EVENT_MOE_ROUTING` records for a
smoke-suite prompt record, deterministic by seed.  Used to:

* unit-test the analyzer + writer without loading the 232GB GLM-5.2 model;
* produce a sample ``reports/glm52_moe_trace_report.md`` so the report pipeline
  is demonstrably end-to-end before a live run.

The routing distribution is mildly task- and language-biased (coding prefers a
smaller set of experts; non-Latin scripts shift weight slightly) so analyzer
overlap/specialization metrics produce interesting, non-degenerate output.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    DEFAULT_ACTIVATION_STEMS,
    DEFAULT_ACTIVATION_TOPK,
    PHASE_GENERATION,
    PHASE_PREFILL,
    RunMetadata,
    MoeRoutingRecord,
    ActivationSummaryRecord,
    SCHEMA_VERSION,
)
from .writer import TraceWriter

_DEFAULT_N_EXPERT = 256
_DEFAULT_N_USED = 4
_DEFAULT_N_LAYERS = 79
_DEFAULT_N_CHANNELS = 6144  # GLM-5.2 embedding_length; real tensor wide=6144

_TASK_BIAS: dict[str, list[int]] = {
    "coding": [3, 17, 42, 104, 201, 9, 77, 55, 88, 140],
    "computer_science": [17, 42, 104, 55, 88, 201, 3, 140, 9, 77],
    "cybersecurity": [104, 201, 9, 77, 55, 88, 140, 3, 17, 42],
    "math": [30, 31, 32, 33, 60, 61, 150, 151, 152, 153],
    "physics": [40, 41, 42, 43, 70, 71, 160, 161, 162, 163],
    "engineering": [50, 51, 52, 53, 80, 81, 170, 171, 172, 173],
    "chemistry": [60, 61, 62, 63, 90, 91, 180, 181, 182, 183],
    "science-reading": [40, 60, 70, 90, 150, 160, 170, 180],
}
# Shift bias per language (small systematic difference so overlap < 1)
_LANG_SHIFT: dict[str, int] = {
    "en": 0, "it": 1, "es": 2, "fr": 3, "de": 4, "pt": 5, "zh": 7,
}


def _seed_for(test_id: str, language: str, phase: str, salt: str = "glm52-trace-smoke") -> int:
    h = hashlib.sha256(f"{salt}|{test_id}|{language}|{phase}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _biased_pool(task: str, language: str, n_expert: int = _DEFAULT_N_EXPERT) -> list[int]:
    base = _TASK_BIAS.get(task, list(range(0, n_expert, 17)))
    shift = _LANG_SHIFT.get(language, 0)
    pool = [(e + shift) % n_expert for e in base]
    return pool


def generate_records(
    prompt_record: dict[str, Any],
    *,
    n_prefill: int = 24,
    n_gen: int = 16,
    n_layers: int = _DEFAULT_N_LAYERS,
    n_expert: int = _DEFAULT_N_EXPERT,
    n_expert_used: int = _DEFAULT_N_USED,
    salt: str = "glm52-trace-smoke",
    model: str = "GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest",
    run_id: str | None = None,
) -> list[MoeRoutingRecord]:
    test_id = prompt_record.get("test_id", "unknown")
    language = prompt_record.get("language", "en")
    task = prompt_record.get("prompt_family") or prompt_record.get("domain") or "misc"
    script = prompt_record.get("script", "Latin")
    pool = _biased_pool(task, language, n_expert)

    out: list[MoeRoutingRecord] = []
    rid = run_id or f"synth-{test_id}-{language}"

    for phase, n_tok in ((PHASE_PREFILL, n_prefill), (PHASE_GENERATION, n_gen)):
        seed = _seed_for(test_id, language, phase, salt)
        rng = random.Random(seed)
        # bias: pick from pool ~70% of the time, random ~30%
        for ti in range(n_tok):
            chosen: list[int] = []
            while len(chosen) < n_expert_used:
                if rng.random() < 0.7 and pool:
                    e = rng.choice(pool)
                else:
                    e = rng.randrange(n_expert)
                if e not in chosen:
                    chosen.append(e)
            # softmax-ish weights
            raw = [rng.random() ** 2 + 0.05 for _ in chosen]
            tot = sum(raw)
            weights = [round(w / tot, 4) for w in raw]
            # re-normalize after rounding
            drift = 1.0 - sum(weights)
            if abs(drift) > 1e-6:
                weights[0] = round(weights[0] + drift, 4)
            layer = rng.randrange(n_layers)
            ent = round(_entropy(weights), 4)
            out.append(
                MoeRoutingRecord(
                    run_id=rid,
                    model=model,
                    phase=phase,
                    token_index=ti,
                    layer=layer,
                    experts=[int(e) for e in chosen],
                    weights=weights,
                    router_entropy=ent,
                    n_expert_used=n_expert_used,
                    n_expert=n_expert,
                    task_label=task,
                    language=language,
                    script=script,
                    prompt_family=task,
                    test_id=test_id,
                    token_text=None,
                    token_id=None,
                )
            )
    return out


def _entropy(weights: list[float]) -> float:
    import math

    tot = sum(weights)
    if tot <= 0:
        return 0.0
    h = 0.0
    for w in weights:
        if w <= 0:
            continue
        p = w / tot
        h -= p * math.log2(p)
    return h


def generate_activation_records(
    prompt_record: dict[str, Any],
    *,
    n_prefill: int = 24,
    n_gen: int = 16,
    n_layers: int = _DEFAULT_N_LAYERS,
    n_channels: int = _DEFAULT_N_CHANNELS,
    topk: int = DEFAULT_ACTIVATION_TOPK,
    stems: tuple[str, ...] = DEFAULT_ACTIVATION_STEMS,
    salt: str = "glm52-trace-smoke-activation",
    model: str = "GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest",
    run_id: str | None = None,
) -> list[ActivationSummaryRecord]:
    """Synthetic bounded activation summaries (Story 6 AC helper).

    Mirrors :func:`generate_records` shape (per-token records per layer), but
    emits :class:`ActivationSummaryRecord` instead. Each record carries:
    - top-K channels by magnitude (deterministic, biased by task+lang so
      analyzer overlap metrics produce non-degenerate output)
    - per-token L2 norm / mean / std / max_abs (deterministic)
    """
    test_id = prompt_record.get("test_id", "unknown")
    language = prompt_record.get("language", "en")
    task = prompt_record.get("prompt_family") or prompt_record.get("domain") or "misc"
    script = prompt_record.get("script", "Latin")
    rid = run_id or f"synth-{test_id}-{language}"

    # Per (task, language, stem), pick a small biased channel pool (~32 channels)
    # so top-k channels frequently repeat across tokens but not all identical.
    def _bias_seed(stem: str) -> int:
        return _seed_for(test_id, language, stem, salt)
    rng_per_stem = {stem: random.Random(_bias_seed(stem)) for stem in stems}
    channel_pools = {
        stem: [
            rng_per_stem[stem].randrange(n_channels)
            for _ in range(64)
        ]
        for stem in stems
    }

    out: list[ActivationSummaryRecord] = []
    for phase, n_tok in ((PHASE_PREFILL, n_prefill), (PHASE_GENERATION, n_gen)):
        seed = _seed_for(test_id, language, f"{phase}-act-", salt)
        rng = random.Random(seed)
        for ti in range(n_tok):
            # Each layer in this token's prefill/gen emits one ActivationSummaryRecord
            # per stem. To keep synth volume manageable, emit only for n_layers//2 sampled
            # layers (every other layer) — same density the C++ tracer tends to use
            # with --trace-activations to keep prefill JSONL bounded.
            for layer in range(0, n_layers, 2):
                for stem in stems:
                    pool = channel_pools[stem]
                    # 70% biased picks (concentrated), 30% random (variety)
                    chosen: set[int] = set()
                    while len(chosen) < topk:
                        if rng.random() < 0.7:
                            chosen.add(pool[rng.randrange(len(pool))])
                        else:
                            chosen.add(rng.randrange(n_channels))
                    # Magnitudes: exponential-ish so a few are clearly dominant
                    mags = sorted(
                        (round(rng.expovariate(1.0), 4) + 0.001 for _ in range(topk)),
                        reverse=True,
                    )
                    top_k_channels = [[int(ch), float(m)] for ch, m in zip(sorted(chosen), mags)]
                    # Per-token summary stats: synthetic but realistic ranges
                    l2_norm = round(rng.uniform(1.0, 6.0), 4)
                    mean = round(rng.uniform(-0.05, 0.05), 4)
                    std = round(rng.uniform(0.6, 1.4), 4)
                    max_abs = round(max(m for _, m in top_k_channels), 4)
                    out.append(
                        ActivationSummaryRecord(
                            run_id=rid,
                            model=model,
                            phase=phase,
                            token_index=ti,
                            layer=layer,
                            tensor_stem=stem,
                            n_channels=n_channels,
                            topk=topk,
                            top_k_channels=top_k_channels,
                            l2_norm=l2_norm,
                            mean=mean,
                            std=std,
                            max_abs=max_abs,
                            task_label=task,
                            language=language,
                            script=script,
                            prompt_family=task,
                            test_id=test_id,
                        )
                    )
    return out


def write_synth_trace(
    prompt_record: dict[str, Any],
    out_path: str | str,
    *,
    n_prefill: int = 24,
    n_gen: int = 16,
    n_layers: int = _DEFAULT_N_LAYERS,
    n_expert: int = _DEFAULT_N_EXPERT,
    n_expert_used: int = _DEFAULT_N_USED,
    salt: str = "glm52-trace-smoke",
    model: str = "GLM-5.2-mixed-IQ2S-experts-IQ4NL-rest",
    backpressure: str = "sample",
    cli_path: str = "synth",
    thinking_mode: str = "disabled",
    reasoning_effort: str = "none",
    max_new_tokens: int = 16,
    activations: bool = False,
    activation_stems: tuple[str, ...] = DEFAULT_ACTIVATION_STEMS,
    activation_topk: int = DEFAULT_ACTIVATION_TOPK,
) -> str:
    """Generate records and write them (with a metadata sidecar) to ``out_path``.

    When ``activations=True``, also synthesizes bounded activation-summary
    records (Story 6 AC) interleaved with routing records in the same JSONL,
    so the analyzer pipeline can be tested end-to-end without a real model.
    """
    records = generate_records(
        prompt_record,
        n_prefill=n_prefill,
        n_gen=n_gen,
        n_layers=n_layers,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        salt=salt,
        model=model,
    )
    if activations:
        act_records = generate_activation_records(
            prompt_record,
            n_prefill=n_prefill,
            n_gen=n_gen,
            n_layers=n_layers,
            n_channels=_DEFAULT_N_CHANNELS,
            topk=activation_topk,
            stems=activation_stems,
            salt=salt + "-activation",
            model=model,
        )
    else:
        act_records = []
    test_id = prompt_record.get("test_id", "unknown")
    language = prompt_record.get("language", "en")
    task = prompt_record.get("prompt_family") or prompt_record.get("domain") or "misc"
    prompt = prompt_record.get("prompt", "")
    import hashlib

    md = RunMetadata(
        run_id=f"synth-{test_id}-{language}",
        model=model,
        model_path="synthetic",
        command_line=f"python -m glm52_kitchen.tracing.synth --out {out_path}",
        prompt_path="synthetic",
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        task_label=task,
        language=language,
        script=prompt_record.get("script", "Latin"),
        prompt_family=task,
        test_id=test_id,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
        phase_modes={"prefill": True, "generation": True},
        trace_layers=f"0..{n_layers - 1}",
        trace_max_tokens=n_prefill + n_gen,
        backpressure=backpressure,
        queue_size=8192,
        prompt_token_count=n_prefill,
        gen_token_count=n_gen,
        cli_path=cli_path,
        cli_build="synthetic generator",
        notes=["synthetic trace; no real inference run"],
    )
    w = TraceWriter(out_path, md, backpressure=backpressure).open()
    for rec in records:
        w.push(rec)
    for rec in act_records:
        w.push(rec)
    md_out = w.close()
    total_written = len(records) + len(act_records)
    assert md_out.records_written == total_written, (
        f"writer dropped {total_written - md_out.records_written} records in synth path"
    )
    return out_path


def iter_suite(path: str) -> Iterable[dict[str, Any]]:
    import json

    p = Path(path)
    if p.is_dir():
        for f in sorted(p.glob("*.jsonl")):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
    else:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


__all__ = [
    "generate_records",
    "generate_activation_records",
    "write_synth_trace",
    "iter_suite",
    "SCHEMA_VERSION",
    "_DEFAULT_N_EXPERT",
    "_DEFAULT_N_USED",
    "_DEFAULT_N_LAYERS",
    "_DEFAULT_N_CHANNELS",
]
