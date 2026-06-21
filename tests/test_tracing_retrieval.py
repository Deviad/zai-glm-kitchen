"""Tests for the MLA retrieval-pattern analyzer (Phase 3 / Story 5 re-scoped).

Story 5's original DSA-indexer premise was empirically rejected on 2026-06-20
(see `GLM52_SESSION_MEMORY.md`). The re-scoped goal is to trace MLA's actual
retrieval mechanism — full attention over the KV cache — using the existing
``activation_summary`` event type on ``q_nope_absorbed`` (query) and
``kv_cmpr`` (key). This test file pins:

- the signed-overlap similarity metric on top-K channels
- distance-bucket mapping against prompt length
- full analyze_retrieval pipeline on deterministic synthetic data
  with a known sentinel-positioned key
- sentinel overlap detection
- markdown rendering section
- empty-records case (defensive: no crash, empty section)
- multi-run / multi-layer isolation (queries don't score against other runs' keys)
- prefill records aren't treated as queries
"""

from __future__ import annotations

import pytest

from glm52_kitchen.tracing.retrieval import (
    BUCKET_FAR,
    BUCKET_FUTURE,
    BUCKET_MEDIUM,
    BUCKET_ORDER,
    BUCKET_RECENT,
    BUCKET_VERY_FAR,
    DEFAULT_K_STEM,
    DEFAULT_Q_STEM,
    DEFAULT_RETRIEVAL_TOPN,
    RetrievalAnalysis,
    analyze_retrieval,
    distance_bucket,
    render_markdown,
    signed_overlap,
    to_summary_dict,
)
from glm52_kitchen.tracing.schema import (
    ActivationSummaryRecord,
    PHASE_GENERATION,
    PHASE_PREFILL,
)


# --------------------------------------------------------------------------- helpers


def _q(tok: int, top: list[list[float]], *, run_id: str = "test-run", layer: int = 0,
       task: str = "coding", lang: str = "en") -> ActivationSummaryRecord:
    return ActivationSummaryRecord(
        run_id=run_id, model="m", phase=PHASE_GENERATION, token_index=tok, layer=layer,
        tensor_stem=DEFAULT_Q_STEM, n_channels=512, task_label=task, language=lang,
        topk=len(top), top_k_channels=top,
    )


def _k(tok: int, top: list[list[float]], *, run_id: str = "test-run", layer: int = 0,
       phase: str = PHASE_PREFILL, task: str = "coding", lang: str = "en") -> ActivationSummaryRecord:
    return ActivationSummaryRecord(
        run_id=run_id, model="m", phase=phase, token_index=tok, layer=layer,
        tensor_stem=DEFAULT_K_STEM, n_channels=512, task_label=task, language=lang,
        topk=len(top), top_k_channels=top,
    )


# --------------------------------------------------------------------------- signed overlap


class TestSignedOverlap:
    def test_empty_lists_returns_zero(self):
        score, n = signed_overlap([], [])
        assert score == 0.0
        assert n == 0

    def test_disjoint_channels_zero_score(self):
        q = [[7, 0.5], [3, 0.4]]
        k = [[100, 0.5], [200, 0.4]]
        score, n = signed_overlap(q, k)
        assert score == 0.0
        assert n == 0

    def test_full_overlap_signed_score(self):
        q = [[7, 0.5], [3, 0.4]]
        k = [[7, 0.6], [3, 0.5]]
        score, n = signed_overlap(q, k)
        # 0.5*0.6 + 0.4*0.5 = 0.30 + 0.20 = 0.50
        assert n == 2
        assert score == pytest.approx(0.50)

    def test_partial_overlap_only_shared_counted(self):
        q = [[7, 0.5], [3, 0.4], [99, 0.3]]
        k = [[7, 0.6], [3, -0.2], [55, 0.9]]
        score, n = signed_overlap(q, k)
        # Shared = {7, 3}. score = 0.5*0.6 + 0.4*(-0.2) = 0.30 - 0.08 = 0.22
        assert n == 2
        assert score == pytest.approx(0.22)

    def test_negative_overlap_when_opposite_signs(self):
        # If query says channel 7 is strongly +, key says channel 7 is strongly -,
        # the "overlap" is a large NEGATIVE number — this is correct behaviour:
        # the key's polarity is opposite to the query, so it's an
        # anti-correlation, not a retrieval.
        q = [[7, 0.5]]
        k = [[7, -0.6]]
        score, n = signed_overlap(q, k)
        assert n == 1
        assert score == pytest.approx(-0.30)

    def test_coefficients_are_int_coerced(self):
        # Records may arrive with string channel indices if loaded from JSON
        # without prior coercion. signed_overlap must int() them.
        q = [["7", 0.5]]
        k = [["7", 0.6]]
        score, n = signed_overlap(q, k)
        assert n == 1
        assert score == pytest.approx(0.30)


# --------------------------------------------------------------------------- distance buckets


class TestDistanceBucket:
    def test_future_when_retrieved_at_or_after_query(self):
        assert distance_bucket(10, 10, 100) == BUCKET_FUTURE
        assert distance_bucket(10, 11, 100) == BUCKET_FUTURE

    def test_recent_within_floor(self):
        # 5 tokens back: well under RECENT_FLOOR=64 even at huge prompt
        assert distance_bucket(100, 95, 10000) == BUCKET_RECENT

    def test_recent_within_5_pct_of_prompt(self):
        # 50 tokens back, prompt_len 1000 → 5% = 50, just at the threshold
        assert distance_bucket(50, 0, 1000) == BUCKET_RECENT

    def test_medium_5_to_30_pct(self):
        # 100 tokens back, prompt_len 1000 → 10% = 100, in medium
        assert distance_bucket(100, 0, 1000) == BUCKET_MEDIUM

    def test_far_30_to_70_pct(self):
        # 500 tokens back, prompt_len 1000 → 50% = 500, in far
        assert distance_bucket(500, 0, 1000) == BUCKET_FAR

    def test_very_far_past_70_pct(self):
        # 800 tokens back, prompt_len 1000 → 80% > 70%
        assert distance_bucket(800, 0, 1000) == BUCKET_VERY_FAR

    def test_no_prompt_length_falls_back_to_absolute_thresholds(self):
        # No prompt length signal → absolute thresholds kick in
        assert distance_bucket(64, 0, 0) == BUCKET_RECENT          # ≤64
        assert distance_bucket(256, 0, 0) == BUCKET_RECENT         # ≤256
        assert distance_bucket(1000, 0, 0) == BUCKET_MEDIUM        # ≤2048
        assert distance_bucket(4000, 0, 0) == BUCKET_FAR            # ≤8192
        assert distance_bucket(20000, 0, 0) == BUCKET_VERY_FAR     # >8192

    def test_bucket_order_is_complete(self):
        # Sanity: buckets enumerate exactly the categories we report on
        assert set(BUCKET_ORDER) == {
            BUCKET_RECENT, BUCKET_MEDIUM, BUCKET_FAR, BUCKET_VERY_FAR, BUCKET_FUTURE
        }


# --------------------------------------------------------------------------- analyze_retrieval end-to-end


class TestAnalyzeRetrieval:
    def test_smoke_sentinel_positioned_key_ranks_first(self):
        """The canonical smoke: a prefill key positioned with the sentinel
        channel-overlap with the gen query should rank #1 in retrieval, and
        the sentinel range should report a hit."""
        records = [
            _k(0, [[100, 0.5], [200, 0.4], [300, 0.3]]),          # no overlap
            _k(1, [[7, 0.6], [3, 0.5], [55, -0.4]]),              # sentinel-positioned
            _k(2, [[50, 0.5], [60, 0.4], [70, 0.3]]),             # no overlap
            _k(3, [[80, 0.5], [90, 0.4], [99, -0.2]]),            # tiny overlap (1 chan, opposite sign)
            _k(4, [[7, -0.1], [3, -0.1], [55, -0.1]]),            # shares 7,3 but negative → low/neg score
            _q(5, [[7, 0.5], [3, 0.4], [99, 0.3]]),
        ]
        analysis = analyze_retrieval(records, q_stem=DEFAULT_Q_STEM, k_stem=DEFAULT_K_STEM,
                                     topn=3, sentinel_range=(1, 1))
        assert len(analysis.results) == 1
        assert analysis.results[0].query_token == 5
        assert analysis.results[0].prompt_len == 5     # max prefill token_index + 1
        # Top retrieved position must be the sentinel-positioned key (token 1)
        top = analysis.results[0].top_positions
        assert top[0].position == 1
        # Signed overlap: 0.5*0.6 + 0.4*0.5 = 0.50
        assert top[0].score == pytest.approx(0.50)
        assert top[0].shared_channels == 2
        # Sentinel hit recorded (1/1)
        assert analysis.sentinel_hits == 1
        assert analysis.sentinel_total == 1

    def test_no_query_records_yields_empty_results(self):
        # Only prefill key records, no generation query records → no retrieval
        # (prefill queries aren't retrieval — they're encoding, not later lookup).
        records = [_k(0, [[7, 0.5]]), _k(1, [[7, 0.6]])]
        analysis = analyze_retrieval(records)
        assert analysis.results == []
        assert analysis.n_query_records == 0
        assert analysis.n_key_records == 2
        assert analysis.sentinel_total == 0   # never reached the loop body

    def test_no_key_records_yields_empty_results_per_query(self):
        # Query but zero earlier keys → result is empty top_positions list
        records = [_q(5, [[7, 0.5]])]
        analysis = analyze_retrieval(records)
        assert len(analysis.results) == 1
        assert analysis.results[0].top_positions == []

    def test_only_prefill_queries_dont_score(self):
        # A q_stem record in PREFILL phase shouldn't be treated as a retrieval
        # query — prefill queries are part of encoding, not later lookup.
        records = [
            _k(0, [[7, 0.6]]),
            ActivationSummaryRecord(
                run_id="test-run", model="m", phase=PHASE_PREFILL, token_index=10, layer=0,
                tensor_stem=DEFAULT_Q_STEM, n_channels=512, task_label="coding", language="en",
                topk=1, top_k_channels=[[7, 0.5]],
            ),
        ]
        analysis = analyze_retrieval(records)
        assert analysis.results == []
        # The prefill q record IS counted as a query record seen, but excluded from scoring
        assert analysis.n_query_records == 1

    def test_multi_run_isolation(self):
        # Keys from run A should NOT be candidates when scoring a query from run B.
        q_run_b = ActivationSummaryRecord(
            run_id="run-B", model="m", phase=PHASE_GENERATION, token_index=5, layer=0,
            tensor_stem=DEFAULT_Q_STEM, n_channels=512, task_label="coding", language="en",
            topk=2, top_k_channels=[[7, 0.5], [3, 0.4]],
        )
        records = [
            # run-A key at token 1 with strong overlap to run-B's query
            ActivationSummaryRecord(
                run_id="run-A", model="m", phase=PHASE_PREFILL, token_index=1, layer=0,
                tensor_stem=DEFAULT_K_STEM, n_channels=512, task_label="coding", language="en",
                topk=2, top_k_channels=[[7, 5.0], [3, 5.0]],  # would score 5 if it matched
            ),
            # run-B prefill key with weak overlap to run-B's query
            _k(0, [[7, 0.1]], run_id="run-B"),
            _q(5, [[7, 0.5]], run_id="run-B") if False else q_run_b,
        ]
        analysis = analyze_retrieval(records, topn=5)
        assert len(analysis.results) == 1
        r = analysis.results[0]
        assert r.run_id == "run-B"
        # Run-A's strongly-overlapping key should NOT be a candidate
        positions = {rp.position for rp in r.top_positions}
        assert 1 not in positions   # run-A token 1 excluded
        assert 0 in positions       # run-B token 0 included

    def test_multi_layer_isolation(self):
        # A query at layer 5 should NOT score against k records at layer 7.
        records = [
            _k(0, [[7, 5.0]], layer=7),    # would score 5 if matched
            _q(5, [[7, 0.5]], layer=5),
        ]
        analysis = analyze_retrieval(records, topn=5)
        assert len(analysis.results) == 1
        r = analysis.results[0]
        assert r.layer == 5
        # The layer-7 key shouldn't appear; no candidates → empty result
        assert r.top_positions == []

    def test_future_positions_excluded_from_scores(self):
        # k records at token_index >= query's token_index must be skipped
        # (we only retrieve EARLIER positions by definition).
        records = [
            _k(3, [[7, 5.0]]),  # earlier (token 3 < query token 7)
            _k(7, [[7, 5.0]]),  # same as query → future bucket
            _k(10, [[7, 5.0]]), # after query → future bucket
            _q(7, [[7, 0.5]]),
        ]
        analysis = analyze_retrieval(records, topn=5)
        r = analysis.results[0]
        positions = {rp.position for rp in r.top_positions}
        assert positions == {3}  # only the earlier position counted

    def test_topn_caps_ranked_positions(self):
        # Build 10 prefill k records each overlapping with the query on a
        # different channel; topn=3 should keep only the top 3 by score.
        records: list[ActivationSummaryRecord] = []
        for i in range(10):
            # Each key shares ONE channel with the query; the further-to-right
            # keys get larger magnitudes so they score higher.
            records.append(_k(i, [[7, 0.1 + 0.1 * i]]))
        records.append(_q(10, [[7, 0.5]]))
        analysis = analyze_retrieval(records, topn=3)
        r = analysis.results[0]
        assert len(r.top_positions) == 3
        # Top 3 by score should be tokens 9, 8, 7 (descending magnitude)
        positions = [rp.position for rp in r.top_positions]
        assert positions == [9, 8, 7]

    def test_duplicate_kv_records_collapse_to_one(self):
        # kv_cmpr fires as both matmul-output and cache-write; the same
        # (token_index, layer) appears multiple times. analyze_retrieval
        # must collapse these so we don't double-count same-position candidates.
        records = [
            _k(1, [[7, 0.6]]),
            _k(1, [[7, 0.6]]),  # exact duplicate (re-emitted)
            _q(5, [[7, 0.5]]),
        ]
        analysis = analyze_retrieval(records, topn=5)
        r = analysis.results[0]
        positions = [rp.position for rp in r.top_positions]
        # Token 1 counts as ONE candidate (deduped), even though presented twice
        assert positions.count(1) == 1

    def test_empty_records_yields_empty_analysis(self):
        analysis = analyze_retrieval([])
        assert analysis.results == []
        # Empty markdown → caller should skip
        assert render_markdown(analysis) == ""

    def test_sentinel_range_math(self):
        # Multiple (query, layer) pairs, only some hit the sentinel range
        records = [
            # Layer 0: query 5 → only key with overlap is token 1 (sentinel) ✓
            _k(0, [[100, 0.5]], layer=0),
            _k(1, [[7, 0.6], [3, 0.5]], layer=0),
            _q(5, [[7, 0.5]], layer=0),
            # Layer 1: query 5 → only key with overlap is token 4 (NOT in sentinel) ✗
            _k(0, [[100, 0.5]], layer=1),
            _k(4, [[7, 0.6], [3, 0.5]], layer=1),
            _q(5, [[7, 0.5]], layer=1),
        ]
        analysis = analyze_retrieval(records, topn=5, sentinel_range=(1, 1))
        # We expect 1/2 sentinel hits: layer-0 query hit, layer-1 query missed
        assert analysis.sentinel_total == 2
        assert analysis.sentinel_hits == 1


# --------------------------------------------------------------------------- serialization


class TestRetrievalSummaryDict:
    def test_round_trip_summary_dict_fields(self):
        records = [
            _k(1, [[7, 0.6], [3, 0.5]]),
            _q(5, [[7, 0.5], [3, 0.4]]),
        ]
        analysis = analyze_retrieval(records, topn=3, sentinel_range=(1, 1))
        d = to_summary_dict(analysis)
        assert d["q_stem"] == "q_nope_absorbed"
        assert d["k_stem"] == "kv_cmpr"
        assert d["topn"] == 3
        assert d["sentinel_range"] == [1, 1]
        assert d["n_query_records_seen"] == 1
        assert d["n_key_records_seen"] == 1
        assert d["n_results"] == 1
        assert d["sentinel_hits"] == 1
        assert d["sentinel_total"] == 1
        assert d["sentinel_hit_rate"] == 1.0
        assert len(d["sample_results"]) == 1
        sr = d["sample_results"][0]
        assert sr["query_token"] == 5
        assert sr["layer"] == 0
        assert sr["top_positions"][0]["position"] == 1
        # Bucket counts enumerated in canonical order
        assert set(d["bucket_counts"].keys()) == set(BUCKET_ORDER)
        assert set(d["bucket_fractions"].keys()) == set(BUCKET_ORDER)

    def test_summary_dict_without_sentinel_range(self):
        records = [_k(1, [[7, 0.6]]), _q(5, [[7, 0.5]])]
        analysis = analyze_retrieval(records, topn=3)
        d = to_summary_dict(analysis)
        assert d["sentinel_range"] is None
        assert d["sentinel_hits"] == 0
        assert d["sentinel_total"] == 0
        assert d["sentinel_hit_rate"] is None


# --------------------------------------------------------------------------- markdown


class TestRetrievalMarkdown:
    def test_empty_analysis_renders_empty_string(self):
        analysis = RetrievalAnalysis(
            q_stem=DEFAULT_Q_STEM, k_stem=DEFAULT_K_STEM, topn=DEFAULT_RETRIEVAL_TOPN,
        )
        assert render_markdown(analysis) == ""

    def test_populated_analysis_renders_sections(self):
        records = [
            _k(0, [[100, 0.5]]),
            _k(1, [[7, 0.6], [3, 0.5]]),
            _q(5, [[7, 0.5], [3, 0.4]]),
        ]
        analysis = analyze_retrieval(records, topn=3, sentinel_range=(1, 1))
        md = render_markdown(analysis)
        assert "## MLA retrieval analysis" in md
        assert "### Distance buckets" in md
        assert "### Sentinel section retrieval" in md
        assert "### Sample retrieval results" in md
        assert "**1** / 1 (query, layer) pairs" in md
        assert "1.000" in md   # hit rate
        # Sentinel-positioned key appears in the sample table
        assert "1@" in md      # pos=1@score

    def test_markdown_without_sentinel_range_omits_sentinel_section(self):
        records = [
            _k(0, [[100, 0.5]]),
            _k(1, [[7, 0.6]]),
            _q(5, [[7, 0.5]]),
        ]
        analysis = analyze_retrieval(records, topn=3)
        md = render_markdown(analysis)
        assert "### Distance buckets" in md
        # No sentinel section when range isn't provided
        assert "### Sentinel section retrieval" not in md

    def test_markdown_includes_approximation_caveat(self):
        # Every retrieval markdown section must explain the approximation
        # so downstream readers know top-K overlap != full softmax(QK) attention.
        records = [_k(1, [[7, 0.6]]), _q(5, [[7, 0.5]])]
        analysis = analyze_retrieval(records, topn=3)
        md = render_markdown(analysis)
        assert "Approximation only" in md
        assert "top-K is" in md


# --------------------------------------------------------------------------- defaults


class TestDefaults:
    """Module-level defaults pinned so analyzer + tracer stem conventions match."""

    def test_default_q_stem(self):
        # The absorbed query in MLA — what each generation-step token asks for.
        assert DEFAULT_Q_STEM == "q_nope_absorbed"

    def test_default_k_stem(self):
        # The lora-compressed KV — what each prefill token offers.
        assert DEFAULT_K_STEM == "kv_cmpr"

    def test_default_topn(self):
        # Keep top-10 retrieved positions per (query, layer) by default
        assert DEFAULT_RETRIEVAL_TOPN == 10
