"""End-to-end smoke test for the refinement pipeline (no API calls).

Exercises the full chain:
  1. Pre-populate an :class:`OutcomesCache` with v1 entries (mimicking what
     a completed PO-LEU round-1 training would have written).
  2. Write a synthetic ``val_per_event.json`` + a tiny ``records.pkl`` to
     a temp run dir.
  3. Run :mod:`scripts.identify_failure_events` → ``failure_events.json``
     should appear with the expected top-K events.
  4. Run :func:`src.outcomes.refine.refine_outcomes` over each (cust, alt)
     pair using stub LLM clients, write v2_refined entries to the cache
     under :func:`build_cache_prompt_version`-derived keys.
  5. Verify the cascade lookup in :func:`src.data.batching.assemble_batch`
     prefers v2_refined entries by directly probing the same composite
     keys via the public helper.

Pure unit-test scope — no SQL injection of provider SDKs, no network.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from src.outcomes.cache import OutcomesCache
from src.outcomes.generate import (
    GenerationResult,
    build_cache_prompt_version,
)
from src.outcomes.prompts import REFINED_PROMPT_VERSION
from src.outcomes.refine import refine_outcomes


# ---------------------------------------------------------------------------
# Tiny stub LLM client (same shape as real Anthropic / OpenAI / Gemini).
# ---------------------------------------------------------------------------


class _CannedClient:
    def __init__(self, model_id: str, replies: list[str]) -> None:
        self.model_id = model_id
        self._replies = list(replies)

    def generate(
        self, messages, *, temperature, top_p, max_tokens, seed,
    ) -> GenerationResult:
        if not self._replies:
            raise AssertionError("_CannedClient out of replies")
        return GenerationResult(
            text=self._replies.pop(0),
            finish_reason="end_turn",
            model_id=self.model_id,
        )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _alt(j: int) -> dict:
    return {
        "title": f"Alternative {j}",
        "category": "household",
        "price": str(10.0 + j),
        "popularity_rank": "high" if j == 0 else "medium",
    }


def _records(N: int = 4, J: int = 3) -> list[dict]:
    out = []
    for i in range(N):
        c_d = f"Customer {i} is a 30-year-old in a small city."
        out.append({
            "customer_id": f"cust_{i}",
            "chosen_asin": f"asin_{i}_0",
            "choice_asins": [f"asin_{i}_{j}" for j in range(J)],
            "chosen_idx": 0,
            "c_d": c_d,
            "alt_texts": [_alt(j) for j in range(J)],
        })
    return out


def _per_event_payload(records: list[dict], worst_idx: list[int]) -> dict:
    """Synthetic val_per_event.json. ``worst_idx`` get high NLL."""
    rows = []
    for i, r in enumerate(records):
        nll = 4.0 if i in worst_idx else 0.5
        p_chosen = 0.02 if i in worst_idx else 0.6
        rows.append({
            "event_idx": i,
            "customer_id": r["customer_id"],
            "asin_chosen": r["chosen_asin"],
            "c_star": int(r["chosen_idx"]),
            "p_chosen": p_chosen,
            "nll": nll,
            "top1_correct": False if i in worst_idx else True,
        })
    return {"per_event": rows, "n_events": len(rows)}


def _cache_v1(
    cache: OutcomesCache, records: list[dict], K: int, seed: int,
    writer_model_id: str, prompt_version: str = "v2",
) -> None:
    """Populate the cache with deterministic v1 outcomes for every (cust, alt)."""
    for r in records:
        for j, asin in enumerate(r["choice_asins"]):
            cache_pv = build_cache_prompt_version(
                prompt_version=prompt_version, K=K,
                model_id=writer_model_id, c_d=r["c_d"],
            )
            outcomes = [f"v1 outcome {idx} for {r['customer_id']} alt{j}." for idx in range(K)]
            cache.put_outcomes(
                r["customer_id"], asin, seed, cache_pv,
                outcomes=outcomes,
                metadata={"prompt_version": prompt_version, "model_id": writer_model_id},
            )


# ---------------------------------------------------------------------------
# 1. identify_failure_events.py — single-process invocation
# ---------------------------------------------------------------------------


def test_identify_failure_events_writes_top_k(tmp_path: Path) -> None:
    """Top-K worst val events appear in failure_events.json with full alt info."""
    records = _records(N=5, J=3)
    run_dir = tmp_path / "poleu_run"
    run_dir.mkdir()

    # records.pkl bundle (the val split is what identify reads).
    with (run_dir / "records.pkl").open("wb") as fh:
        pickle.dump({"train": [], "val": records, "test": []}, fh)

    # val_per_event.json with events 1 + 3 marked as worst.
    payload = _per_event_payload(records, worst_idx=[1, 3])
    (run_dir / "val_per_event.json").write_text(json.dumps(payload))

    # Invoke main() of identify_failure_events directly.
    import sys
    from scripts.identify_failure_events import main as identify_main
    saved = sys.argv
    sys.argv = [
        "identify_failure_events",
        "--run-dir", str(run_dir),
        "--top-k", "2",
    ]
    try:
        rc = identify_main()
    finally:
        sys.argv = saved

    assert rc == 0
    out = json.loads((run_dir / "failure_events.json").read_text())
    assert out["k_selected"] == 2
    selected_idx = sorted(e["event_idx"] for e in out["events"])
    assert selected_idx == [1, 3]
    # Each selected event carries the full choice set + alt_texts.
    for e in out["events"]:
        assert len(e["choice_asins"]) == 3
        assert len(e["alt_texts"]) == 3
        assert e["alt_texts"][0]["title"] == "Alternative 0"


# ---------------------------------------------------------------------------
# 2. refine_outcomes — write v2_refined entries that the cascade can find
# ---------------------------------------------------------------------------


def test_refine_writes_cascade_findable_entries(tmp_path: Path) -> None:
    """End-to-end: refine -> cache.get_outcomes(v2_refined key) returns new outs."""
    K = 3
    seed = 7
    writer_model = "stub-writer-1"
    records = _records(N=3, J=2)
    cache = OutcomesCache(tmp_path / "outcomes.sqlite")
    _cache_v1(cache, records, K=K, seed=seed, writer_model_id=writer_model)

    # Refine event 0 alt 0 only — minimal smoke. Critic returns 2/2 (below
    # threshold), so writer is invoked.
    rec = records[0]
    alt = rec["alt_texts"][0]
    asin = rec["choice_asins"][0]

    critic = _CannedClient(
        "stub-critic-1",
        ['{"plausibility": 2, "diversity": 2, "notes": "too generic"}'],
    )
    revised_text = (
        "I save fifteen dollars on the grocery bill this trip.\n"
        "I move through checkout in under a minute, fitting a tight schedule.\n"
        "I feel confident the brand matches what my partner already trusts."
    )
    writer = _CannedClient(writer_model, [revised_text])

    # Pull v1 outcomes that refine_outcomes will critique.
    v1_pv = build_cache_prompt_version(
        prompt_version="v2", K=K, model_id=writer_model, c_d=rec["c_d"],
    )
    v1_payload = cache.get_outcomes(rec["customer_id"], asin, seed, v1_pv)
    v1_outcomes = list(v1_payload["outcomes"])

    result = refine_outcomes(
        c_d=rec["c_d"], alt=alt, outcomes=v1_outcomes, K=K,
        writer_client=writer, critic_client=critic, seed=seed,
        accept_threshold=4,
    )
    assert result.skipped is False
    assert len(result.revised_outcomes) == K

    # Driver does this — write v2_refined entry under the composite key.
    v2_pv = build_cache_prompt_version(
        prompt_version=REFINED_PROMPT_VERSION, K=K,
        model_id=writer_model, c_d=rec["c_d"],
    )
    cache.put_outcomes(
        rec["customer_id"], asin, seed, v2_pv,
        outcomes=result.revised_outcomes,
        metadata={"prompt_version": REFINED_PROMPT_VERSION},
    )

    # The cascade probes v2_refined first, then falls back to v2.
    # (1) v2_refined hit on the refined alt:
    cascade_hit = cache.get_outcomes(rec["customer_id"], asin, seed, v2_pv)
    assert cascade_hit is not None
    assert cascade_hit["outcomes"] == result.revised_outcomes

    # (2) v2_refined miss on a non-refined alt — must fall back to v2.
    other_asin = records[1]["choice_asins"][0]
    other_v2_pv = build_cache_prompt_version(
        prompt_version=REFINED_PROMPT_VERSION, K=K,
        model_id=writer_model, c_d=records[1]["c_d"],
    )
    miss = cache.get_outcomes(records[1]["customer_id"], other_asin, seed, other_v2_pv)
    assert miss is None  # no v2_refined entry written for this alt

    # The fallback entry is reachable under v2:
    other_v1_pv = build_cache_prompt_version(
        prompt_version="v2", K=K,
        model_id=writer_model, c_d=records[1]["c_d"],
    )
    fallback = cache.get_outcomes(records[1]["customer_id"], other_asin, seed, other_v1_pv)
    assert fallback is not None
    assert len(fallback["outcomes"]) == K


# ---------------------------------------------------------------------------
# 3. assemble_batch cascade — verify the helper contract holds
# ---------------------------------------------------------------------------


def test_v1_prompt_version_contract_writer_matches_reader() -> None:
    """The v1 prompt_version writer + reader defaults MUST match.

    Concretely:
      - WRITER side: src/data/batching.py:assemble_batch's ``prompt_version``
        default — the value cache entries get tagged with when round-1 PO-LEU
        training runs.
      - READER side: scripts/refine_outcomes.py's ``--v1-prompt-version``
        argparse default — the value the refinement loop assumes when it
        reads round-1 entries before critique-and-revise.
      - CASCADE side: run_full_evaluation.sh's
        ``--prompt-version-cascade v2_refined v1`` — the fallback the
        round-3 training cascades to when v2_refined misses.

    All three must use the same string. The fix in commit XXX was: writer
    default is "v1" (assemble_batch signature), so reader default must
    also be "v1" and the cascade fallback must also be "v1". A mismatch
    silently regenerates every outcome under a fresh prompt_version,
    burning thousands of LLM calls.
    """
    import argparse
    import inspect
    import re
    from pathlib import Path

    from src.data.batching import assemble_batch

    # WRITER: introspect assemble_batch signature.
    sig = inspect.signature(assemble_batch)
    writer_default = sig.parameters["prompt_version"].default

    # READER: parse refine_outcomes.py's argparse to find --v1-prompt-version
    # without invoking main(). We import the parser-builder directly.
    from scripts.refine_outcomes import _build_parser
    parser: argparse.ArgumentParser = _build_parser()
    reader_default = parser.get_default("v1_prompt_version")

    assert writer_default == reader_default, (
        f"prompt_version writer/reader contract broken: "
        f"assemble_batch defaults to {writer_default!r} but "
        f"refine_outcomes --v1-prompt-version defaults to {reader_default!r}. "
        f"They must match or refinement silently fails to find round-1 entries."
    )

    # CASCADE: grep run_full_evaluation.sh for the fallback token.
    sh_path = Path(__file__).resolve().parents[1] / "run_full_evaluation.sh"
    sh = sh_path.read_text()
    m = re.search(
        r"--prompt-version-cascade\s+v2_refined\s+(\S+)",
        sh,
    )
    assert m is not None, (
        "could not find '--prompt-version-cascade v2_refined <fallback>' "
        f"in {sh_path}; refine pipeline may be misconfigured."
    )
    cascade_fallback = m.group(1).strip().rstrip("\\")
    assert cascade_fallback == writer_default, (
        f"cascade fallback in run_full_evaluation.sh is "
        f"{cascade_fallback!r} but assemble_batch writes under "
        f"{writer_default!r}. Round-3 PO-LEU-REFINED training would "
        f"miss every cache and fresh-generate every outcome."
    )


def test_cascade_helper_matches_runtime_composite() -> None:
    """build_cache_prompt_version stays in sync with generate_outcomes' inline composite.

    If this drifts, refine_outcomes.py would write v2_refined entries under
    a key that the round-3 training cannot find — silent breakage.
    """
    import hashlib
    from src.outcomes.generate import _sanitize_model_id
    pv, K, model_id, c_d = "v2_refined", 3, "Claude/Sonnet-4.6", "ctx"
    expected = (
        f"{pv}-K{K}-"
        f"{_sanitize_model_id(model_id)}-cd"
        f"{hashlib.sha256(c_d.encode('utf-8')).hexdigest()[:16]}"
    )
    assert build_cache_prompt_version(
        prompt_version=pv, K=K, model_id=model_id, c_d=c_d,
    ) == expected
