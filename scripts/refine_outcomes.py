"""Critique-and-revise outcomes for PO-LEU's worst validation events.

Reads ``failure_events.json`` (from :mod:`scripts.identify_failure_events`)
and a populated v1 :class:`OutcomesCache`, then for each (customer, alt)
pair in every failure event runs the critique-and-revise loop from
:mod:`src.outcomes.refine` and writes the revised outcomes back to the
cache under :data:`src.outcomes.prompts.REFINED_PROMPT_VERSION`.

Two LLM clients:
  * **writer**: same family as the v1 generator (default Anthropic). Used
    for the revise call.
  * **critic**: a different family (default OpenAI; falls back to Gemini
    if ``OPENAI_API_KEY`` is not set; falls back to writer-as-critic if
    neither is available, with a loud warning).

Output sidecar (``refined_outcomes.json``) records before/after texts plus
critic scores so the experiment is auditable without re-querying the cache.

Usage::

    python -m scripts.refine_outcomes \\
        --failure-events results_data/poleu_50cust_seed7_no_residual/failure_events.json \\
        --outcomes-cache outcomes_cache/outcomes.sqlite \\
        --output results_data/poleu_50cust_seed7_no_residual/refined_outcomes.json \\
        --writer anthropic --critic openai

The downstream :mod:`scripts.run_dataset` invocation must pass
``--prompt-version-cascade v2_refined v1`` (or whatever pair matches your
v1 cache tag) so the cache cascade picks up the new entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_CLOUD_PROJECT
# are visible when this driver is invoked from a sub-shell.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("refine_outcomes")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refine_outcomes.py",
        description=__doc__,
    )
    p.add_argument(
        "--failure-events",
        type=Path,
        required=True,
        help="failure_events.json from scripts.identify_failure_events.",
    )
    p.add_argument(
        "--outcomes-cache",
        type=Path,
        default=Path("outcomes_cache/outcomes.sqlite"),
        help="Path to the SQLite outcomes cache. Default outcomes_cache/outcomes.sqlite.",
    )
    p.add_argument(
        "--K", type=int, default=3,
        help="Outcomes per alternative. Must match the round-1 K. Default 3.",
    )
    p.add_argument(
        "--seed", type=int, default=7,
        help="Seed used in the round-1 cache. Default 7.",
    )
    p.add_argument(
        "--v1-prompt-version", default="v1",
        help=(
            "Base prompt_version of the round-1 cache entries. The composite "
            "key folds in K + writer model_id + c_d hash. Default 'v1' — "
            "this MUST match the prompt_version assemble_batch uses when "
            "writing v1 entries (currently the default 'v1' from its "
            "function signature). The constant prompts.PROMPT_VERSION='v2' "
            "is intentionally a separate concept (template-content "
            "version) and is NOT what's stored in the cache key."
        ),
    )
    p.add_argument(
        "--writer", choices=["anthropic"], default="anthropic",
        help="Writer LLM family — same as round-1 generator.",
    )
    p.add_argument(
        "--critic", choices=["openai", "gemini", "writer"], default="writer",
        help=(
            "Critic LLM family. Default 'writer' — same Anthropic client "
            "used by the writer (no Gemini / OpenAI calls in the loop; "
            "all spend stays on Anthropic). Trade-off: the critic is in "
            "the same family that wrote v1, so its plausibility/diversity "
            "scores are less independent than a cross-family critic. Use "
            "'gemini' for Gemini 2.5 Flash (cross-family, cheap, fast) or "
            "'openai' for GPT-5 (reasoning model — needs max_tokens ≥ 1500)."
        ),
    )
    p.add_argument(
        "--accept-threshold", type=int, default=4,
        help=(
            "If both critic scores >= this, skip the revise call (keep "
            "originals). Default 4 (out of 5)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sidecar path. Defaults to <run-dir>/refined_outcomes.json.",
    )
    p.add_argument(
        "--max-events", type=int, default=None,
        help="Hard cap on events to refine (debug). Default: all.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _build_writer_client(family: str) -> Any:
    from src.outcomes.generate import AnthropicLLMClient
    if family == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "refine_outcomes: --writer anthropic requires ANTHROPIC_API_KEY"
            )
        model_id = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return AnthropicLLMClient(model_id=model_id, api_key=api_key)
    raise SystemExit(f"refine_outcomes: unsupported writer family: {family!r}")


def _build_critic_client(family: str, writer_client: Any) -> Any:
    if family == "writer":
        logger.warning(
            "critic=writer: critic is the same client as the writer — "
            "this is an echo-chamber fallback intended for testing only."
        )
        return writer_client
    if family == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY not set; falling back to gemini critic."
            )
            return _build_critic_client("gemini", writer_client)
        from src.outcomes._openai_client import OpenAILLMClient
        model_id = os.environ.get("OPENAI_MODEL", "gpt-5")
        return OpenAILLMClient(model_id=model_id)
    if family == "gemini":
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            logger.warning(
                "GOOGLE_CLOUD_PROJECT not set; falling back to writer-as-critic."
            )
            return writer_client
        from src.outcomes._gemini_client import GeminiLLMClient
        model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiLLMClient(model_id=model_id)
    raise SystemExit(f"refine_outcomes: unsupported critic family: {family!r}")


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.failure_events.exists():
        logger.error("--failure-events not found: %s", args.failure_events)
        return 2

    payload = json.loads(args.failure_events.read_text())
    events = payload.get("events", [])
    if args.max_events is not None:
        events = events[: max(0, int(args.max_events))]
    if not events:
        logger.warning("no failure events; nothing to refine.")
        return 0
    logger.info("loaded %d failure events from %s", len(events), args.failure_events)

    output_path: Path = args.output or (
        args.failure_events.parent / "refined_outcomes.json"
    )

    from src.outcomes.cache import OutcomesCache
    from src.outcomes.generate import build_cache_prompt_version
    from src.outcomes.prompts import REFINED_PROMPT_VERSION
    from src.outcomes.refine import refine_outcomes as do_refine

    cache = OutcomesCache(args.outcomes_cache)
    writer_client = _build_writer_client(args.writer)
    critic_client = _build_critic_client(args.critic, writer_client)
    writer_model_id = getattr(writer_client, "model_id", "unknown")
    critic_model_id = getattr(critic_client, "model_id", "unknown")
    logger.info(
        "writer=%s (%s) critic=%s (%s)",
        args.writer, writer_model_id, args.critic, critic_model_id,
    )

    bookkeeping: list[dict] = []
    n_pairs = 0
    n_refined = 0
    n_skipped = 0
    n_cache_miss = 0
    n_revise_failures = 0

    for ev in events:
        c_d = str(ev["c_d"])
        customer_id = str(ev["customer_id"])
        choice_asins = list(ev.get("choice_asins", []))
        alt_texts = list(ev.get("alt_texts", []))
        if not choice_asins or len(choice_asins) != len(alt_texts):
            logger.warning(
                "event %s: malformed choice/alt arrays; skipping",
                ev.get("event_idx"),
            )
            continue

        for j, (asin, alt) in enumerate(zip(choice_asins, alt_texts)):
            n_pairs += 1
            v1_cache_pv = build_cache_prompt_version(
                prompt_version=str(args.v1_prompt_version),
                K=int(args.K),
                model_id=writer_model_id,
                c_d=c_d,
            )
            cached = cache.get_outcomes(
                customer_id, str(asin), int(args.seed), v1_cache_pv,
            )
            if cached is None:
                n_cache_miss += 1
                logger.warning(
                    "v1 cache miss for (event=%s, alt=%d, asin=%s) — "
                    "skipping (run_dataset must populate v1 first)",
                    ev.get("event_idx"), j, asin,
                )
                continue
            v1_outcomes = list(cached.get("outcomes", []))
            if len(v1_outcomes) != int(args.K):
                logger.warning(
                    "v1 cache entry for (event=%s, alt=%d) has %d outcomes "
                    "(expected K=%d); skipping",
                    ev.get("event_idx"), j, len(v1_outcomes), args.K,
                )
                continue

            try:
                result = do_refine(
                    c_d=c_d,
                    alt=alt,
                    outcomes=v1_outcomes,
                    K=int(args.K),
                    writer_client=writer_client,
                    critic_client=critic_client,
                    seed=int(args.seed),
                    accept_threshold=int(args.accept_threshold),
                )
            except Exception as exc:  # noqa: BLE001
                n_revise_failures += 1
                logger.warning(
                    "refine failed for (event=%s, alt=%d, asin=%s): %s",
                    ev.get("event_idx"), j, asin, exc,
                )
                continue

            v2_cache_pv = build_cache_prompt_version(
                prompt_version=REFINED_PROMPT_VERSION,
                K=int(args.K),
                model_id=writer_model_id,
                c_d=c_d,
            )
            metadata = dict(result.metadata)
            metadata.update({
                "v1_cache_prompt_version": v1_cache_pv,
                "model_id": writer_model_id,
                "skipped_revise": bool(result.skipped),
                "critic_plausibility": int(result.critique.plausibility),
                "critic_diversity": int(result.critique.diversity),
                "critic_notes": result.critique.notes,
            })
            cache.put_outcomes(
                customer_id, str(asin), int(args.seed), v2_cache_pv,
                outcomes=result.revised_outcomes,
                metadata=metadata,
            )

            if result.skipped:
                n_skipped += 1
            else:
                n_refined += 1

            bookkeeping.append({
                "event_idx": ev.get("event_idx"),
                "customer_id": customer_id,
                "asin": str(asin),
                "alt_idx": j,
                "skipped": bool(result.skipped),
                "critic": {
                    "plausibility": int(result.critique.plausibility),
                    "diversity": int(result.critique.diversity),
                    "notes": result.critique.notes,
                    "model_id": result.critique.model_id,
                },
                "v1_outcomes": v1_outcomes,
                "v2_outcomes": result.revised_outcomes,
            })

    summary = {
        "n_events": len(events),
        "n_pairs_total": n_pairs,
        "n_revised": n_refined,
        "n_skipped_above_threshold": n_skipped,
        "n_v1_cache_miss": n_cache_miss,
        "n_revise_failures": n_revise_failures,
        "writer_model_id": writer_model_id,
        "critic_model_id": critic_model_id,
        "accept_threshold": int(args.accept_threshold),
        "v1_prompt_version": str(args.v1_prompt_version),
        "v2_prompt_version": REFINED_PROMPT_VERSION,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {"summary": summary, "items": bookkeeping}, indent=2,
    ))
    logger.info(
        "refinement complete: %d refined, %d skipped (passed threshold), "
        "%d v1 cache misses, %d revise failures (out of %d pairs in "
        "%d events). bookkeeping: %s",
        n_refined, n_skipped, n_cache_miss, n_revise_failures,
        n_pairs, len(events), output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
