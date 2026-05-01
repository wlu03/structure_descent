"""Wave 10 end-to-end orchestration driver (design doc §4).

CLI-driven orchestration of the full PO-LEU pipeline. Takes an adapter
name, a customer budget, and an output dir; produces a fully materialized
training run plus evaluation + interpretability reports.

The driver always uses the real Anthropic LLM client and the real
``sentence-transformers`` encoder. Stub clients used to live behind a
``--stub-llm`` flag but were removed: they shared the outcomes-cache
key schema with real calls, which meant a switch from stub to real
could silently serve stale stub entries into real training. Stubs still
exist in the codebase (``StubLLMClient``, ``StubEncoder``) for unit
tests, but the production driver refuses them by construction.

Usage
-----

    python scripts/run_dataset.py \\
        --adapter amazon \\
        --n-customers 100 \\
        --n-epochs 1 \\
        --batch-size 32 \\
        --output-dir reports/amazon_smoke

Requires the ``ANTHROPIC_API_KEY`` environment variable and the
``anthropic`` + ``sentence-transformers`` packages to be installed.

See ``NOTES.md`` ("Wave 10 — glue modules") for the full design contract,
especially the val/test z_d fit caveat (val/test currently reuses the
train-only fit statistics, deferred to Wave 11/12).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env before any LLM client is constructed so ANTHROPIC_API_KEY,
# OPENAI_API_KEY, GOOGLE_CLOUD_PROJECT, etc. are visible to os.environ.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# Heavy / optional imports kept lazy inside main() for ``anthropic`` and
# ``sentence_transformers`` -- we want a helpful error message (not an
# ImportError at module load) when those packages are missing.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the argparse parser defining the CLI contract (design doc §4)."""
    parser = argparse.ArgumentParser(
        prog="run_dataset.py",
        description="Wave 10 end-to-end orchestration driver for PO-LEU.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        choices=["amazon", "synthetic", "mobility_boston"],
        help="Dataset adapter name. 'synthetic' requires an explicit "
             "--dataset-config (no built-in synthetic YAML ships).",
    )
    parser.add_argument(
        "--n-customers",
        type=int,
        required=True,
        help="Appendix-C subsample budget. No full-population fallback; if "
             "subsample fails we exit with code 1.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to the default config YAML.",
    )
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="Training epochs; default 5. Early stopping on val NLL with patience 5 applies; 1 is the cost-capped smoke default.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Overrides the config's train.batch_size.",
    )
    parser.add_argument("--min-events-per-customer", type=int, default=5)
    parser.add_argument(
        "--drop-pool-starved-events",
        action="store_true",
        help=(
            "Drop events whose available-negative pool is smaller than "
            "J-1 instead of cycle-padding the choice set. Cleaner "
            "leaderboard (every choice set is J genuinely-distinct "
            "ASINs) at the cost of losing a few events from thin "
            "customers / very-late-timeline positions. Off by default "
            "— the legacy cyclic-padding path stays so existing runs "
            "stay bit-identical."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports; defaults to reports/<adapter>_<timestamp>.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Outcomes per alternative. Default: config.outcomes.K.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Dataset YAML path. Defaults to configs/datasets/<adapter>.yaml.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["temporal", "cold_start"],
        default="temporal",
        help=(
            "Which split to use. 'temporal' = per-customer time split "
            "(default; every customer appears in train+val+test). "
            "'cold_start' = between-customer split (val/test customers "
            "have no train events — measures generalization to unseen users)."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the cold-start partition. Defaults to --seed.",
    )
    parser.add_argument(
        "--tabular-residual",
        choices=["yaml", "true", "false"],
        default="yaml",
        help=(
            "Override configs/default.yaml model.tabular_residual.enabled. "
            "'yaml' (default) honors the YAML; 'true' / 'false' force the "
            "Sifringer feature-partition residual on / off. Used by "
            "run_full_evaluation.sh to drive the PO-LEU vs PO-LEU-RESIDUAL "
            "leaderboard pair from a single YAML."
        ),
    )
    parser.add_argument(
        "--add-event-time-to-c-d",
        action="store_true",
        help=(
            "Render a per-event time-of-day phrase (weekday + daypart + "
            "weekend flag) into c_d via build_context_string's current_time "
            "kwarg. Off by default (preserves Amazon snapshots bit-"
            "identical); flip on for mobility-style datasets where 'when' "
            "is a strong choice predictor."
        ),
    )
    parser.add_argument(
        "--add-event-origin-to-c-d",
        action="store_true",
        help=(
            "Render a per-event origin context line into c_d ('Just came "
            "from home' / 'their workplace' / 'a <category> place'). Off "
            "by default; flip on for mobility-style datasets where 'where "
            "from' is a strong predictor of 'where to'."
        ),
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help=(
            "Run a 2-round identify→refine→retrain pipeline in one shot. "
            "Round 1 = initial PO-LEU training (artifacts land in "
            "``<output-dir>/round1/``). Round 2 = identify the worst val "
            "events, run the critic+reviser loop, then retrain with the "
            "refined cascade prepended (``v3_refined → <original cascade>``); "
            "artifacts land in ``<output-dir>/`` matching a normal run. "
            "Refined-outcome bookkeeping (``failure_events.json`` + "
            "``refined_outcomes.json``) lives in ``<output-dir>/`` for "
            "inspection."
        ),
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=30,
        help="Number of worst val events to refine (default 30).",
    )
    parser.add_argument(
        "--refine-accept-threshold",
        type=int,
        default=4,
        help=(
            "Critic accept threshold (1–5). Pairs with both critic scores "
            "≥ threshold AND no flagged weak positions are skipped. "
            "Default 4."
        ),
    )
    parser.add_argument(
        "--refine-per-outcome",
        choices=["on", "off"],
        default="on",
        help=(
            "When 'on' (default), reviser only rewrites positions the "
            "critic flagged via weak_outcome_indices; strong outcomes "
            "are preserved byte-identical via splice. 'off' falls back "
            "to monolithic K-sentence rewrite (legacy v2_refined)."
        ),
    )
    parser.add_argument(
        "--refine-critic",
        choices=["writer", "anthropic", "gemini", "openai"],
        default="writer",
        help=(
            "Critic LLM family. 'writer' (default) reuses the writer's "
            "client (cheapest, but family-correlated); cross-family "
            "options give independent scoring at extra API spend."
        ),
    )
    parser.add_argument(
        "--prompt-version-cascade",
        nargs="+",
        default=None,
        metavar="VERSION",
        help=(
            "Curriculum-refinement cascade. When set, assemble_batch first "
            "tries each version's outcomes_cache for a hit; only the LAST "
            "version is allowed to call the LLM on a full miss. Use "
            "`--prompt-version-cascade v2_refined v2` to fold refined "
            "outcomes (round-3) into a PO-LEU-REFINED training run while "
            "reusing the v2 cache for non-failure events. When unset, "
            "assemble_batch uses the legacy single prompt_version path."
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_dataset_config(args: argparse.Namespace) -> Path:
    """Return the dataset YAML path (explicit override wins)."""
    if args.dataset_config is not None:
        return Path(args.dataset_config)
    if args.adapter == "synthetic":
        raise SystemExit(
            "--adapter synthetic requires --dataset-config <path> "
            "(no built-in synthetic YAML ships with the repo)."
        )
    return REPO_ROOT / "configs" / "datasets" / f"{args.adapter}.yaml"


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the output dir, defaulting to reports/<adapter>_<ts>."""
    if args.output_dir is not None:
        out = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = REPO_ROOT / "reports" / f"{args.adapter}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _omega_stats(weights: np.ndarray | None) -> dict[str, float]:
    """Compute min/median/max/mean/std of a weight array."""
    if weights is None or len(weights) == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(w)),
        "median": float(np.median(w)),
        "max": float(np.max(w)),
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
    }


def _p_reduction_reason(adapter: Any) -> str:
    """Diagnose which canonical one-hot vocabularies shrank below spec width.

    §2.1 nominal widths: age_bucket=6, income_bucket=5, city_size=4,
    household_size=5, plus 6 continuous dims -> canonical p=26. When a
    dataset adapter declares a smaller vocabulary (or ships an empty
    external_lookup), the corresponding one-hot slice shrinks and
    ``batch_train.z_d`` carries ``effective_p < 26``. This helper inspects
    :meth:`adapter.categorical_vocabularies` and returns a human-readable
    string summarizing every shortfall, or ``"none"`` when everything is
    canonical.
    """
    try:
        vocabs = adapter.categorical_vocabularies()
    except Exception:  # pragma: no cover - adapter may not implement it
        return "adapter does not expose categorical_vocabularies()"

    reasons: list[str] = []
    city = getattr(vocabs, "city_size", ())
    if len(city) < 4:
        reasons.append(
            f"city_size vocabulary reduced to {len(city)} label(s) "
            f"(canonical 4)"
        )
    age = getattr(vocabs, "age_bucket", ())
    if len(age) < 6:
        reasons.append(
            f"age_bucket vocabulary reduced to {len(age)} label(s) "
            f"(canonical 6)"
        )
    income = getattr(vocabs, "income_bucket", ())
    if len(income) < 5:
        reasons.append(
            f"income_bucket vocabulary reduced to {len(income)} label(s) "
            f"(canonical 5)"
        )
    hh = getattr(vocabs, "household_size_categories", ())
    if len(hh) < 5:
        reasons.append(
            f"household_size_categories vocabulary reduced to {len(hh)} "
            f"label(s) (canonical 5)"
        )
    if not reasons:
        return "none"
    return "; ".join(reasons)


def _build_llm_and_encoder(args: argparse.Namespace, config: dict) -> tuple[Any, Any]:
    """Instantiate the real LLM client + sentence encoder.

    Provider selection is env-var driven so the same orchestrator can fan
    out across providers without CLI surgery:

    * ``LLM_PROVIDER=anthropic`` (default). Reads ``ANTHROPIC_API_KEY``;
      ``ANTHROPIC_MODEL`` (default ``claude-sonnet-4-6``) controls the
      model id.
    * ``LLM_PROVIDER=gemini``. Reads ``GEMINI_API_KEY`` (or
      ``GOOGLE_API_KEY``); ``GEMINI_MODEL`` (default
      ``gemini-2.5-flash``) controls the model id.
    * ``LLM_PROVIDER=openai``. Reads ``OPENAI_API_KEY``;
      ``OPENAI_MODEL`` (default ``gpt-5``) controls the model id.

    The driver is real-only: stubs used to live behind a ``--stub-llm``
    flag but shared the outcomes-cache key schema with real calls, which
    silently fed stale stub entries into real training runs. Each
    provider's model_id is folded into the cache key
    (``build_cache_prompt_version``), so different providers / models
    never collide in the cache.
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()

    # Encoder is always sentence-transformers regardless of LLM provider.
    try:
        import sentence_transformers  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        sys.stderr.write(
            "run_dataset.py: the `sentence_transformers` package is required "
            "but not installed. Install it with "
            "`pip install sentence-transformers`.\n"
        )
        sys.exit(2)

    from src.outcomes.encode import SentenceTransformersEncoder, StubEncoder
    from src.outcomes.generate import StubLLMClient

    enc_cfg = (config.get("outcomes") or {}).get("encoder") or {}
    encoder: Any = SentenceTransformersEncoder(
        model_id=enc_cfg.get("model_id",
                             "sentence-transformers/all-mpnet-base-v2"),
        max_length=int(enc_cfg.get("max_length", 64)),
        pooling=enc_cfg.get("pooling", "mean"),
    )

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.stderr.write("run_dataset.py: ANTHROPIC_API_KEY missing.\n")
            sys.exit(2)
        try:
            import anthropic  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            sys.stderr.write("run_dataset.py: install `anthropic`.\n")
            sys.exit(2)
        from src.outcomes.generate import AnthropicLLMClient
        model_id = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        llm_client: Any = AnthropicLLMClient(model_id=model_id, api_key=api_key)

    elif provider == "gemini":
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            sys.stderr.write(
                "run_dataset.py: GEMINI_API_KEY (or GOOGLE_API_KEY) missing.\n"
            )
            sys.exit(2)
        try:
            import google.genai  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            sys.stderr.write("run_dataset.py: install `google-genai`.\n")
            sys.exit(2)
        from src.outcomes._gemini_client import GeminiLLMClient
        model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        llm_client = GeminiLLMClient(model_id=model_id, api_key=api_key)

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.stderr.write("run_dataset.py: OPENAI_API_KEY missing.\n")
            sys.exit(2)
        try:
            import openai  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            sys.stderr.write("run_dataset.py: install `openai`.\n")
            sys.exit(2)
        from src.outcomes._openai_client import OpenAILLMClient
        model_id = os.environ.get("OPENAI_MODEL", "gpt-5")
        llm_client = OpenAILLMClient(model_id=model_id, api_key=api_key)

    else:
        sys.stderr.write(
            f"run_dataset.py: LLM_PROVIDER={provider!r} unrecognised "
            f"(expected one of: anthropic, gemini, openai).\n"
        )
        sys.exit(2)

    logger.info(
        "LLM provider=%s model_id=%s", provider, llm_client.model_id,
    )

    if isinstance(llm_client, StubLLMClient):
        raise RuntimeError(
            "Production driver refuses StubLLMClient; run_dataset.py is "
            "real-LLM-only by construction."
        )
    if isinstance(encoder, StubEncoder):
        raise RuntimeError(
            "Production driver refuses StubEncoder; run_dataset.py is "
            "real-encoder-only by construction."
        )
    return llm_client, encoder


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _run_pipeline_once(args: argparse.Namespace) -> int:
    """Orchestrate the Wave 10 pipeline. Returns a POSIX-style exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    torch.manual_seed(int(args.seed))

    # ---- 1. resolve paths + config -------------------------------------
    out_dir = _resolve_output_dir(args)
    logger.info("output_dir=%s", out_dir)

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    config = _load_yaml(config_path)

    # ``getattr`` default keeps in-process callers (e.g. smoke tests that
    # build a Namespace by hand) working without listing every CLI flag.
    tabular_residual_choice = getattr(args, "tabular_residual", "yaml")
    if tabular_residual_choice != "yaml":
        forced = tabular_residual_choice == "true"
        model_block = config.setdefault("model", {})
        tab_block = model_block.setdefault("tabular_residual", {})
        tab_block["enabled"] = forced
        logger.info(
            "CLI override: model.tabular_residual.enabled = %s "
            "(--tabular-residual %s)",
            forced,
            tabular_residual_choice,
        )

    dataset_yaml = _resolve_dataset_config(args)
    if not dataset_yaml.exists():
        raise SystemExit(f"dataset YAML not found: {dataset_yaml}")

    K = int(args.K) if args.K is not None else int(
        (config.get("outcomes") or {}).get("K", 3)
    )

    # ---- 2. adapter -----------------------------------------------------
    from src.data.adapter import YamlAdapter

    adapter = YamlAdapter(dataset_yaml)
    logger.info("adapter=%r (yaml=%s)", adapter.name, dataset_yaml)

    # Guard on fixture presence: the adapter's events CSV must exist before
    # we bother with any downstream work.
    if not Path(adapter.schema.events_path).exists():
        raise SystemExit(
            f"adapter events_path does not exist: {adapter.schema.events_path} "
            f"(adapter YAML: {dataset_yaml})."
        )
    if not Path(adapter.schema.persons_path).exists():
        raise SystemExit(
            f"adapter persons_path does not exist: {adapter.schema.persons_path} "
            f"(adapter YAML: {dataset_yaml})."
        )

    # ---- 3. load + clean + survey-join + state features ---------------
    logger.info("stage: load")
    events_raw = adapter.load_events()
    persons_raw = adapter.load_persons()

    from src.data.clean import clean_events
    from src.data.state_features import (
        attach_train_brand_map,
        attach_train_popularity,
        compute_state_features,
    )
    from src.data.split import cold_start_split, temporal_split
    from src.data.survey_join import join_survey

    logger.info("stage: clean_events")
    events = clean_events(events_raw, adapter.schema)
    logger.info("stage: join_survey")
    events = join_survey(events, persons_raw, adapter.schema)
    logger.info("stage: compute_state_features")
    events = compute_state_features(events)

    # ---- 4. min-events-per-customer filter (BEFORE split) --------------
    n_events_before_filter = len(events)
    counts = events.groupby("customer_id").size()
    keep_customers = set(counts[counts >= int(args.min_events_per_customer)].index)
    n_dropped_customers = int(
        events["customer_id"].nunique() - len(keep_customers)
    )
    events = events[events["customer_id"].isin(keep_customers)].reset_index(
        drop=True
    )
    logger.info(
        "min_events filter: dropped %d customers (<%d events); %d events remain "
        "(was %d).",
        n_dropped_customers,
        int(args.min_events_per_customer),
        len(events),
        n_events_before_filter,
    )

    # ---- 5. split + popularity ----------------------------------------
    split_mode = str(getattr(args, "split_mode", "temporal"))
    if split_mode == "cold_start":
        split_seed_raw = getattr(args, "split_seed", None)
        effective_split_seed = int(
            split_seed_raw if split_seed_raw is not None else args.seed
        )
        logger.info(
            "stage: cold_start_split (between-customer; split_seed=%d)",
            effective_split_seed,
        )
        events = cold_start_split(
            events, schema=adapter.schema, seed=effective_split_seed
        )
    else:
        logger.info("stage: temporal_split (per-customer within-customer)")
        events = temporal_split(events, adapter.schema)
    logger.info("stage: attach_train_popularity")
    events = attach_train_popularity(events)
    logger.info("stage: attach_train_brand_map")
    events = attach_train_brand_map(events)

    train_events = events[events["split"] == "train"].copy()

    # ---- 6. Appendix-C subsample --------------------------------------
    subsample_method = str(
        (config.get("subsample") or {}).get("method", "leverage")
    ).lower()
    logger.info(
        "stage: subsample (n_customers=%d, method=%s)",
        int(args.n_customers),
        subsample_method,
    )
    from src.train.loop import try_import_subsample_weights

    selected_ids, event_weights = try_import_subsample_weights(
        train_events,
        n_customers=int(args.n_customers),
        seed=int(args.seed),
        method=subsample_method,
    )
    if selected_ids is None:
        logger.error(
            "Subsample required but fell back to omega=1; see WARNING above. "
            "Check that state_features ran correctly."
        )
        return 1

    selected_id_set = set(selected_ids.tolist() if hasattr(selected_ids, "tolist")
                          else list(selected_ids))
    # Subsample is a TRAIN-customer selection (Appendix-C leverage
    # scores are computed on train events only). Under cold-start,
    # val/test customers are never in ``selected_ids``; a blanket
    # ``.isin`` filter would drop every val/test event.
    #
    # Generalized rule: keep an event iff its customer is in
    # ``selected_id_set`` OR its customer has no train rows at all.
    # Under temporal split every customer has train rows, so the second
    # clause is empty and the rule collapses to the pre-fix
    # ``.isin(selected_id_set)`` behavior exactly. Under cold-start,
    # val/test customers are precisely "customers with no train rows",
    # so all their events survive.
    train_customer_set = set(
        events.loc[events["split"] == "train", "customer_id"].unique().tolist()
    )
    val_test_only_customers = (
        set(events["customer_id"].unique().tolist()) - train_customer_set
    )
    keep_customers = selected_id_set | val_test_only_customers
    events_subset = events[
        events["customer_id"].isin(keep_customers)
    ].reset_index(drop=True)
    logger.info(
        "subsample: selected %d customers, %d events total.",
        len(selected_id_set),
        len(events_subset),
    )

    # Recompute per-split subsets on the filtered frame.
    train_events_subset = events_subset[events_subset["split"] == "train"].copy()
    val_events_subset = events_subset[events_subset["split"] == "val"].copy()

    # ---- 7. subsample_diagnostics.json --------------------------------
    subsample_diag = {
        "n_customers": int(len(selected_id_set)),
        "customer_ids": [str(c) for c in sorted(selected_id_set)],
        "omega_stats": _omega_stats(event_weights),
    }
    (out_dir / "subsample_diagnostics.json").write_text(
        json.dumps(subsample_diag, indent=2)
    )

    # ---- 8. popularity-percentile wire-up + orphan filter -------------
    from src.data.alt_rendering import register_on_adapter

    register_on_adapter(adapter, train_events_subset)

    # Orphan filter: customers present in BOTH events_subset.train AND
    # persons_raw (survey-joined). The "surveyed-customer" set is everyone
    # whose raw persons-id column maps to a real row (non-null).
    persons_id_col = adapter.schema.persons_id_column
    if persons_id_col not in persons_raw.columns:
        raise SystemExit(
            f"persons_raw missing id column {persons_id_col!r} "
            f"(schema: {dataset_yaml})."
        )
    surveyed_customers = set(
        persons_raw[persons_id_col].dropna().astype(str).unique().tolist()
    )
    # Use ALL customers in the subsampled frame (train + val + test), not
    # just the train-split customers. Under cold-start, val/test
    # customers have no train events by construction, and filtering on
    # ``train_customers`` alone would drop every val/test customer's
    # events from the pipeline. The downstream ``translate_z_d`` call is
    # already fit-on-train-only (via ``train_events_subset`` below), so
    # widening here does not introduce leakage.
    all_customers = set(
        events_subset["customer_id"].astype(str).unique().tolist()
    )
    train_customers = set(
        train_events_subset["customer_id"].astype(str).unique().tolist()
    )
    joint_customers = all_customers & surveyed_customers
    if not joint_customers:
        raise SystemExit(
            "orphan filter: no customers in both events and survey "
            "(events=%d, survey=%d)." % (
                len(all_customers), len(surveyed_customers),
            )
        )
    logger.info(
        "orphan filter: %d customers in events∩survey "
        "(events=%d, train=%d, survey=%d).",
        len(joint_customers),
        len(all_customers),
        len(train_customers),
        len(surveyed_customers),
    )
    events_subset = events_subset[
        events_subset["customer_id"].astype(str).isin(joint_customers)
    ].reset_index(drop=True)
    train_events_subset = events_subset[events_subset["split"] == "train"].copy()
    val_events_subset = events_subset[events_subset["split"] == "val"].copy()

    # Align the event_weights to filtered train events.
    # event_weights from try_import_subsample_weights is aligned with
    # train_events_subset BEFORE the orphan filter. Rebuild as per-customer
    # weights; every train event of a customer gets the same weight.
    customer_weight = {}
    pre_orphan_train = events[events["split"] == "train"].reset_index(drop=True)
    pre_orphan_train = pre_orphan_train[
        pre_orphan_train["customer_id"].isin(selected_id_set)
    ].reset_index(drop=True)
    ew_arr = np.asarray(event_weights, dtype=np.float64)
    if len(pre_orphan_train) == len(ew_arr):
        for cid, w in zip(
            pre_orphan_train["customer_id"].astype(str).tolist(),
            ew_arr.tolist(),
        ):
            customer_weight.setdefault(cid, float(w))
    # Per-event weights aligned with train_events_subset:
    train_event_weights_aligned = np.asarray(
        [
            float(customer_weight.get(str(cid), 1.0))
            for cid in train_events_subset["customer_id"].tolist()
        ],
        dtype=np.float32,
    )

    # ---- 9. translate z_d on survived customers, fit stats on TRAIN ---
    logger.info("stage: translate_z_d (fit-on-train)")
    # persons_raw slice for surviving customers — kept to only those rows
    # that actually have survey data (joint_customers).
    persons_raw_keep = persons_raw[
        persons_raw[persons_id_col].astype(str).isin(joint_customers)
    ].copy()

    persons_canonical = adapter.translate_z_d(
        persons_raw_keep,
        training_events=train_events_subset,
    )

    # Wave-11 fix: normalize purchase_frequency from total count to
    # events-per-week using the actual training-window duration. Without
    # this, context_string._phrase_purchase_frequency misreads a
    # 9-events-in-5-years customer as "almost daily". The transform is
    # applied to BOTH z_d inputs (via the canonical column) and c_d
    # rendering (via the same column); standardization stats fit on
    # the rate are numerically equivalent to stats fit on the count,
    # just on a different natural scale.
    if len(train_events_subset) > 0:
        _train_dates = pd.to_datetime(train_events_subset["order_date"])
        _window_days = max((_train_dates.max() - _train_dates.min()).days, 1)
        _window_weeks = max(_window_days / 7.0, 1.0)
        if "purchase_frequency" in persons_canonical.columns:
            persons_canonical = persons_canonical.copy()
            persons_canonical["purchase_frequency"] = (
                persons_canonical["purchase_frequency"].astype(float)
                / _window_weeks
            )
            logger.info(
                "normalized purchase_frequency count -> events/week over a "
                "%d-day (%.1f-week) training window.",
                _window_days, _window_weeks,
            )

    # Post-translate orphan filter: translate_z_d may drop rows via
    # drop_on_unknown (e.g. "Prefer not to say" on income/education).
    # Events referencing dropped customers would KeyError at z_d lookup
    # inside build_choice_sets. Re-filter events_subset and
    # train_events_subset to only the customers that survived translate.
    surviving = set(persons_canonical["customer_id"])
    n_before = len(events_subset)
    events_subset = events_subset[
        events_subset["customer_id"].isin(surviving)
    ].copy()
    train_events_subset = train_events_subset[
        train_events_subset["customer_id"].isin(surviving)
    ].copy()
    n_dropped = n_before - len(events_subset)
    if n_dropped > 0:
        logger.info(
            "translate_z_d dropped customers via drop_on_unknown; "
            "cascaded filter removed %d events (of %d) and left %d "
            "customers for choice-set construction.",
            n_dropped, n_before, len(surviving),
        )
        # Rebuild the event-weight alignment against the filtered train
        # set so assemble_batch's omega-shape invariant holds.
        train_event_weights_aligned = np.asarray(
            [
                float(customer_weight.get(str(cid), 1.0))
                for cid in train_events_subset["customer_id"].tolist()
            ],
            dtype=np.float32,
        )

    # --- Wave-10 caveat: val z_d fit on train -------------------------
    # We run build_choice_sets on the WHOLE events_subset (train+val+test
    # rows for the surviving customers). Because persons_canonical lists
    # only customers with train events, the fit-on-train assertion passes,
    # and every record — even val-split rows — gets z_d from the TRAIN fit.
    # That matches the "minimal-valid" option in the Wave-10 brief. The
    # stats are train-derived; the val/test transform is identical to the
    # train transform, which is exactly what the model sees at inference.
    # Deferred to Wave 11/12: a fit-on-train, transform-on-all pattern that
    # supports val/test customers not present in train.
    from src.data.choice_sets import build_choice_sets
    from src.data.context_string import extract_extra_fields_from_row

    # Wave-11 c_d enrichment: build a {customer_id -> extras} map from the
    # dataset YAML's optional ``persons.c_d_extra_fields`` block and the
    # RAW (pre-translate) persons rows. Unknown / NaN values are dropped
    # inside the helper, so customers who refused gender / have no
    # recent life-change / etc. simply get those clauses omitted.
    dataset_yaml_dict = _load_yaml(dataset_yaml)
    extras_block = (
        dataset_yaml_dict.get("dataset", {})
        .get("persons", {})
        .get("c_d_extra_fields", {})
    ) or {}
    customer_to_extras: dict[str, dict] = {}
    if extras_block:
        survivors = set(persons_canonical["customer_id"].astype(str))
        for _, raw_row in persons_raw.iterrows():
            cid = str(raw_row.get(persons_id_col, ""))
            if cid not in survivors:
                continue
            customer_to_extras[cid] = extract_extra_fields_from_row(
                raw_row.to_dict(), extras_block
            )
        logger.info(
            "c_d enrichment: extras_block loaded (%d fields); populated "
            "extras for %d of %d survivors.",
            len(extras_block),
            sum(1 for v in customer_to_extras.values() if v),
            len(survivors),
        )

    # Customer-aggregate c_d enrichment: opt-in via YAML
    # ``data.enrich_customer_context`` (default false). When true, per-
    # customer summaries (top brand, top categories, average price,
    # repeat rate) are merged into ``customer_to_extras`` and rendered
    # into c_d by the new clauses in build_context_string. Keeps existing
    # Wave-11 fields (gender / life_event / amazon_frequency) — agg keys
    # use distinct names so they cannot collide.
    data_cfg = (config.get("data") or {})
    enrich_ctx = bool(data_cfg.get("enrich_customer_context", False))
    if enrich_ctx:
        from src.data.context_string import compute_customer_aggregates
        aggs = compute_customer_aggregates(events_subset, train_only=True)
        n_enriched = 0
        for cid, agg in aggs.items():
            if not agg:
                continue
            existing = customer_to_extras.get(str(cid), {}) or {}
            existing.update(agg)
            customer_to_extras[str(cid)] = existing
            n_enriched += 1
        logger.info(
            "c_d enrichment (customer aggregates): populated extras for "
            "%d / %d customers (top_brand / top_categories / avg_price / "
            "repeat_rate)",
            n_enriched, len(aggs),
        )

    # Mobility-specific c_d enrichment: typical trip length, weekend
    # share, daypart preference. Computed train-only so val/test
    # transform under the same per-customer summary the model trained on.
    if str(args.adapter) == "mobility_boston":
        from src.data.context_string import compute_mobility_aggregates
        m_aggs = compute_mobility_aggregates(events_subset, train_only=True)
        n_m_enriched = 0
        for cid, agg in m_aggs.items():
            if not agg:
                continue
            existing = customer_to_extras.get(str(cid), {}) or {}
            existing.update(agg)
            customer_to_extras[str(cid)] = existing
            n_m_enriched += 1
        logger.info(
            "c_d enrichment (mobility aggregates): populated extras for "
            "%d / %d customers (typical_distance_km / weekend_share / "
            "daypart_preference)",
            n_m_enriched, len(m_aggs),
        )

    # Hard-negative sampling rate: opt-in via YAML ``data.hard_negative_rate``
    # (default 0.0 — preserves the legacy half-cat / half-random sampler).
    # When > 0, that fraction of n_negatives is drawn from same-category +
    # price-band-similar ASINs, forcing PO-LEU to discriminate on subtle
    # signals instead of obvious distractors.
    hard_neg_rate = float(data_cfg.get("hard_negative_rate", 0.0) or 0.0)
    hard_neg_band = float(data_cfg.get("hard_negative_price_band", 0.5) or 0.5)
    logger.info("stage: build_choice_sets (train+val+test rows together)")
    if hard_neg_rate > 0.0:
        logger.info(
            "hard-negative sampling: rate=%.2f, price_band=%.2f",
            hard_neg_rate, hard_neg_band,
        )
    # Per-event time-of-day phrase in c_d. Off by default (preserves
    # Amazon snapshot tests bit-identical); flip on either via the YAML
    # (``data.add_event_time_to_c_d: true``) or the CLI flag
    # ``--add-event-time-to-c-d``. CLI wins.
    add_event_time = bool(
        getattr(args, "add_event_time_to_c_d", False)
        or data_cfg.get("add_event_time_to_c_d", False)
    )
    if add_event_time:
        logger.info("c_d enrichment: per-event time-of-day phrase enabled")
    add_event_origin = bool(
        getattr(args, "add_event_origin_to_c_d", False)
        or data_cfg.get("add_event_origin_to_c_d", False)
    )
    if add_event_origin:
        logger.info("c_d enrichment: per-event origin context enabled")

    # Leak fix (audit Finding 1, mobility-only): symmetric per-(event,
    # alt) price = haversine(event.from_cbg, alt.typical_to_cbg). Train-
    # fit; same formula for chosen and negatives. For Amazon and any
    # adapter without from_cbg / to_cbg / a centroid CSV, the override
    # closure is None and price stays exactly as before.
    overrides_fn = None
    if str(args.adapter) == "mobility_boston":
        from src.data.mobility_geodistance import (
            make_per_event_alt_overrides_fn,
        )
        centroid_path = (
            REPO_ROOT / "mobility_trajectory_boston"
            / "Basic_Geographic_Statistics_CBG_Boston.csv"
        )
        overrides_fn = make_per_event_alt_overrides_fn(
            events_subset, centroid_path,
        )
        logger.info(
            "leak fix: symmetric per-(event, alt) price closure wired "
            "(centroid_path=%s)", centroid_path,
        )

    records_all = build_choice_sets(
        events_subset,
        persons_canonical,
        adapter,
        seed=int(args.seed),
        n_resamples=int(adapter.schema.n_resamples),
        n_negatives=int(adapter.schema.choice_set_size) - 1,
        customer_to_extras=customer_to_extras or None,
        drop_pool_starved=bool(getattr(args, "drop_pool_starved_events", False)),
        hard_negative_rate=hard_neg_rate,
        hard_negative_price_band=hard_neg_band,
        add_event_time_to_c_d=add_event_time,
        add_event_origin_to_c_d=add_event_origin,
        per_event_alt_overrides_fn=overrides_fn,
    )

    # Split records by the split label embedded in each record. Needed
    # once drop_pool_starved drops rows from records_all — can no longer
    # zip against events_subset since lengths may diverge.
    records_train = [r for r in records_all if r.get("split") == "train"]
    records_val = [r for r in records_all if r.get("split") == "val"]
    records_test = [r for r in records_all if r.get("split") == "test"]
    logger.info(
        "records: train=%d val=%d test=%d.",
        len(records_train),
        len(records_val),
        len(records_test),
    )

    # Persist the record set so downstream drivers
    # (scripts/run_baselines.py --records-from) can score the same events
    # PO-LEU saw. Without this, the baselines re-derive records
    # independently (slightly different filter chain) and end up on a
    # different slice, breaking apples-to-apples comparison on the
    # leaderboard.
    import pickle
    records_pkl = out_dir / "records.pkl"
    with records_pkl.open("wb") as _fh:
        pickle.dump(
            {"train": records_train, "val": records_val, "test": records_test},
            _fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info(
        "wrote %s (%d+%d+%d records)",
        records_pkl, len(records_train), len(records_val), len(records_test),
    )

    # Re-align the per-event importance weights to the SURVIVING train
    # records. ``train_event_weights_aligned`` was built earlier against
    # ``train_events_subset`` (pre-build_choice_sets) and will therefore
    # carry one entry per ORIGINAL train event; with
    # ``--drop-pool-starved-events`` a few of those events are absent from
    # ``records_train`` and the ``assemble_batch`` omega-shape invariant
    # would fail. Rebuild from ``customer_weight`` using each surviving
    # record's customer_id so length and order match records_train exactly.
    if len(records_train) != len(train_event_weights_aligned):
        train_event_weights_aligned = np.asarray(
            [
                float(customer_weight.get(str(r["customer_id"]), 1.0))
                for r in records_train
            ],
            dtype=np.float32,
        )

    if len(records_train) == 0:
        raise SystemExit(
            "No training records after all filters; cannot train. "
            "Check --n-customers / --min-events-per-customer thresholds."
        )
    if len(records_val) == 0:
        logger.warning(
            "No validation records after filters; evaluation will reuse "
            "training batch for metrics."
        )

    # ---- 10. outcomes / embeddings cache + batch assembly -------------
    from src.data.batching import assemble_batch, iter_to_torch_batches

    llm_client, encoder = _build_llm_and_encoder(args, config)

    # Caches live at repo-root defaults unless overridden in config or
    # via the ``OUTCOMES_CACHE_PATH`` / ``EMBEDDINGS_CACHE_PATH`` env
    # vars. Env vars take highest precedence — they let parallel
    # tmux sessions point each seed at its own SQLite DB so the WAL-mode
    # write contention you'd otherwise see across processes goes away
    # entirely. Different seeds NEVER share cache entries (seed is
    # folded into the composite key) so per-seed DBs don't change
    # any LLM cost; the only effect is removing lock contention.
    paths_cfg = config.get("paths") or {}
    cache_cfg = (config.get("outcomes") or {}).get("cache") or {}
    env_outcomes = os.environ.get("OUTCOMES_CACHE_PATH", "").strip()
    env_embeddings = os.environ.get("EMBEDDINGS_CACHE_PATH", "").strip()
    outcomes_cache_path = Path(
        env_outcomes or
        cache_cfg.get("outcomes_path",
                      (paths_cfg.get("outcomes_cache", "outcomes_cache/")
                       + "outcomes.sqlite"))
    )
    embeddings_cache_path = Path(
        env_embeddings or
        cache_cfg.get("embeddings_path",
                      (paths_cfg.get("embeddings_cache", "embeddings_cache/")
                       + "embeddings.sqlite"))
    )
    # Make absolute against the repo root if relative.
    if not outcomes_cache_path.is_absolute():
        outcomes_cache_path = REPO_ROOT / outcomes_cache_path
    if not embeddings_cache_path.is_absolute():
        embeddings_cache_path = REPO_ROOT / embeddings_cache_path
    outcomes_cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if env_outcomes or env_embeddings:
        logger.info(
            "cache paths overridden via env: outcomes=%s, embeddings=%s",
            outcomes_cache_path, embeddings_cache_path,
        )

    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter as _default_div_filter

    outcomes_cache = OutcomesCache(outcomes_cache_path)
    embeddings_cache = EmbeddingsCache(embeddings_cache_path)

    # Strategy B tabular residual: read the YAML gate. When ``enabled`` is
    # true we materialise the optional ``x_tab`` tensor on every assembled
    # batch and pass the same feature-name list to the model below. When
    # the block is missing, off, or set to an empty list, ``tab_names`` is
    # ``None`` and assemble_batch silently leaves ``batch.x_tab`` as None
    # — bit-identical to pre-residual runs.
    tab_cfg = (config.get("model") or {}).get("tabular_residual") or {}
    tab_enabled = bool(tab_cfg.get("enabled", False))
    tab_features_yaml = list(tab_cfg.get("features") or [])
    tab_names: list[str] | None = (
        tab_features_yaml if (tab_enabled and tab_features_yaml) else None
    )
    if tab_enabled and not tab_features_yaml:
        logger.warning(
            "tabular_residual.enabled=true but features list is empty; "
            "running without the residual."
        )
    if tab_names is not None:
        logger.info(
            "Strategy B tabular residual: enabled (features=%s)",
            tab_names,
        )

    # Curriculum-refinement cascade: when set, assemble_batch tries each
    # version's cache before falling through to the LAST version's
    # generate_outcomes (the only version permitted to LLM-call on miss).
    # ``getattr`` keeps in-process callers (smoke tests building a
    # Namespace by hand) working without listing every CLI flag.
    cascade_arg = getattr(args, "prompt_version_cascade", None)
    prompt_version_cascade = (
        tuple(cascade_arg) if cascade_arg else None
    )
    if prompt_version_cascade:
        logger.info(
            "prompt_version cascade active: %s (last entry is the LLM-fallback)",
            list(prompt_version_cascade),
        )

    logger.info("stage: assemble_batch (train)")
    batch_train = assemble_batch(
        records_train,
        adapter=adapter,
        llm_client=llm_client,
        encoder=encoder,
        outcomes_cache=outcomes_cache,
        embeddings_cache=embeddings_cache,
        K=K,
        seed=int(args.seed),
        prompt_version_cascade=prompt_version_cascade,
        diversity_filter=_default_div_filter,
        omega=train_event_weights_aligned,
        tabular_feature_names=tab_names,
        max_concurrent_llm_calls=int(
            os.environ.get("MAX_CONCURRENT_LLM_CALLS", "8")
        ),
    )

    # Val batch: reuse train batch when val is empty so downstream forward
    # pass still has a usable tensor. (Doesn't affect train metrics.)
    if len(records_val) == 0:
        logger.warning("val records empty; reusing train batch for evaluation.")
        batch_val = batch_train
    else:
        logger.info("stage: assemble_batch (val)")
        batch_val = assemble_batch(
            records_val,
            adapter=adapter,
            llm_client=llm_client,
            encoder=encoder,
            outcomes_cache=outcomes_cache,
            embeddings_cache=embeddings_cache,
            K=K,
            seed=int(args.seed),
            prompt_version_cascade=prompt_version_cascade,
            diversity_filter=_default_div_filter,
            omega=None,
            tabular_feature_names=tab_names,
            max_concurrent_llm_calls=int(
                os.environ.get("MAX_CONCURRENT_LLM_CALLS", "8")
            ),
        )

    # Test batch: §9.1 held-out split, un-weighted. Reuse val batch when
    # empty so we can still emit metrics_test.json (with a WARNING) -- this
    # matches the 1-customer edge case where temporal_split leaves exactly
    # one test event.
    if len(records_test) == 0:
        logger.warning(
            "test records empty after cascaded filters; reusing val batch "
            "for metrics_test.json. Test metrics will equal val metrics."
        )
        batch_test = batch_val
    else:
        logger.info("stage: assemble_batch (test)")
        batch_test = assemble_batch(
            records_test,
            adapter=adapter,
            llm_client=llm_client,
            encoder=encoder,
            outcomes_cache=outcomes_cache,
            embeddings_cache=embeddings_cache,
            K=K,
            seed=int(args.seed),
            prompt_version_cascade=prompt_version_cascade,
            diversity_filter=_default_div_filter,
            omega=None,
            tabular_feature_names=tab_names,
            max_concurrent_llm_calls=int(
                os.environ.get("MAX_CONCURRENT_LLM_CALLS", "8")
            ),
        )
        if len(batch_test) < 10:
            logger.warning(
                "n_test=%d < 10; test metrics have large variance "
                "(single-customer / tiny-fixture edge case).",
                len(batch_test),
            )

    # ---- 11. POLEU model + training setup -----------------------------
    from src.model.po_leu import POLEU
    from src.train.loop import TrainConfig, fit
    from src.train.regularizers import RegularizerConfig

    # p is whatever the train batch carries — do NOT hardcode.
    p = int(batch_train.z_d.shape[1])
    J = int(batch_train.E.shape[1])
    d_e = int(batch_train.E.shape[3])

    model_cfg = config.get("model") or {}
    attr_cfg = model_cfg.get("attribute_heads") or {}
    wnet_cfg = model_cfg.get("weight_net") or {}
    snet_cfg = model_cfg.get("salience_net") or {}

    # Group-2: size SalienceNet's category embedding from the assembled
    # batch's category vocabulary. Falls back to 1 (legacy single bucket)
    # when records carry no "category_vocab" key.
    _vocab = getattr(batch_train, "category_vocab", ())
    n_categories_runtime = max(1, len(_vocab))

    poleu_kwargs: dict[str, Any] = dict(
        M=int(model_cfg.get("M", 5)),
        K=K,
        J=J,
        d_e=d_e,
        p=p,
        attribute_hidden=int(attr_cfg.get("hidden", 128)),
        weight_hidden=int(wnet_cfg.get("hidden", 32)),
        salience_hidden=int(snet_cfg.get("hidden", 64)),
        weight_normalization=str(wnet_cfg.get("normalization", "softmax")),
        uniform_salience=bool(model_cfg.get("uniform_salience", False)),
        temperature=float(model_cfg.get("temperature", 1.0)),
        n_categories=n_categories_runtime,
        d_cat=int(snet_cfg.get("d_cat", 8)),
    )
    # Strategy B: only pass the residual kwargs when actively enabled +
    # x_tab is materialised; otherwise stay on POLEU's default-False
    # constructor branch so model.parameters() does not gain a stray
    # beta_tab the training step will never see gradients for.
    if tab_names is not None and batch_train.x_tab is not None:
        poleu_kwargs["tabular_residual_enabled"] = True
        poleu_kwargs["tabular_features"] = tuple(tab_names)
    model = POLEU(**poleu_kwargs)

    # Fit train-set per-feature mean/std and install them as buffers on
    # the model. Standardising before β receives any gradient makes
    # |β_f| comparable across features whose natural scales differ by
    # orders of magnitude (price in dollars vs. price_rank in [0, 1]).
    if (
        tab_names is not None
        and batch_train.x_tab is not None
        and getattr(model, "tabular_residual_enabled", False)
    ):
        flat = batch_train.x_tab.reshape(-1, batch_train.x_tab.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        model.set_tabular_feature_stats(mean, std)
        logger.info(
            "tabular_residual stats fit on train: mean=%s std=%s",
            mean.tolist(),
            std.tolist(),
        )

    # Train config: start from default YAML, then override from CLI.
    train_cfg = TrainConfig.from_default()
    train_cfg.batch_size = int(args.batch_size)
    train_cfg.max_epochs = int(args.n_epochs)
    reg_cfg = RegularizerConfig.from_default()

    # ---- dataset-dependent p + regularizers-active diagnostics -----------
    # Spec §2.1 canonical p=26. Amazon-style datasets with collapsed
    # vocabularies (e.g. a single city_size label under an empty
    # external_lookup) ship a smaller effective p. Surface both in
    # artifacts and log once.
    canonical_p = int((config.get("model") or {}).get("p", 26))
    effective_p = int(batch_train.z_d.shape[1])
    p_is_dataset_dependent = bool(effective_p != canonical_p)
    p_reduction_reason = (
        _p_reduction_reason(adapter) if p_is_dataset_dependent else "none"
    )
    if p_is_dataset_dependent:
        logger.info(
            "effective_p=%d (canonical=%d, dataset-dependent; reason: %s)",
            effective_p, canonical_p, p_reduction_reason,
        )
    else:
        logger.info("effective_p=%d (matches canonical)", effective_p)

    # §9.2 regularizer activation map. ``monotonicity`` requires the
    # config gate AND a real ``prices`` tensor on the train batch -- the
    # AssembledBatch dataclass declares ``prices`` non-optional, but we
    # guard via ``getattr`` for robustness.
    batch_prices = getattr(batch_train, "prices", None)
    regularizers_active = {
        "weight_l2": bool(reg_cfg.weight_l2 > 0),
        "salience_entropy": bool(reg_cfg.salience_entropy > 0),
        "monotonicity": bool(
            reg_cfg.monotonicity_enabled
            and reg_cfg.monotonicity > 0
            and batch_prices is not None
        ),
        "diversity": bool(reg_cfg.diversity > 0),
    }
    active_list = [k for k, v in regularizers_active.items() if v]
    logger.info("regularizers active: %s", ", ".join(active_list) or "none")

    g_train = torch.Generator().manual_seed(int(args.seed))

    def train_batches_fn():
        return iter_to_torch_batches(
            batch_train,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            generator=g_train,
        )

    def val_batches_fn():
        return iter_to_torch_batches(
            batch_val,
            batch_size=train_cfg.batch_size,
            shuffle=False,
        )

    import math as _math
    n_batches_per_epoch = max(1, _math.ceil(len(batch_train) / train_cfg.batch_size))
    total_steps = n_batches_per_epoch * train_cfg.max_epochs

    logger.info(
        "stage: fit (p=%d, J=%d, d_e=%d, n_params=%d, n_train=%d, steps=%d)",
        p, J, d_e, model.num_params(), len(batch_train), total_steps,
    )
    state = fit(
        model,
        train_batches_fn,
        val_batches_fn,
        train_cfg=train_cfg,
        reg_cfg=reg_cfg,
        total_steps=total_steps,
        seed=int(args.seed),
    )

    # ---- 12. evaluate + metrics ---------------------------------------
    from src.eval.metrics import compute_all

    logger.info("stage: evaluate (val)")
    model.eval()
    with torch.no_grad():
        if batch_val.x_tab is not None and getattr(
            model, "tabular_residual_enabled", False
        ):
            logits_val, interm_val = model(
                batch_val.z_d, batch_val.E, batch_val.x_tab
            )
        else:
            logits_val, interm_val = model(batch_val.z_d, batch_val.E)
    metrics = compute_all(
        logits_val,
        batch_val.c_star,
        n_params=model.num_params(),
        n_train=len(batch_train),
    )

    # Diagnostics merged into both metrics.json and smoke_summary.json.
    diagnostics_fields: dict[str, Any] = {
        "effective_p": effective_p,
        "canonical_p": canonical_p,
        "p_is_dataset_dependent": p_is_dataset_dependent,
        "p_reduction_reason": p_reduction_reason,
        "regularizers_active": regularizers_active,
    }

    metrics_payload = dict(metrics.to_dict())
    metrics_payload.update(diagnostics_fields)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    logger.info("stage: evaluate (test)")
    with torch.no_grad():
        if batch_test.x_tab is not None and getattr(
            model, "tabular_residual_enabled", False
        ):
            logits_test, _interm_test = model(
                batch_test.z_d, batch_test.E, batch_test.x_tab
            )
        else:
            logits_test, _interm_test = model(batch_test.z_d, batch_test.E)
    test_metrics = compute_all(
        logits_test,
        batch_test.c_star,
        n_params=model.num_params(),
        n_train=len(batch_train),
    )
    test_metrics_payload = dict(test_metrics.to_dict())
    test_metrics_payload.update(diagnostics_fields)
    test_metrics_payload["n_test_records"] = int(len(batch_test))
    (out_dir / "metrics_test.json").write_text(
        json.dumps(test_metrics_payload, indent=2)
    )

    # ---- 12b. per-event NLL sidecars (val + test) ---------------------
    # The curriculum-refinement loop (scripts/identify_failure_events.py)
    # consumes ``val_per_event.json`` to pick the worst val events to
    # regenerate outcomes for. The test sidecar is for diagnostics only —
    # never use it to choose what to refine (test-set leakage). Schema
    # mirrors src/baselines/evaluate.py:130-139 so downstream tooling
    # can join PO-LEU and baseline rows by event_idx.
    def _write_per_event_sidecar(
        path: Path,
        logits_t: torch.Tensor,
        records: list[dict],
        c_star_t: torch.Tensor,
    ) -> None:
        logits_np = logits_t.detach().cpu().numpy().astype("float64")
        c_star_np = c_star_t.detach().cpu().numpy().astype("int64")
        # Stable log-softmax: shift by row max before exp.
        row_max = logits_np.max(axis=1, keepdims=True)
        shifted = logits_np - row_max
        log_z = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_z
        N = logits_np.shape[0]
        rows = []
        for i in range(N):
            ci = int(c_star_np[i])
            rec = records[i] if i < len(records) else {}
            rows.append({
                "event_idx": i,
                "customer_id": str(rec.get("customer_id", "")),
                "asin_chosen": str(
                    rec.get("choice_asins", [""] * (ci + 1))[ci]
                    if rec.get("choice_asins") else ""
                ),
                "c_star": ci,
                "p_chosen": float(np.exp(log_probs[i, ci])),
                "nll": float(-log_probs[i, ci]),
                "top1_correct": bool(int(np.argmax(logits_np[i])) == ci),
            })
        path.write_text(json.dumps(
            {"per_event": rows, "n_events": N},
            indent=2,
        ))

    _write_per_event_sidecar(
        out_dir / "val_per_event.json",
        logits_val,
        records_val if records_val else records_train,
        batch_val.c_star,
    )
    _write_per_event_sidecar(
        out_dir / "test_per_event.json",
        logits_test,
        records_test if records_test else (records_val or records_train),
        batch_test.c_star,
    )
    logger.info(
        "wrote per-event sidecars: %s, %s",
        out_dir / "val_per_event.json",
        out_dir / "test_per_event.json",
    )

    # ---- 12c. emit test_logits.npz for the leaderboard ----------------
    # Schema matches scripts/run_interpretability_tax.py and is consumed
    # by scripts/run_baselines.py via --external-logits / --poleu-logits.
    # Lets run_full_evaluation.sh fold this PO-LEU run (residual on or off)
    # into the leaderboard as a named row.
    logits_path = out_dir / "test_logits.npz"
    np.savez(
        logits_path,
        logits=logits_test.detach().cpu().numpy().astype("float32"),
        c_star=batch_test.c_star.detach().cpu().numpy().astype("int64"),
        n_params=np.array(int(model.num_params()), dtype=np.int64),
    )
    logger.info("wrote %s", logits_path)

    # ---- 13. §12 reports (interpretability) ---------------------------
    from src.eval.interpret import run_all_reports

    # Pick attribute-head labels matching the prompt version actually used
    # for outcome generation. Default = §5.2 purchase-anchored axes.
    head_names_override: list[str] | None = None
    if prompt_version_cascade:
        last_pv = str(prompt_version_cascade[-1])
        if last_pv.startswith("v4_mobility_anchored"):
            from src.outcomes.prompts import MOBILITY_ANCHORED_AXES
            head_names_override = list(MOBILITY_ANCHORED_AXES)
        elif last_pv.startswith("v3_anchored"):
            from src.outcomes.prompts import ANCHORED_AXES
            head_names_override = list(ANCHORED_AXES)

    logger.info(
        "stage: run_all_reports (head_names=%s)",
        head_names_override or "default",
    )
    run_all_reports(
        model,
        batch_val.z_d,
        batch_val.E,
        batch_val.c_star,
        batch_val.outcomes_nested,
        out_dir=out_dir,
        event_idx=0,
        head_names=head_names_override,
    )

    # ---- 14. smoke_summary.json ---------------------------------------
    reports_written = sorted(
        f.name for f in out_dir.iterdir() if f.suffix == ".json"
    )
    smoke_summary = {
        "config": {
            "adapter": args.adapter,
            "dataset_yaml": str(dataset_yaml),
            "seed": int(args.seed),
            "n_customers_requested": int(args.n_customers),
            "n_customers_actual": int(len(selected_id_set)),
            "min_events_per_customer": int(args.min_events_per_customer),
            "K": K,
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "p": p,
            "J": J,
            "d_e": d_e,
            "n_train_records": int(len(batch_train)),
            "n_val_records": int(len(batch_val)),
            "n_test_records": int(len(batch_test)),
            "llm_mode": "real",
        },
        "train_state": {
            "epoch": int(state.epoch),
            "step": int(state.step),
            "train_loss": float(state.train_loss),
            "val_nll": float(state.val_nll) if state.val_nll is not None else None,
            "best_val_nll": float(state.best_val_nll)
                if state.best_val_nll != float("inf") else None,
            "stopped_early": bool(state.stopped_early),
        },
        "metrics": metrics.to_dict(),
        "metrics_test": test_metrics.to_dict(),
        "effective_p": effective_p,
        "canonical_p": canonical_p,
        "p_is_dataset_dependent": p_is_dataset_dependent,
        "p_reduction_reason": p_reduction_reason,
        "regularizers_active": regularizers_active,
        "reports_written": reports_written,
    }
    (out_dir / "smoke_summary.json").write_text(
        json.dumps(smoke_summary, indent=2)
    )

    # Print a final one-line summary.
    print(
        f"run_dataset OK  adapter={args.adapter}  n_customers={len(selected_id_set)} "
        f"n_train={len(batch_train)}  val_nll={state.val_nll}  top1={metrics.top1:.4f}  "
        f"out_dir={out_dir}"
    )

    # Tidy up SQLite handles.
    try:
        outcomes_cache.close()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
    try:
        embeddings_cache.close()
    except Exception:  # pragma: no cover
        pass

    return 0


# --------------------------------------------------------------------------- #
# Refine orchestration
# --------------------------------------------------------------------------- #


def _run_with_refine(args: argparse.Namespace) -> int:
    """Run round-1 → identify failures → refine → round-2 in one shot.

    Layout:
      ``<output_dir>/round1/``     — round-1 artifacts (records.pkl,
                                     val_per_event.json, test_logits.npz,
                                     metrics.json, …)
      ``<output_dir>/failure_events.json``  — round-1 worst val events
      ``<output_dir>/refined_outcomes.json`` — refine bookkeeping
      ``<output_dir>/<round-2 artifacts>``    — final retrain output

    Cache reuse: round-1 outcome generations populate the configured
    outcomes_cache; refine writes refined entries under
    REFINED_PROMPT_VERSION; round-2 assemble_batch hits the cache for
    every (cust, alt) so no new outcome generations are needed beyond
    the critic+reviser calls.
    """
    final_output_dir = _resolve_output_dir(args)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    round1_dir = final_output_dir / "round1"
    round1_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== --refine: round 1 (initial PO-LEU train) ===")
    round1_args = argparse.Namespace(**vars(args))
    round1_args.refine = False  # avoid recursion
    round1_args.output_dir = round1_dir
    rc1 = _run_pipeline_once(round1_args)
    if rc1 != 0:
        logger.error("--refine: round 1 exit %d; aborting", rc1)
        return rc1

    failure_events_path = final_output_dir / "failure_events.json"
    refined_outcomes_path = final_output_dir / "refined_outcomes.json"

    # Resolve the outcomes-cache path the round-1 run used. Mirrors the
    # env-var-first / config-fallback rule inside _run_pipeline_once.
    config = _load_yaml(args.config)
    paths_cfg = config.get("paths") or {}
    cache_cfg = (config.get("outcomes") or {}).get("cache") or {}
    env_outcomes = os.environ.get("OUTCOMES_CACHE_PATH", "").strip()
    outcomes_cache_path = Path(
        env_outcomes
        or cache_cfg.get(
            "outcomes_path",
            paths_cfg.get("outcomes_cache", "outcomes_cache/")
            + "outcomes.sqlite",
        )
    )
    if not outcomes_cache_path.is_absolute():
        outcomes_cache_path = REPO_ROOT / outcomes_cache_path

    K = int(args.K) if args.K is not None else int(
        (config.get("outcomes") or {}).get("K", 3)
    )

    # Last entry of the cascade is the prompt-version that wrote the v1
    # cache entries we want refine_outcomes to read from (it folds the
    # v1 prompt_version into its lookup key).
    cascade_arg = list(getattr(args, "prompt_version_cascade", None) or [])
    v1_pv = cascade_arg[-1] if cascade_arg else "v1"

    logger.info(
        "=== --refine: identify_failure_events (top-K=%d) ===",
        int(args.refine_top_k),
    )
    import subprocess
    rc_id = subprocess.call(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "identify_failure_events.py"),
            "--run-dir", str(round1_dir),
            "--top-k", str(int(args.refine_top_k)),
            "--output", str(failure_events_path),
        ],
        cwd=str(REPO_ROOT),
    )
    if rc_id != 0:
        logger.error("--refine: identify_failure_events exit %d; aborting", rc_id)
        return rc_id

    logger.info("=== --refine: refine_outcomes (per_outcome=%s) ===",
                args.refine_per_outcome)
    refine_writer = os.environ.get("LLM_PROVIDER", "anthropic")
    if refine_writer not in {"anthropic", "gemini", "openai"}:
        refine_writer = "anthropic"
    rc_rf = subprocess.call(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "refine_outcomes.py"),
            "--failure-events", str(failure_events_path),
            "--outcomes-cache", str(outcomes_cache_path),
            "--K", str(K),
            "--seed", str(int(args.seed)),
            "--v1-prompt-version", str(v1_pv),
            "--writer", refine_writer,
            "--critic", str(args.refine_critic),
            "--accept-threshold", str(int(args.refine_accept_threshold)),
            "--per-outcome", str(args.refine_per_outcome),
            "--output", str(refined_outcomes_path),
        ],
        cwd=str(REPO_ROOT),
    )
    if rc_rf != 0:
        logger.error("--refine: refine_outcomes exit %d; aborting", rc_rf)
        return rc_rf

    logger.info(
        "=== --refine: round 2 (retrain with cascade prepended) ==="
    )
    from src.outcomes.prompts import REFINED_PROMPT_VERSION
    round2_args = argparse.Namespace(**vars(args))
    round2_args.refine = False  # no recursion
    round2_args.output_dir = final_output_dir
    # Prepend the refined version to the cascade so refined entries win.
    round2_args.prompt_version_cascade = (
        [REFINED_PROMPT_VERSION] + cascade_arg
    )
    return _run_pipeline_once(round2_args)


def main(args: argparse.Namespace) -> int:
    """Top-level entry. Dispatches to the refine pipeline when ``--refine``
    is set, else runs the single-pass pipeline."""
    if getattr(args, "refine", False):
        return _run_with_refine(args)
    return _run_pipeline_once(args)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = _build_arg_parser()
    ns = parser.parse_args()
    sys.exit(main(ns))
