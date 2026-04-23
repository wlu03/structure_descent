"""Wave 10 end-to-end orchestration driver (design doc §4).

CLI-driven orchestration of the full PO-LEU pipeline. Takes an adapter
name, a customer budget, a stub/real LLM flag, and an output dir; produces
a fully materialized training run plus evaluation + interpretability
reports.

Usage
-----

    python scripts/run_dataset.py \\
        --adapter amazon \\
        --n-customers 100 \\
        --stub-llm \\
        --n-epochs 1 \\
        --batch-size 32 \\
        --output-dir reports/amazon_smoke

See ``NOTES.md`` ("Wave 10 — glue modules") for the full design contract,
especially the val/test z_d fit caveat (val/test currently reuses the
train-only fit statistics, deferred to Wave 11/12).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Heavy / optional imports kept lazy inside main() per the Wave-10 brief
# (anthropic + sentence_transformers are imported only on --real-llm).
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
        choices=["amazon", "synthetic"],
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

    llm_group = parser.add_mutually_exclusive_group(required=False)
    llm_group.add_argument(
        "--stub-llm",
        dest="stub_llm",
        action="store_true",
        help="Use hermetic StubLLMClient + StubEncoder (default).",
    )
    llm_group.add_argument(
        "--real-llm",
        dest="real_llm",
        action="store_true",
        help="Use AnthropicLLMClient + SentenceTransformersEncoder.",
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
    """Instantiate the LLM client + sentence encoder.

    Stub backends are the default and never import heavy deps. Real
    backends are imported lazily inside the --real-llm branch so the
    script can run in environments without anthropic / sentence_transformers.
    """
    if args.real_llm:
        # Lazy imports; see the Wave-10 brief "DO NOT import anthropic or
        # sentence_transformers at module top level".
        import os

        from src.outcomes.encode import SentenceTransformersEncoder
        from src.outcomes.generate import AnthropicLLMClient

        enc_cfg = (config.get("outcomes") or {}).get("encoder") or {}
        model_id = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5")
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        llm_client: Any = AnthropicLLMClient(model_id=model_id, api_key=api_key)
        encoder: Any = SentenceTransformersEncoder(
            model_id=enc_cfg.get("model_id",
                                 "sentence-transformers/all-mpnet-base-v2"),
            max_length=int(enc_cfg.get("max_length", 64)),
            pooling=enc_cfg.get("pooling", "mean"),
        )
    else:
        from src.outcomes.encode import StubEncoder
        from src.outcomes.generate import StubLLMClient

        llm_client = StubLLMClient()
        encoder = StubEncoder(d_e=768)
    return llm_client, encoder


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(args: argparse.Namespace) -> int:
    """Orchestrate the Wave 10 pipeline. Returns a POSIX-style exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Argparse default handling: when neither --stub-llm nor --real-llm is
    # given, default to stub.
    if not args.real_llm:
        args.stub_llm = True

    torch.manual_seed(int(args.seed))

    # ---- 1. resolve paths + config -------------------------------------
    out_dir = _resolve_output_dir(args)
    logger.info("output_dir=%s", out_dir)

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    config = _load_yaml(config_path)

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
        attach_train_popularity,
        compute_state_features,
    )
    from src.data.split import temporal_split
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

    # ---- 5. temporal split + popularity --------------------------------
    logger.info("stage: temporal_split")
    events = temporal_split(events, adapter.schema)
    logger.info("stage: attach_train_popularity")
    events = attach_train_popularity(events)

    train_events = events[events["split"] == "train"].copy()

    # ---- 6. Appendix-C subsample --------------------------------------
    logger.info("stage: subsample (n_customers=%d)", int(args.n_customers))
    from src.train.loop import try_import_subsample_weights

    selected_ids, event_weights = try_import_subsample_weights(
        train_events,
        n_customers=int(args.n_customers),
        seed=int(args.seed),
    )
    if selected_ids is None:
        logger.error(
            "Subsample required but fell back to omega=1; see WARNING above. "
            "Check that state_features ran correctly."
        )
        return 1

    selected_id_set = set(selected_ids.tolist() if hasattr(selected_ids, "tolist")
                          else list(selected_ids))
    events_subset = events[events["customer_id"].isin(selected_id_set)].reset_index(
        drop=True
    )
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
    train_customers = set(
        train_events_subset["customer_id"].astype(str).unique().tolist()
    )
    joint_customers = train_customers & surveyed_customers
    if not joint_customers:
        raise SystemExit(
            "orphan filter: no customers in both train events and survey "
            "(train=%d, survey=%d)." % (len(train_customers), len(surveyed_customers))
        )
    logger.info(
        "orphan filter: %d customers in train∩survey (train=%d, survey=%d).",
        len(joint_customers),
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

    logger.info("stage: build_choice_sets (train+val+test rows together)")
    records_all = build_choice_sets(
        events_subset,
        persons_canonical,
        adapter,
        seed=int(args.seed),
        n_resamples=int(adapter.schema.n_resamples),
        n_negatives=int(adapter.schema.choice_set_size) - 1,
        customer_to_extras=customer_to_extras or None,
    )

    # Split records by the attached event's split (via per-record index).
    split_by_idx = events_subset["split"].tolist()
    records_train = [r for r, s in zip(records_all, split_by_idx) if s == "train"]
    records_val = [r for r, s in zip(records_all, split_by_idx) if s == "val"]
    records_test = [r for r, s in zip(records_all, split_by_idx) if s == "test"]
    logger.info(
        "records: train=%d val=%d test=%d.",
        len(records_train),
        len(records_val),
        len(records_test),
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

    # Caches live at repo-root defaults unless overridden in config.
    paths_cfg = config.get("paths") or {}
    cache_cfg = (config.get("outcomes") or {}).get("cache") or {}
    outcomes_cache_path = Path(
        cache_cfg.get("outcomes_path",
                      (paths_cfg.get("outcomes_cache", "outcomes_cache/")
                       + "outcomes.sqlite"))
    )
    embeddings_cache_path = Path(
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

    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter as _default_div_filter

    outcomes_cache = OutcomesCache(outcomes_cache_path)
    embeddings_cache = EmbeddingsCache(embeddings_cache_path)

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
        diversity_filter=_default_div_filter,
        omega=train_event_weights_aligned,
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
            diversity_filter=_default_div_filter,
            omega=None,
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
            diversity_filter=_default_div_filter,
            omega=None,
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

    model = POLEU(
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

    # ---- 13. §12 reports (interpretability) ---------------------------
    from src.eval.interpret import run_all_reports

    logger.info("stage: run_all_reports")
    run_all_reports(
        model,
        batch_val.z_d,
        batch_val.E,
        batch_val.c_star,
        batch_val.outcomes_nested,
        out_dir=out_dir,
        event_idx=0,
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
            "llm_mode": "real" if args.real_llm else "stub",
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
# CLI entry point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = _build_arg_parser()
    ns = parser.parse_args()
    sys.exit(main(ns))
