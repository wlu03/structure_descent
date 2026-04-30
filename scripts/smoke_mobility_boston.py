"""Structural smoke for mobility_boston: full PO-LEU pipeline on stubs.

Mirrors scripts/run_dataset.py up through model.fit, but uses StubLLMClient
+ StubEncoder so we can exercise the whole adapter chain (YAML schema ->
clean -> survey-join -> state-features -> split -> popularity -> brand ->
build_choice_sets -> assemble_batch -> POLEU.fit -> compute_all) without
making real LLM calls. The point is structural: prove the mobility data
flows through every layer of the framework end-to-end. Real-LLM training
is a separate run via scripts/run_dataset.py once outcome budgets are set.

Usage
-----

    python scripts/smoke_mobility_boston.py [--n-customers 30] [--n-epochs 1]

Writes a smoke_summary.json into reports/mobility_boston_stub/ with
structural counts and untrained metrics (the latter approximate random
chance because stub embeddings carry no preference signal).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

logger = logging.getLogger("smoke_mobility_boston")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-yaml", type=Path,
                   default=REPO_ROOT / "configs" / "datasets" / "mobility_boston.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "reports" / "mobility_boston_stub")
    p.add_argument("--n-customers", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--K", type=int, default=5,
                   help="Outcomes per alternative. K=5 matches the "
                        "v4_mobility_anchored prompt's 5 axes.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-events-per-customer", type=int, default=5)
    p.add_argument("--prompt-version", type=str,
                   default="v4_mobility_anchored",
                   help="Prompt version for outcome generation. "
                        "v4_mobility_anchored = mobility-tuned heads.")
    p.add_argument("--add-event-time", action="store_true", default=True,
                   help="Render per-event time-of-day phrase into c_d.")
    return p


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))

    from src.data.adapter import YamlAdapter
    from src.data.alt_rendering import register_on_adapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import extract_extra_fields_from_row
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_brand_map, attach_train_popularity, compute_state_features,
    )
    from src.data.survey_join import join_survey
    from src.train.loop import try_import_subsample_weights

    adapter = YamlAdapter(args.dataset_yaml)
    logger.info("adapter=%s yaml=%s", adapter.name, args.dataset_yaml)

    events = clean_events(adapter.load_events(), adapter.schema)
    persons_raw = adapter.load_persons()
    events = join_survey(events, persons_raw, adapter.schema)
    events = compute_state_features(events)

    counts = events.groupby("customer_id").size()
    keep = set(counts[counts >= int(args.min_events_per_customer)].index)
    events = events[events["customer_id"].isin(keep)].reset_index(drop=True)
    events = temporal_split(events, adapter.schema)
    events = attach_train_popularity(events)
    events = attach_train_brand_map(events)

    train_events = events[events["split"] == "train"].copy()
    selected_ids, _w = try_import_subsample_weights(
        train_events, n_customers=int(args.n_customers), seed=int(args.seed),
    )
    if selected_ids is None:
        logger.error("subsample failed")
        return 1
    selected = set(selected_ids.tolist() if hasattr(selected_ids, "tolist") else list(selected_ids))
    events = events[events["customer_id"].isin(selected)].reset_index(drop=True)
    train_events_subset = events[events["split"] == "train"].copy()

    register_on_adapter(adapter, train_events_subset)

    persons_id_col = adapter.schema.persons_id_column
    surveyed = set(persons_raw[persons_id_col].dropna().astype(str).unique().tolist())
    joint = set(events["customer_id"].astype(str).unique().tolist()) & surveyed
    events = events[events["customer_id"].astype(str).isin(joint)].reset_index(drop=True)
    train_events_subset = events[events["split"] == "train"].copy()

    persons_raw_keep = persons_raw[
        persons_raw[persons_id_col].astype(str).isin(joint)
    ].copy()
    persons_canonical = adapter.translate_z_d(
        persons_raw_keep, training_events=train_events_subset,
    )

    if len(train_events_subset) > 0 and "purchase_frequency" in persons_canonical.columns:
        d = pd.to_datetime(train_events_subset["order_date"])
        weeks = max((d.max() - d.min()).days / 7.0, 1.0)
        persons_canonical = persons_canonical.copy()
        persons_canonical["purchase_frequency"] = (
            persons_canonical["purchase_frequency"].astype(float) / weeks
        )

    surviving = set(persons_canonical["customer_id"])
    events = events[events["customer_id"].isin(surviving)].reset_index(drop=True)

    yaml_doc = yaml.safe_load(args.dataset_yaml.read_text(encoding="utf-8")) or {}
    extras_block = (
        yaml_doc.get("dataset", {}).get("persons", {}).get("c_d_extra_fields", {})
    ) or {}
    customer_to_extras = {}
    if extras_block:
        for _, raw_row in persons_raw.iterrows():
            cid = str(raw_row.get(persons_id_col, ""))
            if cid not in surviving:
                continue
            customer_to_extras[cid] = extract_extra_fields_from_row(
                raw_row.to_dict(), extras_block,
            )

    records = build_choice_sets(
        events, persons_canonical, adapter,
        seed=int(args.seed),
        n_resamples=int(adapter.schema.n_resamples),
        n_negatives=int(adapter.schema.choice_set_size) - 1,
        customer_to_extras=customer_to_extras or None,
        add_event_time_to_c_d=bool(args.add_event_time),
    )
    rec_train = [r for r in records if r.get("split") == "train"]
    rec_val   = [r for r in records if r.get("split") == "val"]
    rec_test  = [r for r in records if r.get("split") == "test"]
    logger.info("records: train=%d val=%d test=%d", len(rec_train), len(rec_val), len(rec_test))

    # Show a sample c_d so we can eyeball the rendered narrative.
    sample_cd = rec_train[0]["c_d"] if rec_train else "(no train records)"

    # Stub LLM + StubEncoder for hermetic PO-LEU forward + backward.
    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter
    from src.outcomes.encode import StubEncoder
    from src.outcomes.generate import StubLLMClient
    from src.data.batching import assemble_batch, iter_to_torch_batches

    out_cache = OutcomesCache(args.output_dir / "outcomes.sqlite")
    emb_cache = EmbeddingsCache(args.output_dir / "embeddings.sqlite")

    pv = str(args.prompt_version)

    def _assemble(recs):
        return assemble_batch(
            recs, adapter=adapter,
            llm_client=StubLLMClient(),
            encoder=StubEncoder(encoder_id="stub-mobility", d_e=64),
            outcomes_cache=out_cache, embeddings_cache=emb_cache,
            K=int(args.K), seed=int(args.seed),
            prompt_version=pv,
            diversity_filter=diversity_filter,
            omega=None,
        )

    batch_train = _assemble(rec_train)
    batch_val   = _assemble(rec_val) if rec_val else batch_train
    batch_test  = _assemble(rec_test) if rec_test else batch_val

    from src.model.po_leu import POLEU
    from src.train.loop import TrainConfig, fit
    from src.train.regularizers import RegularizerConfig
    from src.eval.metrics import compute_all

    p_eff = int(batch_train.z_d.shape[1])
    J = int(batch_train.E.shape[1])
    d_e = int(batch_train.E.shape[3])

    n_categories = max(1, len(getattr(batch_train, "category_vocab", ()) or ()))
    model = POLEU(K=int(args.K), J=J, d_e=d_e, p=p_eff,
                  n_categories=n_categories)
    tcfg = TrainConfig.from_default()
    tcfg.batch_size = int(args.batch_size)
    tcfg.max_epochs = int(args.n_epochs)
    rcfg = RegularizerConfig.from_default()

    g_train = torch.Generator().manual_seed(int(args.seed))

    def _train_iter():
        return iter_to_torch_batches(batch_train, batch_size=tcfg.batch_size,
                                     shuffle=True, generator=g_train)

    def _val_iter():
        return iter_to_torch_batches(batch_val, batch_size=tcfg.batch_size,
                                     shuffle=False)

    n_per_epoch = max(1, math.ceil(len(batch_train) / tcfg.batch_size))
    total_steps = n_per_epoch * tcfg.max_epochs

    state = fit(model, _train_iter, _val_iter, train_cfg=tcfg, reg_cfg=rcfg,
                total_steps=total_steps, seed=int(args.seed))

    model.eval()
    with torch.no_grad():
        logits_test, _interm = model(batch_test.z_d, batch_test.E)
    metrics = compute_all(logits_test, batch_test.c_star,
                          n_params=model.num_params(),
                          n_train=len(batch_train))

    summary = {
        "structural": True,
        "stub_llm": True,
        "n_customers_selected": len(selected & surviving),
        "n_train_records": len(batch_train),
        "n_val_records": len(batch_val),
        "n_test_records": len(batch_test),
        "p": p_eff, "J": J, "K": int(args.K), "d_e": d_e,
        "n_categories": n_categories,
        "param_count": model.num_params(),
        "epochs": tcfg.max_epochs,
        "train_loss": float(state.train_loss),
        "val_nll": float(state.val_nll) if state.val_nll is not None else None,
        "metrics_test": metrics.to_dict(),
        "sample_c_d": sample_cd,
    }
    summary_path = args.output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"smoke_mobility_boston OK  n_train={len(batch_train)} "
          f"top1={metrics.top1:.4f}  out={summary_path}")
    out_cache.close()
    emb_cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(_build_arg_parser().parse_args()))
