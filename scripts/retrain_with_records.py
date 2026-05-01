"""Replay-train PO-LEU from a saved records.pkl, no LLM calls.

Bypasses ``build_choice_sets`` so it inherits the exact c_d strings the
original sweep used → outcomes_cache hits 100% → zero LLM API spend.
Use to swap training-only knobs (higher residual β LR, different
regularizers, etc.) and compare to the original run.

Usage::

    python -m scripts.retrain_with_records \\
        --records   results_data/poleu_25cust_seed7_residual/records.pkl \\
        --config    configs/higher_beta.yaml \\
        --output-dir results_data/seed7_higher_beta \\
        --tabular-residual true \\
        --seed 7

Reads ``OUTCOMES_CACHE_PATH`` / ``EMBEDDINGS_CACHE_PATH`` env vars to
point at the per-seed cache from the original run (e.g.
``outcomes_cache/seed7/outcomes.sqlite``). Same env-var contract as
``scripts.run_dataset``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrain_with_records.py", description=__doc__)
    p.add_argument("--records", type=Path, required=True,
                   help="Path to records.pkl from a prior PO-LEU run.")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs" / "default.yaml")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7,
                   help="Used for the model RNG init only — does NOT affect "
                        "the records (those come from --records as-is).")
    p.add_argument("--n-epochs", type=int, default=None,
                   help="Override max_epochs. Default: read from YAML.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size. Default: read from YAML.")
    p.add_argument("--tabular-residual", choices=["yaml", "true", "false"],
                   default="yaml",
                   help="Override model.tabular_residual.enabled.")
    p.add_argument("--residual-lr-multiplier", type=float, default=None,
                   help="Override train.residual_lr_multiplier from config.")
    p.add_argument(
        "--K", type=int, default=None,
        help=(
            "Override outcomes.K from config. Must match the K the original "
            "outcomes cache was populated with — different K → different "
            "cache_prompt_version → 100%% cache miss → cold LLM calls."
        ),
    )
    p.add_argument(
        "--prompt-version-cascade",
        nargs="+",
        default=None,
        metavar="VERSION",
        help=(
            "Curriculum-refinement cascade forwarded to assemble_batch. "
            "Must match the prompt_version(s) the original outcomes cache "
            "was populated with (e.g. ``v4_mobility_anchored``) — different "
            "prompt_version → different cache_prompt_version → 100%% cache "
            "miss → cold LLM calls."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output_dir=%s", out_dir)

    # 1. Load records.
    if not args.records.exists():
        raise SystemExit(f"records not found: {args.records}")
    bundle = pickle.loads(args.records.read_bytes())
    train_recs = list(bundle["train"])
    val_recs = list(bundle["val"])
    test_recs = list(bundle["test"])
    logger.info("loaded records: train=%d val=%d test=%d (from %s)",
                len(train_recs), len(val_recs), len(test_recs), args.records)

    # 2. Load config + apply CLI overrides.
    config = yaml.safe_load(args.config.read_text()) or {}
    if args.tabular_residual != "yaml":
        forced = args.tabular_residual == "true"
        config.setdefault("model", {}).setdefault("tabular_residual", {})["enabled"] = forced
        logger.info("CLI override: tabular_residual.enabled=%s", forced)
    if args.n_epochs is not None:
        config.setdefault("train", {})["max_epochs"] = int(args.n_epochs)
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.residual_lr_multiplier is not None:
        config.setdefault("train", {})["residual_lr_multiplier"] = float(args.residual_lr_multiplier)
        logger.info("CLI override: train.residual_lr_multiplier=%.2f", args.residual_lr_multiplier)

    # 3. Build llm_client + encoder + caches (real-only — same as run_dataset).
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY missing — driver is real-LLM-only.")

    from src.outcomes.encode import SentenceTransformersEncoder
    from src.outcomes.generate import AnthropicLLMClient

    enc_cfg = (config.get("outcomes") or {}).get("encoder") or {}
    model_id = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    llm_client = AnthropicLLMClient(model_id=model_id, api_key=api_key)
    encoder = SentenceTransformersEncoder(
        model_id=enc_cfg.get("model_id", "sentence-transformers/all-mpnet-base-v2"),
        max_length=int(enc_cfg.get("max_length", 64)),
        pooling=enc_cfg.get("pooling", "mean"),
    )

    # Cache paths — env vars override (same contract as run_dataset).
    paths_cfg = config.get("paths") or {}
    cache_cfg = (config.get("outcomes") or {}).get("cache") or {}
    env_outcomes = os.environ.get("OUTCOMES_CACHE_PATH", "").strip()
    env_embeddings = os.environ.get("EMBEDDINGS_CACHE_PATH", "").strip()
    outcomes_cache_path = Path(env_outcomes or cache_cfg.get(
        "outcomes_path", paths_cfg.get("outcomes_cache", "outcomes_cache/") + "outcomes.sqlite"))
    embeddings_cache_path = Path(env_embeddings or cache_cfg.get(
        "embeddings_path", paths_cfg.get("embeddings_cache", "embeddings_cache/") + "embeddings.sqlite"))
    if not outcomes_cache_path.is_absolute():
        outcomes_cache_path = REPO_ROOT / outcomes_cache_path
    if not embeddings_cache_path.is_absolute():
        embeddings_cache_path = REPO_ROOT / embeddings_cache_path
    logger.info("cache: outcomes=%s, embeddings=%s",
                outcomes_cache_path, embeddings_cache_path)

    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.diversity_filter import diversity_filter as _div_filter

    outcomes_cache = OutcomesCache(outcomes_cache_path)
    embeddings_cache = EmbeddingsCache(embeddings_cache_path)

    # 4. Resolve tabular feature config.
    tab_cfg = (config.get("model") or {}).get("tabular_residual") or {}
    tab_enabled = bool(tab_cfg.get("enabled", False))
    tab_features = list(tab_cfg.get("features") or [])
    tab_names: list[str] | None = (
        tuple(tab_features) if (tab_enabled and tab_features) else None
    )
    if tab_names is not None:
        logger.info("tabular_residual: features=%s", list(tab_names))

    # 5. assemble_batch — should hit cache on every (cust, alt) since
    #    records' c_d strings are identical to the original sweep's.
    #    K and prompt_version_cascade are folded into the cache_prompt
    #    _version key, so any drift between this run's pair and the
    #    original-write's pair causes 100% cache miss + cold LLM calls.
    from src.data.batching import assemble_batch
    K = int(args.K) if args.K is not None else int(
        (config.get("outcomes") or {}).get("K", 3)
    )
    cascade = (
        tuple(args.prompt_version_cascade)
        if args.prompt_version_cascade else None
    )
    if cascade:
        logger.info("prompt_version_cascade=%s K=%d", list(cascade), K)
    else:
        logger.info("prompt_version_cascade unset; using assemble_batch default v1 K=%d", K)

    logger.info("stage: assemble_batch (train) — expect 100%% cache hits")
    t0 = time.perf_counter()
    batch_train = assemble_batch(
        train_recs, adapter=None, llm_client=llm_client, encoder=encoder,
        outcomes_cache=outcomes_cache, embeddings_cache=embeddings_cache,
        K=K, seed=int(args.seed),
        prompt_version_cascade=cascade,
        diversity_filter=_div_filter,
        tabular_feature_names=tab_names,
    )
    logger.info("  train assembled in %.1fs", time.perf_counter() - t0)
    logger.info("stage: assemble_batch (val)")
    batch_val = assemble_batch(
        val_recs if val_recs else train_recs[:1],
        adapter=None, llm_client=llm_client, encoder=encoder,
        outcomes_cache=outcomes_cache, embeddings_cache=embeddings_cache,
        K=K, seed=int(args.seed),
        prompt_version_cascade=cascade,
        diversity_filter=_div_filter,
        tabular_feature_names=tab_names,
    )
    logger.info("stage: assemble_batch (test)")
    batch_test = assemble_batch(
        test_recs if test_recs else val_recs,
        adapter=None, llm_client=llm_client, encoder=encoder,
        outcomes_cache=outcomes_cache, embeddings_cache=embeddings_cache,
        K=K, seed=int(args.seed),
        prompt_version_cascade=cascade,
        diversity_filter=_div_filter,
        tabular_feature_names=tab_names,
    )

    # 6. Build POLEU.
    from src.model.po_leu import POLEU
    p = int(batch_train.z_d.shape[1])
    J = int(batch_train.E.shape[1])
    d_e = int(batch_train.E.shape[3])

    model_cfg = config.get("model") or {}
    attr_cfg = model_cfg.get("attribute_heads") or {}
    wnet_cfg = model_cfg.get("weight_net") or {}
    snet_cfg = model_cfg.get("salience_net") or {}
    _vocab = getattr(batch_train, "category_vocab", ())
    n_categories_runtime = max(1, len(_vocab))
    poleu_kwargs: dict[str, Any] = dict(
        M=int(model_cfg.get("M", 5)), K=K, J=J, d_e=d_e, p=p,
        attribute_hidden=int(attr_cfg.get("hidden", 128)),
        weight_hidden=int(wnet_cfg.get("hidden", 32)),
        salience_hidden=int(snet_cfg.get("hidden", 64)),
        weight_normalization=str(wnet_cfg.get("normalization", "softmax")),
        uniform_salience=bool(model_cfg.get("uniform_salience", False)),
        temperature=float(model_cfg.get("temperature", 1.0)),
        n_categories=n_categories_runtime,
        d_cat=int(snet_cfg.get("d_cat", 8)),
    )
    if tab_names is not None and batch_train.x_tab is not None:
        poleu_kwargs["tabular_residual_enabled"] = True
        poleu_kwargs["tabular_features"] = tuple(tab_names)
    model = POLEU(**poleu_kwargs)

    if (tab_names is not None and batch_train.x_tab is not None
            and getattr(model, "tabular_residual_enabled", False)):
        flat = batch_train.x_tab.reshape(-1, batch_train.x_tab.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        model.set_tabular_feature_stats(mean, std)
        logger.info("tabular stats fit on train: mean=%s std=%s",
                    mean.tolist(), std.tolist())

    # 7. Train.
    from src.train.loop import TrainConfig, fit
    from src.train.regularizers import RegularizerConfig
    from src.data.batching import iter_to_torch_batches

    train_cfg = TrainConfig.from_default()  # picks up residual_lr_multiplier from YAML
    # Re-load the user's YAML on top so per-experiment overrides apply.
    train_block = (config.get("train") or {})
    for fld in ("batch_size", "lr", "lr_min", "max_epochs",
                "early_stopping_patience", "grad_clip", "residual_lr_multiplier"):
        if fld in train_block:
            setattr(train_cfg, fld, type(getattr(train_cfg, fld))(train_block[fld]))

    reg_block = config.get("regularizers") or {}
    reg_cfg = RegularizerConfig(
        weight_l2=float(reg_block.get("weight_l2", 0.0) or 0.0),
        salience_entropy=float(reg_block.get("salience_entropy", 0.0) or 0.0),
        diversity=float(reg_block.get("diversity", 0.0) or 0.0),
        head_variance=float(reg_block.get("head_variance", 0.0) or 0.0),
    )
    mono_block = reg_block.get("monotonicity") or {}
    if bool(mono_block.get("enabled", False)):
        reg_cfg.monotonicity_lambda = float(mono_block.get("lambda", 0.0) or 0.0)
        reg_cfg.monotonicity_price_key = str(mono_block.get("price_key", "price"))

    def train_batches_fn():
        return iter_to_torch_batches(batch_train, batch_size=train_cfg.batch_size, shuffle=True)

    def val_batches_fn():
        return iter_to_torch_batches(batch_val, batch_size=train_cfg.batch_size, shuffle=False)

    import math
    n_batches = max(1, math.ceil(len(batch_train) / train_cfg.batch_size))
    total_steps = n_batches * train_cfg.max_epochs
    logger.info("stage: fit (residual_lr_multiplier=%.2f, max_epochs=%d, batch_size=%d, n_train=%d)",
                train_cfg.residual_lr_multiplier, train_cfg.max_epochs,
                train_cfg.batch_size, len(batch_train))
    state = fit(model, train_batches_fn, val_batches_fn, train_cfg=train_cfg,
                reg_cfg=reg_cfg, total_steps=total_steps, seed=int(args.seed))

    # 8. Eval + write outputs.
    from src.eval.metrics import compute_all
    model.eval()
    with torch.no_grad():
        if batch_test.x_tab is not None and getattr(model, "tabular_residual_enabled", False):
            logits_test, _ = model(batch_test.z_d, batch_test.E, batch_test.x_tab)
        else:
            logits_test, _ = model(batch_test.z_d, batch_test.E)
    test_metrics = compute_all(logits_test, batch_test.c_star,
                               n_params=model.num_params(), n_train=len(batch_train))
    (out_dir / "metrics_test.json").write_text(json.dumps({
        **test_metrics.to_dict(), "n_test_records": int(len(batch_test)),
    }, indent=2))
    np.savez(
        out_dir / "test_logits.npz",
        logits=logits_test.detach().cpu().numpy().astype("float32"),
        c_star=batch_test.c_star.detach().cpu().numpy().astype("int64"),
        n_params=np.array(int(model.num_params()), dtype=np.int64),
    )

    summary = {
        "records_from": str(args.records),
        "config_used": str(args.config),
        "residual_lr_multiplier": train_cfg.residual_lr_multiplier,
        "tabular_residual_enabled": getattr(model, "tabular_residual_enabled", False),
        "n_train": len(batch_train), "n_val": len(batch_val), "n_test": len(batch_test),
        "n_params": int(model.num_params()),
        "train_state": asdict(state),
        "metrics_test": test_metrics.to_dict(),
    }
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nretrain_with_records OK")
    print(f"  test top1={test_metrics.top1*100:.2f}%  top3={test_metrics.top3*100:.2f}%  "
          f"top5={test_metrics.top5*100:.2f}%")
    print(f"  test NLL={test_metrics.nll_val:.4f}  Brier={test_metrics.brier_val:.4f}")
    print(f"  → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
