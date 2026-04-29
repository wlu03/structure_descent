"""Run the A7 (ConcatUtility) leg of the interpretability-tax experiment.

Consumes an existing PO-LEU run directory's ``records.pkl`` (so A0 and A7
score the EXACT same train/val/test split), reuses the outcomes /
embeddings caches verbatim (zero new LLM calls for any slice the A0 run
already covered), trains ConcatUtility with the same TrainConfig PO-LEU
used, and writes ``test_logits.npz`` in the format
``scripts/run_baselines.py --poleu-logits`` consumes.

Usage
-----

    python scripts/run_interpretability_tax.py \\
        --records-from reports/records_roundtrip_seed99 \\
        --output-dir   reports/interp_tax_A7_seed99 \\
        --n-epochs 30 --batch-size 32 --seed 99

The records.pkl path is the *directory* of an A0 run (or a path ending in
``records.pkl``); the script locates the file automatically.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.data.adapter import YamlAdapter  # noqa: E402
from src.data.batching import assemble_batch, iter_to_torch_batches  # noqa: E402
from src.eval.metrics import compute_all  # noqa: E402
from src.model.ablations import ConcatUtility  # noqa: E402
from src.outcomes.cache import EmbeddingsCache, OutcomesCache  # noqa: E402
from src.outcomes.diversity_filter import diversity_filter as _default_div_filter  # noqa: E402
from src.outcomes.encode import SentenceTransformersEncoder  # noqa: E402
from src.outcomes.generate import AnthropicLLMClient  # noqa: E402
from src.train.loop import TrainConfig, fit  # noqa: E402

logger = logging.getLogger("run_interpretability_tax")


def _resolve_records_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        p = p / "records.pkl"
    if not p.exists():
        raise SystemExit(f"records.pkl not found at {p}")
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-from", required=True, type=str,
                        help="Path to an A0 run dir or its records.pkl.")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--dataset-yaml", type=str,
                        default=str(REPO_ROOT / "configs/datasets/amazon.yaml"))
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load records ------------------------------------------------
    records_path = _resolve_records_path(args.records_from)
    logger.info("loading records from %s", records_path)
    with records_path.open("rb") as fh:
        bundle = pickle.load(fh)
    records_train, records_val, records_test = (
        bundle["train"], bundle["val"], bundle["test"]
    )
    logger.info(
        "records: train=%d val=%d test=%d",
        len(records_train), len(records_val), len(records_test),
    )

    # ---- 2. adapter + LLM/encoder (cache-hit paths) ---------------------
    adapter = YamlAdapter(Path(args.dataset_yaml))
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set; required for cache key parity.")
    llm_client = AnthropicLLMClient(
        model_id=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        api_key=api_key,
    )
    encoder = SentenceTransformersEncoder(
        model_id="sentence-transformers/all-mpnet-base-v2",
        max_length=64, pooling="mean",
    )
    outcomes_cache = OutcomesCache(REPO_ROOT / "outcomes_cache/outcomes.sqlite")
    embeddings_cache = EmbeddingsCache(REPO_ROOT / "embeddings_cache/embeddings.sqlite")
    div_filter = _default_div_filter

    # ---- 3. assemble batches (cache hits for the A0 slice) -------------
    logger.info("assemble_batch (train) — expect cache hits")
    omega = np.ones(len(records_train), dtype=np.float32)
    batch_train = assemble_batch(
        records_train, adapter=adapter, llm_client=llm_client, encoder=encoder,
        outcomes_cache=outcomes_cache, embeddings_cache=embeddings_cache,
        K=int(args.K), seed=int(args.seed), diversity_filter=div_filter,
        omega=omega,
    )
    batch_val = assemble_batch(
        records_val, adapter=adapter, llm_client=llm_client, encoder=encoder,
        outcomes_cache=outcomes_cache, embeddings_cache=embeddings_cache,
        K=int(args.K), seed=int(args.seed), diversity_filter=div_filter,
        omega=None,
    ) if records_val else batch_train
    batch_test = assemble_batch(
        records_test, adapter=adapter, llm_client=llm_client, encoder=encoder,
        outcomes_cache=outcomes_cache, embeddings_cache=embeddings_cache,
        K=int(args.K), seed=int(args.seed), diversity_filter=div_filter,
        omega=None,
    ) if records_test else batch_val

    p = int(batch_train.z_d.shape[1])
    J = int(batch_train.E.shape[1])
    d_e = int(batch_train.E.shape[3])
    logger.info("shapes: p=%d J=%d K=%d d_e=%d", p, J, int(args.K), d_e)

    # ---- 4. ConcatUtility (A7) -----------------------------------------
    model = ConcatUtility(
        d_e=d_e, p=p,
        hidden=int(args.hidden),
        salience_hidden=64,
        uniform_salience=False,
        temperature=float(args.temperature),
    )
    n_params = int(sum(p_.numel() for p_ in model.parameters() if p_.requires_grad))
    logger.info("ConcatUtility: n_params=%d", n_params)

    # ---- 5. fit (no PO-LEU regularizers; A7 has no weight_net / heads) -
    train_cfg = TrainConfig.from_default()
    train_cfg.batch_size = int(args.batch_size)
    train_cfg.max_epochs = int(args.n_epochs)
    # Post-hoc T-scaling assigns a float to model.temperature; ConcatUtility
    # registers temperature as a buffer (not a free attribute), which raises
    # at assignment. Disable for A7; the comparison to A0 stays fair because
    # A7's logit scale is already calibrated by training.
    train_cfg.temperature_scaling = False

    train_gen = torch.Generator().manual_seed(int(args.seed))

    def _train_batches():
        return iter_to_torch_batches(
            batch_train, batch_size=train_cfg.batch_size,
            shuffle=True, generator=train_gen,
        )

    def _val_batches():
        return iter_to_torch_batches(
            batch_val, batch_size=train_cfg.batch_size, shuffle=False,
        )

    n_train_batches = max(1, (len(batch_train) + train_cfg.batch_size - 1)
                         // train_cfg.batch_size)
    total_steps = n_train_batches * train_cfg.max_epochs

    t0 = time.perf_counter()
    state = fit(
        model, _train_batches, _val_batches,
        train_cfg=train_cfg, reg_cfg=None,
        total_steps=total_steps, seed=int(args.seed),
        on_epoch_end=lambda s: logger.info(
            "epoch=%d val_nll=%.4f best=%.4f early=%s",
            s.epoch, s.val_nll, s.best_val_nll, s.stopped_early,
        ),
    )
    fit_seconds = time.perf_counter() - t0
    logger.info(
        "fit done: epochs=%d val_nll=%.4f best=%.4f early=%s fit=%.1fs",
        state.epoch, state.val_nll, state.best_val_nll, state.stopped_early,
        fit_seconds,
    )

    # ---- 6. evaluate on test, save logits ------------------------------
    model.eval()
    with torch.no_grad():
        all_logits = []
        all_c_star = []
        for batch in iter_to_torch_batches(
            batch_test, batch_size=train_cfg.batch_size, shuffle=False
        ):
            logits, _ = model(batch["z_d"], batch["E"])
            all_logits.append(logits.cpu().numpy())
            all_c_star.append(batch["c_star"].cpu().numpy())
    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
    c_star_arr = np.concatenate(all_c_star, axis=0).astype(np.int64)

    em = compute_all(
        logits_arr, c_star_arr, n_params=n_params, n_train=len(batch_train),
    )
    metrics = {
        "top1": float(em.top1), "top5": float(em.top5), "mrr": float(em.mrr_val),
        "test_nll": float(em.nll_val), "aic": float(em.aic_val),
        "bic": float(em.bic_val), "n_params": int(n_params),
        "n_test": int(len(batch_test)), "n_train": int(len(batch_train)),
        "fit_seconds": float(fit_seconds),
    }

    np.savez(out_dir / "test_logits.npz",
             logits=logits_arr, c_star=c_star_arr,
             n_params=np.array(n_params, dtype=np.int64))
    import json
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    logger.info("wrote %s", out_dir / "test_logits.npz")
    logger.info("wrote %s", out_dir / "metrics_test.json")

    print()
    print(f"A7 ConcatUtility on n_test={metrics['n_test']}:")
    print(f"  top1={metrics['top1']:.3f}  top5={metrics['top5']:.3f}  "
          f"MRR={metrics['mrr']:.4f}  NLL={metrics['test_nll']:.4f}")
    print(f"  AIC={metrics['aic']:.1f}  BIC={metrics['bic']:.1f}  "
          f"params={metrics['n_params']}  fit={metrics['fit_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
