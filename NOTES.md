# PO-LEU Build Log

Source of truth: [`docs/redesign.md`](docs/redesign.md). Every deviation below is
also a note to self about what the paper/codebase must disclose.

---

## Scaffold

**What was created**
- §14 directory tree (`src/{data,outcomes,model,train,eval,baselines}`, `configs/`, `tests/`, `scripts/`, `outcomes_cache/`, `embeddings_cache/`, `checkpoints/`, `reports/`).
- `pyproject.toml` (Python 3.11+, torch, numpy, pyyaml, sentence-transformers, tqdm, pandas, scikit-learn; `[dev]` extras for pytest; `[llm]` extras for anthropic).
- `configs/default.yaml` encoding every row of redesign.md Appendix B plus a few structural keys (paths, subsample block, eval strata list).
- `tests/conftest.py` with `synthetic_batch` fixture (B=4, J=10, K=3, d_e=768, p=26) and a `default_config` loader.
- `README.md` with the wave plan.
- `src/train/subsample.py` copied verbatim from `old_pipeline/src/subsample.py` (Appendix C).

**Spec decisions recorded here (not deviations, just places redesign.md did not fully specify)**
- Repo root is `/Users/wesleylu/Desktop/structure_descent/` rather than a nested `po-leu/` directory — §14 showed a project root by that name, but we are already inside the checkout. The relative layout under §14 is preserved.
- Default `outcomes.generator.model_id = "stub"` so hermetic tests never hit an API. A real-API client swap is documented in the final integration summary.
- Default `subsample.enabled = false` because the smoke test runs on synthetic data (Appendix C importance weights degenerate to 1.0).

---

## Wave 1 — data prep + cache + prompts

(pending)

---

## Wave 2 — LLM generation + diversity filter

(pending)

---

## Wave 3 — encoder

(pending)

---

## Wave 4 — model modules

(pending)

---

## Wave 5 — assembly + ablations

(pending)

---

## Wave 6 — training + eval

(pending)

---

## Wave 7 — interpretability + ablation configs

(pending)

---

## Final integration

(pending)
