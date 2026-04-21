# PO-LEU

Perceived-Outcome, LLM-generated, Embedding-based Utility.
Specification: [`docs/redesign.md`](docs/redesign.md).

## Status

Implemented via a wave-based orchestration. See `NOTES.md` for the running
build log, deviations from spec, and unresolved ambiguities.

## Wave plan

| Wave | Scope | Modules |
|---|---|---|
| Scaffold | tree, default config, conftest | `configs/default.yaml`, `tests/conftest.py` |
| 1 | data prep + cache + prompts | `src/data/person_features.py`, `src/data/context_string.py`, `src/outcomes/prompts.py`, `src/outcomes/cache.py` |
| 2 | LLM generation + diversity | `src/outcomes/generate.py`, `src/outcomes/diversity_filter.py` |
| 3 | encoder | `src/outcomes/encode.py` |
| 4 | model modules | `src/model/attribute_heads.py`, `src/model/weight_net.py`, `src/model/salience_net.py` |
| 5 | assembly + ablations | `src/model/po_leu.py`, `src/model/ablations.py` |
| 6 | training + eval | `src/train/regularizers.py`, `src/train/loop.py`, `src/eval/metrics.py`, `src/eval/strata.py` |
| 7 | interpretability + ablation configs | `src/eval/interpret.py`, `configs/ablation_*.yaml` |
| Final | smoke test | `scripts/smoke_end_to_end.py` |

## Data

Source data lives in `amazon_ecom/` (Amazon purchase logs + survey).
Upstream v2.0 pipeline stages (load/clean/survey-join/state-features/split)
are referenced at their interfaces only — not reimplemented here.
`src/train/subsample.py` is retained verbatim from v1 (Appendix C).

## Layout

See redesign.md §14.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Deliverable

Codebase + passing unit tests + `scripts/smoke_end_to_end.py` that executes
one forward + backward pass on synthetic data and emits an interpretability
report. No end-to-end training.
