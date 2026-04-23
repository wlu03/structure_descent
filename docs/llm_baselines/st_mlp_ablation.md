# ST+MLP Ablation — "PO-LEU minus narratives"

Status: **Design only. Do not implement until reviewed.**
Target files: `src/baselines/st_mlp_ablation.py`,
`tests/baselines/test_st_mlp_ablation.py`.
Registry entry: `STMLPChoice` in `src/baselines/run_all.py`.

## 1. Method

**Sentence-Transformer + MLP Ablation** strips the LLM-narrative step out of
PO-LEU's pipeline while holding the frozen encoder fixed. For each event, the
seven-key alternative descriptor on `records[e]["alt_texts"][j]` is rendered
as a single natural-language sentence (`_render_alt_text`), embedded with
the **same** frozen encoder PO-LEU uses
(`sentence-transformers/all-mpnet-base-v2`, `d_e = 768`, mean-pooled,
L2-normalised, 64-token truncation — identical to
`src.outcomes.encode.SentenceTransformersEncoder`), and fed through a small
two-layer MLP `(d_e → 64 → 1)`. The resulting per-alt scalar is softmaxed
across `J` alternatives and trained against `c_star` under cross-entropy.
If this baseline matches PO-LEU's accuracy, LLM-generated narratives add
no signal beyond what a frozen encoder extracts from metadata; the head
is the minimal capacity needed to map `(J, d_e) → (J,)`.

## 2. Baseline-paradigm citations

* Harte et al., *"LLMs for Sequential Recommendation"* (RecSys 2023 LBR):
  frozen sentence embeddings + light head outperform ID-only baselines in
  small-data regimes.
* Li et al., *"Bridging Language and Items"* (arXiv:2403.03952, 2024):
  sentence encoders + linear/MLP heads are a strong baseline that many
  LLM4Rec systems fail to beat.
* OpenAI Cookbook, *"Classification using embeddings"*: reference recipe.
* Hou et al., *"LLMs are Zero-Shot Rankers"* (EMNLP 2023) §6: embedding-
  nearest-neighbour baseline is the same family.

Collectively these justify "encode metadata + small MLP" as the canonical
frozen-LLM baseline for text-shaped choice tasks.

## 3. Variant choice — **B (independent MLP)**

* **Variant A (strict):** embed metadata into `(N, J, 1, d_e)` and feed
  through PO-LEU's unchanged heads. Cleanly isolates "narrative vs
  metadata" but requires `z_d`, couples to head code, and inflates
  parameter count to ~545k — not a *small independent baseline*.
* **Variant B (chosen):** generic MLP on `(N, J, d_e)`. Confounds two
  axes (narrative-vs-metadata *and* PO-LEU-head-vs-MLP) but the goal is
  a **competitive frozen-LLM baseline**, not a pure internal ablation
  of PO-LEU's heads. `src/model/ablations.py` already hosts head-level
  ablations; adding another here duplicates them.

The confound is called out explicitly in the docstring (§11) so downstream
analysis does not over-claim.

## 4. Text rendering function

The 7 keys are `title, category, price, popularity_rank, brand, is_repeat,
state` (see `src/data/adapter.py::alt_text`). One stationary sentence
template per alt:

```python
def _render_alt_text(alt: Mapping[str, object], *, drop_is_repeat: bool = False) -> str:
    """Render a 7-key alt_texts dict into a single sentence.

    * Canonical key ordering; every event uses the same template so the
      frozen encoder sees a stationary prompt.
    * Missing / None values collapse to sensible literals so the output
      is always non-empty and contains no ``None`` / ``nan`` substrings.
    * ``drop_is_repeat`` omits the is_repeat sentence — used when the
      training batch exhibits the "is_repeat=True only on the chosen
      alt" label-leak pattern (see §12.4).
    """
    title = str(alt.get("title") or "unknown title").strip()
    category = str(alt.get("category") or "unknown category").strip()
    brand = str(alt.get("brand") or "unknown_brand").strip()
    try:
        price = float(alt.get("price") or 0.0)
    except (TypeError, ValueError):
        price = 0.0
    pop_raw = alt.get("popularity_rank", 0)
    pop = str(pop_raw).strip() if pop_raw is not None else "popularity score 0"
    state = str(alt.get("state") or "").strip()

    parts = [
        f"{title}.",
        f"Category: {category}.",
        f"Brand: {brand}.",
        f"Price: ${price:.1f}.",
        f"Popularity: {pop}.",
    ]
    if not drop_is_repeat:
        parts.append(f"Repeat purchase: {bool(alt.get('is_repeat', False))}.")
    parts.append(f"State: {state}.")
    return " ".join(parts)
```

**`c_d` is intentionally omitted.** PO-LEU's encoder sees `c_d` only
indirectly through generated narratives. Injecting `c_d` into the ST+MLP
input would give this baseline a signal PO-LEU lacks at the encoder
layer, biasing the comparison.

## 5. Encoder interface

The class takes an `EncoderClient` at construction (`StubEncoder` in
tests, `SentenceTransformersEncoder` in production). Usage contract:

* `encoder.d_e` read once at `fit` to size the MLP input.
* Encoding routes through `src.outcomes.encode.encode_batch(texts,
  client=encoder, cache=...)` so the §4.3 cache is honoured identically
  to PO-LEU.

Per-batch flatten and reshape:

```python
flat_texts = [_render_alt_text(a) for ev in alt_texts_per_event for a in ev]   # N*J
flat_vecs  = encode_batch(flat_texts, client=encoder, cache=cache)             # (N*J, d_e)
E2         = flat_vecs.reshape(N, J, d_e).astype(np.float32)                   # (N, J, d_e)
```

Note the intentional absence of a `K` axis — no narratives, so the tensor
is 3-D `(N, J, d_e)` vs PO-LEU's 4-D `(N, J, K, d_e)`.

## 6. MLP architecture

Applied per alternative (broadcast over `J`):

```
Linear(d_e, hidden) → GELU → Dropout(p) → Linear(hidden, 1) → squeeze(-1)
log_softmax over J
```

Defaults: `hidden=64`, `dropout=0.1`, `GELU` (matches mpnet's internal
activation). Parameter count:

```
n_params = (d_e + 1) * hidden  +  (hidden + 1)
```

At defaults (`d_e=768, hidden=64`): `49,216 + 65 = 49,281` — about 11×
fewer than PO-LEU's 544,779.

## 7. Training loop

```python
torch.manual_seed(seed)
rng = np.random.default_rng(seed)

model     = STMLPHead(d_e, hidden, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
best_val, best_state, patience_left = float("inf"), None, patience

for epoch in range(n_epochs):
    model.train()
    perm = rng.permutation(N_tr)
    for start in range(0, N_tr, batch_size):
        idx = perm[start : start + batch_size]
        Eb, yb = E_train[idx], y_train[idx]                       # (B, J, d_e), (B,)
        wb = omega_train[idx] if omega_train is not None else None
        logits = model(Eb)                                        # (B, J)
        per_event = F.cross_entropy(logits, yb, reduction="none")
        loss = per_event.mean() if wb is None else \
               (wb * per_event).sum() / wb.sum().clamp_min(1e-12)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if E_val is not None:
        model.eval()
        with torch.no_grad():
            v_nll = F.cross_entropy(model(E_val), y_val).item()
        if v_nll < best_val - 1e-6:
            best_val, best_state = v_nll, {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0: break

if best_state is not None: model.load_state_dict(best_state)
```

Hyper-param defaults: `lr=1e-3`, `weight_decay=1e-4`, `dropout=0.1`,
`hidden=64`, `batch_size=64`, `n_epochs=100`, `patience=15`, `seed=7`.
Omega: if `batch.metadata[e]` carries `"omega"` it is collected and used
exactly as in `src.model.po_leu.cross_entropy_loss`; otherwise `omega=None`.

## 8. Class signatures

```python
# src/baselines/st_mlp_ablation.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np, torch
from torch import nn
from src.outcomes.encode import EncoderClient, encode_batch
from src.outcomes.cache import EmbeddingsCache
from .base import BaselineEventBatch, FittedBaseline


class STMLPHead(nn.Module):
    """Two-layer MLP applied per alternative."""
    def __init__(self, d_e: int, hidden: int, dropout: float) -> None: ...
    def forward(self, E2: torch.Tensor) -> torch.Tensor:
        """E2: (B, J, d_e) -> logits (B, J)."""


@dataclass
class STMLPFitted:
    name: str
    state_dict: dict
    d_e: int
    hidden: int
    dropout: float
    encoder: EncoderClient
    embeddings_cache: Optional[EmbeddingsCache]
    drop_is_repeat: bool
    n_params_total: int
    train_nll: float
    val_nll: float

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]: ...
    @property
    def n_params(self) -> int: ...
    @property
    def description(self) -> str: ...


class STMLPChoice:
    name: str = "ST+MLP"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        seed: int = 7,
        encoder: Optional[EncoderClient] = None,
        embeddings_cache: Optional[EmbeddingsCache] = None,
    ) -> None: ...

    def fit(self, train: BaselineEventBatch, val: BaselineEventBatch) -> STMLPFitted: ...
```

`score_events` re-encodes `batch.raw_events[*]["alt_texts"]`, reshapes to
`(N, J, d_e)`, runs one `eval()` forward pass, and returns `[logits[n]]`.
The shared harness handles softmax / top-1 / MRR / NLL.

**Raw-events dependency:** the baseline requires `batch.raw_events` with
`alt_texts`. If absent it raises `ValueError("STMLPChoice requires
raw_events[e]['alt_texts']; re-produce the batch with
records_to_baseline_batch.")`. The 6-column numeric
`base_features_list` is not used — this baseline embeds text, not
features.

## 9. Caching strategy

Two layers, in order:

1. **Project-global `EmbeddingsCache`** (SQLite, `src/outcomes/cache.py`):
   if the caller passes `embeddings_cache=<cache>`, all encode calls go
   through `encode_batch(..., cache=embeddings_cache)` which keys on
   `sha256(text || encoder.encoder_id)`. ST+MLP and PO-LEU share one
   cache file by design — they key on different strings (metadata
   sentences vs narratives), so no collision.
2. **Per-fit in-memory dict** (`dict[str, np.ndarray]`) on the
   `STMLPChoice` instance: built once from the training `alt_texts` and
   reused across `fit` → `score_events` on the same process. Avoids
   round-tripping popular duplicates through SQLite.

The global cache is authoritative; the local dict is a hot-path
optimisation.

## 10. Test strategy

File `tests/baselines/test_st_mlp_ablation.py`, `StubEncoder(d_e=64)` for
hermetic runs. Required tests:

| name | asserts |
|---|---|
| `test_render_alt_text_format` | All 7 fields present; missing keys fall back; no `None`/`nan` in output. |
| `test_render_alt_text_stable_across_calls` | Same dict → identical string (cache-key stability). |
| `test_score_events_shape_and_probabilities_sum_to_one_after_softmax` | Returns `List[np.ndarray]`, shapes `(J,)`, `softmax(·).sum() == 1` within 1e-6. |
| `test_n_params_matches_mlp_param_count` | `n_params == (d_e+1)*hidden + (hidden+1)`. |
| `test_deterministic_given_seed` | Two `fit` calls, same seed → `torch.allclose` state dicts. |
| `test_fit_learns_synthetic_signal` | Plant a keyword in chosen-alt titles; assert `train_nll < log(J) - 0.1` and `top-1 > 1/J + 0.1`. |
| `test_requires_raw_events` | `fit(batch_without_raw_events)` raises `ValueError`. |
| `test_missing_alt_texts_fields_graceful` | Records lacking `state`/`brand`/`popularity_rank` fit cleanly. |
| `test_cache_roundtrip_hits_global_embeddings_cache` | Second `fit` with the same `EmbeddingsCache` calls the encoder with strictly fewer texts (spy the client). |
| `test_is_repeat_leak_guard_drops_sentence` | Synthetic batch with 1-hot `is_repeat` pattern → `fitted.drop_is_repeat is True` and rendered strings omit the `is_repeat` sentence. |

## 11. Comparison note (docstring)

> **ST+MLP ablation — "PO-LEU minus narratives".** Embeds the 7-key alt
> metadata dict with the same frozen encoder PO-LEU uses
> (`all-mpnet-base-v2`) and trains a small MLP `(d_e → 64 → 1)`.
>
> **What this isolates.** PO-LEU's pipeline is (LLM narratives) →
> (encoder) → (PO-LEU heads). This baseline removes the first stage *and*
> replaces the third with a generic MLP. It answers: *how much of PO-LEU's
> win comes from the frozen encoder alone?*
>
> **What it does NOT isolate.** It does not separate narratives from
> heads. For the pure head ablation (narratives kept, heads replaced),
> see `src/model/ablations.py::variant_A6`. If ST+MLP's top-1 / NLL /
> MRR land within 1 SE of PO-LEU's, the LLM-narrative step is *not*
> adding signal beyond the encoder — report as a negative result.

## 12. Known failure modes

1. **Encoder domain mismatch.** `all-mpnet-base-v2` was trained on web
   text; product titles tokenize awkwardly and the 64-token truncation
   clips long titles. Shared with PO-LEU, so the comparison is fair.
2. **Popularity leak via `popularity_rank`.** `"popularity score N"`
   encodes the training count in-string; the MLP can regex-grep it via
   the encoder. PO-LEU sees the same string, so not a baseline advantage,
   but the comparison to `PopularityBaseline` can become near-tautological
   on ASIN-uniform events — flag in reports.
3. **`state` has no per-alt variance.** Constant within an event under
   softmax → no signal. Kept only for encoder-input parity with PO-LEU.
4. **`is_repeat` label leak.** `alt_text()` returns `is_repeat=False` for
   every negative (negatives come from the ASIN lookup, which has no
   per-customer routine). This makes `is_repeat` 1-hot on the chosen alt,
   trivially solvable by the MLP. **Guard:** at `fit` time, if every
   training event has exactly one `is_repeat=True`, set
   `drop_is_repeat=True` so `_render_alt_text` omits that sentence. This
   same hazard affects PO-LEU; `tests/data/test_alt_text_no_leak.py`
   cross-checks.
5. **Cache collision with PO-LEU.** Rendered sentence format differs
   from narrative format, so keys differ. If a future refactor makes
   them identical, add a `"[ALT]"` prefix to `_render_alt_text`.
6. **Torch-only runtime.** Inherits `DUET`'s "torch required" constraint;
   skip markers mirror `test_duet_ga.py` on torch-less CI.

---

*Sign-off checklist: (a) Variant B vs strict A given existing head
ablations; (b) `is_repeat` guard behaviour; (c) shared global cache.*
