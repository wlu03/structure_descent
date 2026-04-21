# PO-LEU — Per-Subagent Prompts

Each section below is the full prompt you hand to one subagent. They are ordered by
dispatch wave. Every subagent prompt is self-contained: it includes the spec excerpt,
the public API the subagent must expose, and the tests it must write.

---

## 01 · `config-agent` → `config.py`

You are `config-agent`. You own exactly one file: `config.py`.

**Role.** Central source of truth for every hyperparameter, shape, and string constant
referenced by any other module. No other module should hard-code any of these values.

**Spec section.** §0 (notation, default $J, K, M, d_e, p$); §Appendix B (defaults).

**Public API.**

```python
@dataclass(frozen=True)
class Shapes:
    J: int = 10          # alternatives per choice set
    K: int = 3           # outcomes per alternative
    M: int = 5           # attributes
    d_e: int = 768       # encoder dim
    p: int = 26          # effective z_d dim after one-hot

@dataclass(frozen=True)
class ModelConfig:
    head_hidden: int = 128
    weight_hidden: int = 32
    salience_hidden: int = 64
    tau: float = 1.0

@dataclass(frozen=True)
class TrainConfig:
    lr_init: float = 1e-3
    lr_final: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    grad_clip: float = 1.0
    lam_weight_l2: float = 1e-4
    lam_salience_entropy: float = 1e-3
    lam_monotonicity: float = 1e-3
    lam_diversity: float = 1e-4

@dataclass(frozen=True)
class GenConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 180
    prompt_version: str = "v1"
```

Also expose a single `DEFAULT_CONFIG` object that aggregates the four.

**Tests.** Assert `DEFAULT_CONFIG.shapes.K == 3`; assert the dataclasses are frozen
(mutation raises `FrozenInstanceError`).

**Constraint.** No imports other than stdlib `dataclasses`.

---

## 02 · `person-features-agent` → `data/person_features.py`

You own exactly one file: `data/person_features.py`.

**Role.** Build the person feature vector $z_d \in \mathbb{R}^{26}$ from raw customer
columns. Fit standardization stats on the training split only, then apply to val/test.

**Spec section.** §2.1.

**Public API.**

```python
class PersonFeatureBuilder:
    def fit(self, train_df: pd.DataFrame) -> "PersonFeatureBuilder": ...
    def transform(self, df: pd.DataFrame) -> np.ndarray:   # shape (N, 26)
        ...
    def save(self, path: str) -> None: ...   # persist μ, σ, one-hot vocab
    @classmethod
    def load(cls, path: str) -> "PersonFeatureBuilder": ...
```

Feature layout must match §2.1 exactly: 6 (age) + 5 (income) + 1 (household_size,
standardized) + 1 (has_kids) + 4 (city_size) + 1 (education, standardized) +
1 (health_rating, standardized) + 1 (risk_tolerance, standardized) +
1 (log(1 + purchase_frequency), standardized) + 1 (novelty_rate) = 26.

**Constraint.** `transform` must raise if called before `fit`. `fit` may only see
training-split rows; the orchestrator will pass a `split == "train"`-filtered df.

**Tests.** Fit on a synthetic 100-row df and assert output shape is `(100, 26)` and
`np.isfinite(output).all()`.

---

## 03 · `context-string-agent` → `data/context_string.py`

You own exactly one file: `data/context_string.py`.

**Role.** Build `c_d`, the 5-8-line human-readable context string handed to the LLM
generator. Paraphrase the raw columns; do not dump them.

**Spec section.** §2.2.

**Public API.**

```python
def build_context_string(row: pd.Series, now: datetime) -> str: ...
```

Output must be 5-8 lines, first-person-adjacent, no numeric feature values that could
leak test-time truth (e.g. no "will buy X next"), no column-dump phrasing like
"age_bucket=3". `now` is the event timestamp and is used only for the "Current time:"
line.

**Constraint.** Deterministic: same input row → same string, byte for byte.

**Tests.** Build for a fixed synthetic row; assert `5 <= string.count("\n") + 1 <= 8`
and that `"age_bucket"` and `"income_bucket"` never appear.

---

## 04 · `choice-sets-agent` → `data/choice_sets.py`

You own exactly one file: `data/choice_sets.py`.

**Role.** Assemble the per-event input tuple $(z_d, c_d, \{x_j\}, c^*_t)$. Downstream
modules add embeddings to this tuple after the outcome pipeline runs.

**Spec section.** §1.

**Public API.**

```python
@dataclass
class ChoiceEvent:
    customer_id: str
    t: int
    z_d: np.ndarray            # (26,)
    c_d: str
    alternatives: list[dict]   # length J; each dict has title, category, price, asin, ...
    chosen_index: int          # in [0, J)

def build_choice_events(
    purchases_df: pd.DataFrame,
    z_builder: PersonFeatureBuilder,
    split: str,
) -> list[ChoiceEvent]: ...
```

**Constraint.** The chosen alternative must be at some index in `alternatives`; assert
`0 <= chosen_index < len(alternatives)`. All alternatives must include `title`,
`category`, `price`, `asin`.

**Tests.** Build 5 synthetic events and assert every one satisfies the constraint.

---

## 05 · `outcome-cache-agent` → `outcomes/cache.py`

You own exactly one file: `outcomes/cache.py`.

**Role.** SQLite-backed KV store with two namespaces: `generation` (LLM outputs keyed
by customer+asin+seed+prompt_version) and `embedding` (encoder outputs keyed by
outcome string + encoder id). Bumping the prompt version invalidates `generation`
but not `embedding`.

**Spec section.** §3.4, §4.3, §15.

**Public API.**

```python
class Cache:
    def __init__(self, path: str): ...
    def get_generation(self, customer_id: str, asin: str, seed: int,
                       prompt_version: str) -> list[str] | None: ...
    def put_generation(self, customer_id: str, asin: str, seed: int,
                       prompt_version: str, outcomes: list[str],
                       metadata: dict) -> None: ...
    def get_embedding(self, outcome: str, encoder_id: str) -> np.ndarray | None: ...
    def put_embedding(self, outcome: str, encoder_id: str,
                      vec: np.ndarray) -> None: ...
    def stats(self) -> dict: ...    # hit/miss counts per namespace
```

**Constraint.** Threadsafe for reads. Writes serialize through a lock. Key hashing is
`sha256` on the canonical concatenation listed in §3.4 and §4.3.

**Tests.** Put + get round-trip for both namespaces; assert miss returns `None`.

---

## 06 · `outcome-gen-agent` → `outcomes/generate.py`

You own exactly one file: `outcomes/generate.py`.

**Role.** Given `(c_d, x_j)`, produce $K$ first-person outcome narratives by calling a
frozen LLM (real API or local). Cache aggressively. Never register parameters.

**Spec section.** §3.

**Public API.**

```python
def generate_outcomes(
    context: str,
    alternative: dict,
    K: int,
    seed: int,
    cache: Cache,
    config: GenConfig,
    llm_callable: Callable[[str, str], str] | None = None,
) -> list[str]: ...
```

If `llm_callable` is `None`, use a deterministic mock (documented in the file) so the
pipeline runs offline. The mock should still respect `seed` and produce $K$ distinct
short strings.

**Constraint.** On the real path: `temperature=0.8`, `top_p=0.95`, `max_tokens=180`.
Parse the completion by newlines; pad with "no additional consequence." if fewer than
$K$ non-empty lines are produced.

**Tests.** With the mock path, call twice with the same seed and assert identical
output; call with different seeds and assert at least one string differs.

---

## 07 · `diversity-agent` → `outcomes/diversity_filter.py`

You own exactly one file: `outcomes/diversity_filter.py`.

**Role.** Post-generation paraphrase filter: if any pair of outcomes inside one
alternative has cosine similarity > 0.9, regenerate (up to 2 retries) and then accept.

**Spec section.** §3.5.

**Public API.**

```python
def filter_and_regenerate(
    outcomes: list[str],
    regen_fn: Callable[[int], list[str]],   # takes seed, returns K outcomes
    encoder_fn: Callable[[list[str]], np.ndarray],
    threshold: float = 0.9,
    max_retries: int = 2,
) -> tuple[list[str], int]:   # returns (final outcomes, retries used)
    ...
```

**Constraint.** Must not modify `outcomes` in place. Retries advance the seed by +1 per
retry.

**Tests.** Construct a pathological outcome list where two strings are identical;
assert the filter either regenerates or returns the original after `max_retries`.

---

## 08 · `encoder-agent` → `outcomes/encode.py`

You own exactly one file: `outcomes/encode.py`.

**Role.** Frozen sentence encoder → $d_e = 768$-dim embeddings, L2-normalized.

**Spec section.** §4.

**Public API.**

```python
def encode_outcomes(
    strings: list[str],
    cache: Cache,
    encoder_id: str = "all-mpnet-base-v2",
    encoder_callable: Callable[[list[str]], np.ndarray] | None = None,
) -> np.ndarray:   # (len(strings), 768), L2-normalized
    ...
```

If `encoder_callable` is `None`, use a deterministic hash-based mock (SHA-256 of the
string seeded into a numpy RNG → 768-dim Gaussian → L2-normalize). This keeps the
pipeline offline-runnable.

**Constraint.** Every output row has `np.linalg.norm == 1 ± 1e-5`.

**Tests.** Encode `["a", "b", "a"]` with the mock; row 0 == row 2; assert norms == 1.

---

## 09 · `attribute-heads-agent` → `model/attribute_heads.py`

You own exactly one file: `model/attribute_heads.py`.

**Role.** $M$ independent small MLPs, each $\mathbb{R}^{768} \to \mathbb{R}$. Layer
sizes: 768 → 128 → 1 with ReLU. Xavier-uniform init for weights, zero init for biases.

**Spec section.** §5.

**Public API.**

```python
class AttributeHeads(nn.Module):
    def __init__(self, M: int = 5, d_e: int = 768, hidden: int = 128): ...
    def forward(self, E: Tensor) -> Tensor:
        """E: (B, J, K, d_e) → (B, J, K, M)"""
```

**Constraint.** Heads are person-independent (they do NOT take `z_d`). Exactly the
parameter count in §5.1: $98{,}433 \cdot M$.

**Tests.** Instantiate with defaults; assert
`sum(p.numel() for p in model.parameters()) == 98_433 * 5`. Feed a `(2, 10, 3, 768)`
tensor; assert output shape `(2, 10, 3, 5)`.

---

## 10 · `weight-net-agent` → `model/weight_net.py`

You own exactly one file: `model/weight_net.py`.

**Role.** MLP $\mathbb{R}^{26} \to \mathbb{R}^{5}$ followed by softmax. Output is the
person's attentional-budget allocation across the $M$ attributes, on the simplex.

**Spec section.** §6.

**Public API.**

```python
class WeightNet(nn.Module):
    def __init__(self, p: int = 26, M: int = 5, hidden: int = 32,
                 normalization: str = "softmax"): ...    # "softmax" | "softplus"
    def forward(self, z_d: Tensor) -> Tensor:
        """z_d: (B, p) → (B, M), rows sum to 1"""
```

**Constraint.** Output must always sum to 1 along the last dim (up to fp32 tolerance).
`normalization="softplus"` normalizes by the sum of softplus outputs (spec §6.2 ablation).

**Tests.** Forward `torch.randn(4, 26)`; assert `torch.allclose(out.sum(-1),
torch.ones(4), atol=1e-5)`.

---

## 11 · `salience-net-agent` → `model/salience_net.py`

You own exactly one file: `model/salience_net.py`.

**Role.** Per-outcome attention within an alternative. MLP on `[e_k ; z_d]` (794-dim
input) with softmax over the $K$ outcomes of each alternative.

**Spec section.** §7.

**Public API.**

```python
class SalienceNet(nn.Module):
    def __init__(self, d_e: int = 768, p: int = 26, hidden: int = 64,
                 mode: str = "learned"): ...     # "learned" | "uniform"
    def forward(self, E: Tensor, z_d: Tensor) -> Tensor:
        """E: (B, J, K, d_e), z_d: (B, p) → (B, J, K), softmax over K"""
```

**Constraint.** `mode="uniform"` returns $1/K$ everywhere (spec §7.3 ablation). In
learned mode, output rows must sum to 1 over $K$.

**Tests.** Learned mode: `forward(torch.randn(2,10,3,768), torch.randn(2,26))` →
`(2,10,3)` tensor whose sum over last dim is `ones(2,10)`. Uniform mode: every entry
equals `1/K`.

---

## 12 · `po-leu-agent` → `model/po_leu.py`

You own exactly one file: `model/po_leu.py`.

**Role.** End-to-end forward pass: attribute heads + weight net + salience net →
alternative values → softmax choice probabilities. This is the assembled model. No
training logic here; that's `train/loop.py`'s job.

**Spec section.** §8, §9.4, Appendix A.

**Public API.**

```python
class POLEU(nn.Module):
    def __init__(self, cfg: ModelConfig, shapes: Shapes,
                 weight_norm: str = "softmax", salience_mode: str = "learned"): ...
    def forward(self, z_d: Tensor, E: Tensor) -> dict:
        """
        z_d: (B, p), E: (B, J, K, d_e)
        Returns dict with:
          logits:      (B, J)        = V(a_j) / τ
          values:      (B, J)        = V(a_j)
          attr_scores: (B, J, K, M)  = u_m(e_k)
          weights:     (B, M)        = w_m(z_d)
          utilities:   (B, J, K)     = U_k^{(j)}
          salience:    (B, J, K)     = s_k^{(j)}
        """
```

**Constraint.** Must return every intermediate tensor, because `interpret.py` reads
them for the per-decision decomposition report (§12.2). The forward pass must match
Appendix A exactly.

**Tests.** Build default model, forward `(z_d=randn(4,26), E=randn(4,10,3,768))`;
assert every output shape matches the docstring. Assert the model has exactly
`544_075` trainable params when all sub-components use default sizes.

---

## 13 · `regularizers-agent` → `train/regularizers.py`

You own exactly one file: `train/regularizers.py`.

**Role.** The four regularizers from §9.2: weight-net L2, salience entropy (spread-
encouraging, i.e. negative of plain entropy as a *minimization* target),
monotonicity (optional), diversity (optional).

**Spec section.** §9.2.

**Public API.**

```python
def weight_l2(weight_net: nn.Module) -> Tensor: ...
def salience_entropy(S: Tensor) -> Tensor:
    """S: (B, J, K) softmaxed; return scalar term to be ADDED to loss
       (smaller → more uniform; sign chosen so that descending it drives spread)."""
def monotonicity_penalty(attr_scores: Tensor, prices: Tensor,
                         financial_index: int = 0) -> Tensor: ...
def diversity_penalty(E: Tensor) -> Tensor:
    """E: (B, J, K, d_e); penalize pairs of outcomes within each alternative
       with high cosine similarity."""
```

**Constraint.** Every function returns a scalar tensor with grad. When the relevant
tensors have no gradient (e.g. prices in the monotonicity penalty), return
`torch.zeros(())` cleanly.

**Tests.** `salience_entropy(torch.ones(2,10,3)/3)` should be a low-magnitude scalar
(uniform salience has maximal entropy, so the penalty at uniform is the lower bound).

---

## 14 · `train-loop-agent` → `train/loop.py`

You own exactly one file: `train/loop.py`.

**Role.** One epoch of training + one epoch of validation. Cross-entropy loss,
Adam optimizer, cosine LR schedule, gradient clipping, optional early stopping.

**Spec section.** §9.1, §9.3.

**Public API.**

```python
def train_one_epoch(
    model: POLEU,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler,
    cfg: TrainConfig,
    weights_per_event: Tensor | None = None,
) -> float:   # returns mean training loss
    ...

def validate(model: POLEU, loader: DataLoader) -> float:   # returns mean val NLL
    ...

def fit(
    model: POLEU,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    on_epoch_end: Callable | None = None,
) -> dict:   # returns best-val-NLL checkpoint state + training curves
    ...
```

**Constraint.** `fit` must respect `cfg.patience` for early stopping and always restore
the best-val-NLL weights at the end. Gradient clipping at $\|\cdot\|_2 \le 1.0$.

**Tests.** Fit for 2 epochs on a tiny synthetic dataset; assert loss decreases.

---

## 15 · `metrics-agent` → `eval/metrics.py`

You own exactly one file: `eval/metrics.py`.

**Role.** Top-1, Top-5, MRR, NLL, AIC, BIC. Stratified breakdowns by category,
repeat/novel, activity tertile, time-of-day, dominant attribute.

**Spec section.** §13.

**Public API.**

```python
def top1(logits: Tensor, targets: Tensor) -> float: ...
def topk(logits: Tensor, targets: Tensor, k: int = 5) -> float: ...
def mrr(logits: Tensor, targets: Tensor) -> float: ...
def nll(logits: Tensor, targets: Tensor) -> float: ...
def aic(k: int, n_train: int, nll_test: float) -> float: ...
def bic(k: int, n_train: int, nll_test: float) -> float: ...
def stratified(metric_fn: Callable, logits: Tensor, targets: Tensor,
               strata: np.ndarray) -> dict[str, float]: ...
```

**Constraint.** `logits` and `targets` are torch tensors; returns are plain floats.

**Tests.** For a 3-class problem where logits = one-hot of targets, assert
`top1 == 1.0` and `nll ≈ 0`.

---

## 16 · `interpret-agent` → `eval/interpret.py`

You own exactly one file: `eval/interpret.py`.

**Role.** The four interpretability protocols: attribute-head naming, per-decision
decomposition, dominant-attribute evaluation, counterfactual sensitivity.

**Spec section.** §12.

**Public API.**

```python
def top_outcomes_per_head(
    model: POLEU, outcomes_db: list[str],
    embeddings: Tensor, n: int = 100
) -> dict[int, list[tuple[str, float]]]: ...

def per_decision_report(
    model: POLEU, event: ChoiceEvent, E: Tensor, outcome_strings: list[list[str]]
) -> dict: ...

def dominant_attribute(model: POLEU, z_d: Tensor, E: Tensor,
                       chosen: Tensor) -> Tensor: ...

def counterfactual(model: POLEU, z_d: Tensor, E: Tensor,
                   perturb_fn: Callable[[Tensor], Tensor]) -> dict: ...
```

**Constraint.** No randomness; every function is deterministic given model + inputs.

**Tests.** `per_decision_report` returns a dict with keys `outcomes`,
`attr_scores`, `weights`, `salience`, `values`, `choice_prob`.