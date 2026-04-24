# Delphos Baseline (DQN utility-structure search)

**Status**: design — implementation deferred to a follow-up agent.
**Target paths**: `src/baselines/delphos.py`, `tests/baselines/test_delphos.py`.
**Reference**: Nova, Hess, van Cranenburgh, "Delphos: A reinforcement
learning framework for assisting discrete choice model specification,"
TORS 2025 (arXiv:2506.06410).

## 1. Method

Delphos is **not an LLM**. It is a Deep Q-Network (DQN) agent that learns
a policy over `(variable, transformation)` edits to a utility
specification. Each episode builds a specification by sampling
`add / change / terminate` actions with an epsilon-greedy policy; the
terminal state decodes to a `DSLStructure`; an inner-loop MNL fits
coefficients; the reward is a normalized function of AIC/BIC/LL (the
paper's default is `{AIC: 1.0}`). Rewards are distributed back to
transitions (exponential / linear / uniform modes) and pushed into a
replay buffer; a target network is updated every `target_update_freq`
episodes. Training stops on a rolling-window plateau rule.

An older faithful port already exists at
`/Users/wesleylu/Desktop/structure_descent/old_pipeline/src/baselines/delphos.py`
(1151 lines). It preserves Delphos's DQN machinery verbatim and swaps
the Apollo-R estimator for a Python hierarchical MNL. We vendor that
implementation with a minimal adaptation layer to hit the current
`BaselineEventBatch` contract.

## 2. Adapter decision: **Option B — share the 4-feature pool**

The PO-LEU adapter
(`/Users/wesleylu/Desktop/structure_descent/src/baselines/data_adapter.py:86-91`)
emits exactly four per-alt columns: `price, popularity_rank, log1p_price,
price_rank`. The six other columns that were originally produced by
`alt_text()` are per-alt-unsafe (leaky) and were deliberately pruned
(`data_adapter.py:26-66`). Recovering the old 12-term DSL pool for
Delphos would require either (a) re-deriving features from
`raw_events["alt_texts"]` via a Delphos-specific adapter, or (b)
re-introducing leaky columns for Delphos only.

**Recommendation: Option B.** Run Delphos against the same 4-feature
pool the rest of the suite sees. Justifications:

1. **Fair comparison.** The leaderboard's stated purpose is to compare
   baselines on identical inputs. PO-LEU, LASSO-MNL, DUET, ST-MLP all
   see 4 columns (+ nonlinear expansions). Letting Delphos see 12
   would attribute its performance to richer inputs, not its search
   algorithm.
2. **Leakage bar.** Six of the old columns (`recency`, `cat_affinity`,
   `time_match`, `rating_signal`, `delivery_speed`, `co_purchase`) are
   populated only for the chosen alt at this stage of the pipeline
   (`data_adapter.py:26-32`). Fabricating zeros for negatives would
   inflate Delphos's apparent top-1 with a label-indicator signal —
   exactly the bug we caught for `is_repeat` and `brand_known`.
3. **LOC cost.** A second adapter reusable only by Delphos adds
   ~300 LOC with no corresponding gain for other baselines.
4. **Published numbers are not the benchmark.** We are not trying to
   reproduce Nova et al.'s AIC on Swissmetro. We want a rigorous
   internal leaderboard, so the apples-to-apples comparison wins.

**Consequence:** expected absolute performance drops vs the paper
because the DSL has fewer atoms. This is documented in the leaderboard
readme next to the baseline's row.

With 4 variables and 3 transformations (`linear / log / box-cox`) plus
the ASC slot that our implementation drops
(`old_pipeline/src/baselines/delphos.py:759-760`), the search space
collapses from `12 × 3 ≈ 36` atomic specs to `4 × 3 = 12`. The DQN is
overkill on 12 atoms, but running Delphos is still informative: it
exercises the `add/change/terminate` policy and the AIC-regularized
reward.

## 3. Research answers

### 3.1 Inner-loop scope: flat MNL vs hierarchical

`old_pipeline/src/inner_loop.py:41-87` fits the hierarchical
`theta_g + theta_c + delta_i` decomposition; `fit_weights_no_hierarchy`
(`inner_loop.py:122-139`) is the flat-MNL ablation. **Use the flat MNL
for the port.** Pros: (a) param count collapses to `n_terms` (not
`n_terms * (1 + n_cats + n_custs)`), which both speeds up the inner loop
and keeps AIC/BIC comparable to LASSO-MNL; (b) the per-episode
inner-loop cost drops ~30×, making a 500-episode budget viable;
(c) zero customer-id / category ids appear in Delphos's own scoring
signal, so there's no concern about the DQN overfitting hierarchical
wiring. Cons: we can't learn per-category slopes, so the structure it
finds may underfit relative to the paper's published numbers — but that
is consistent with Option B and with how LASSO-MNL / DUET / ST-MLP are
currently evaluated.

### 3.2 DQN episode budget

The paper uses 50k episodes. The old port's default
(`old_pipeline/src/baselines/delphos.py:1025`) is 500. For the Option B
search space of 12 atoms and ≤4 active terms, a rough estimate of the
reachable-spec count is `sum(C(12,k) for k in 1..4) ≈ 794`. A
well-calibrated DQN should revisit specs and stabilize within a few
hundred episodes.

Recommended defaults:
- **20-customer pilot**: `n_episodes=300`, `early_stop_window=50`,
  `patience=10`. Small pilots have high seed variance; running 3 seeds
  and reporting the best-BIC final spec is cheaper than a single
  large-budget run.
- **50-customer paper run**: `n_episodes=1500`, `early_stop_window=100`,
  `patience=20`. Gives the plateau detector two full windows of
  post-`min_percentage` budget to trigger, and leaves headroom if the
  DQN needs longer to converge on the 4-variable pool.

### 3.3 State representation

The existing implementation encodes a partial specification as a
one-hot **integer vector**, not a learned embedding
(`old_pipeline/src/baselines/delphos.py:163-201`). `StateManager` lays
out one slot per `(variable, transformation_code)` pair; an ASC slot
leads the vector. For Option B with 4 variables and 4 transformation
codes (`none/linear/log/box-cox`) and no `specific` taste or covariates,
`get_state_length()` returns `1 + 4*4 = 17`. `encode_state_to_vector`
sets slot `1 + (var-1)*4 + trans_idx` to 1 when the spec has that
`(var, trans)` tuple.

The DQN itself is a plain MLP over that binary vector: input 17, hidden
`[128, 64]`, output `|action_space|` (one Q per action, then the illegal
ones are masked at decision time). No representation learning on the
DSL itself; the policy net learns Q-values directly over the one-hot.
The state→string encoder
(`old_pipeline/src/baselines/delphos.py:203-218`) keys the AIC cache so
specs are fit at most once.

### 3.4 Reward signal

The existing port implements the full Delphos reward machinery
(`old_pipeline/src/baselines/delphos.py:562-605`) with min-max
normalization against the running log of seen specs. For the new
implementation: **keep `reward_weights={"AIC": 1.0}`** — this is the
paper's default
(`old_pipeline/src/baselines/delphos.py:1044`) and the machinery is
already written. The NLL+penalty equivalent (`-NLL - k * n_params`)
would be simpler but would require re-tuning `k` to recover AIC-like
pressure; not worth the risk.

### 3.5 Compute

Per-episode cost is dominated by the inner-loop MNL fit. For the flat
MNL on 9k events, expected wall-clock per uncached estimation is 1-3 s
on one CPU core; cached (string-key) calls are ~50 µs. With a ~50%
cache-hit rate at 500 episodes, expect 250 cold fits × 2 s ≈
**8-10 minutes** total. The DQN gradient step is negligible
(~5 ms per episode).

- **20 customers × 500 episodes**: ~8 min single-threaded, peak RAM
  ~500 MB (feature-matrix cache).
- **50 customers × 1500 episodes**: ~45 min single-threaded, peak RAM
  ~1.5 GB.

No LLM calls, no GPU required.

## 4. Module skeleton

```python
# src/baselines/delphos.py
"""Delphos baseline: DQN over utility-structure edits.

Port of old_pipeline/src/baselines/delphos.py adapted to the 4-feature
data_adapter schema. See docs/llm_baselines/delphos_baseline.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaselineEventBatch, FittedBaseline

# Vendored from old_pipeline/src/dsl.py and old_pipeline/src/inner_loop.py.
# Copy the DSLTerm / DSLStructure / build_structure_features /
# compute_compound_feature definitions into src/baselines/_dsl.py and
# fit_weights_no_hierarchy into src/baselines/_inner_loop.py. The
# hierarchical fit is NOT ported.
from ._dsl import DSLStructure, DSLTerm, build_structure_features
from ._inner_loop import fit_weights_no_hierarchy

# DQN pieces (DQNetwork, ReplayBuffer, StateManager, DQNLearner) are
# copied verbatim from old_pipeline/src/baselines/delphos.py:84-734.
# They live in src/baselines/_delphos_dqn.py to keep this module readable.
from ._delphos_dqn import DQNLearner


class Delphos:
    """DQN-based utility-specification search over the shared 4-feature pool."""

    name = "Delphos"

    def __init__(
        self,
        n_episodes: int = 500,
        feature_names: Optional[Sequence[str]] = None,   # defaults to adapter's names
        hidden_layers: Tuple[int, ...] = (128, 64),
        gamma: float = 0.9,
        batch_size: int = 64,
        target_update_freq: int = 10,
        reward_weights: Optional[Dict[str, float]] = None,   # default {"AIC": 1.0}
        reward_distribution_mode: str = "exponential",
        epsilon_min: float = 0.01,
        min_percentage: float = 0.1,
        early_stop_window: int = 50,
        patience: int = 10,
        sigma: float = 10.0,                             # flat-MNL L2 prior
        seed: int = 0,
    ) -> None: ...

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> "DelphosFitted": ...


@dataclass
class DelphosFitted:
    """Fitted Delphos model conforming to FittedBaseline."""

    name: str
    best_structure: DSLStructure
    best_weights: np.ndarray          # shape (n_terms,) — flat MNL
    base_feature_names: List[str]
    n_episodes_run: int
    candidates_evaluated: int
    train_nll: float
    val_nll: float

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]: ...

    @property
    def n_params(self) -> int: ...        # = len(best_structure.terms)

    @property
    def description(self) -> str: ...
```

### 4.1 Training-loop pseudocode

```
state_space_params = {
    "num_vars": len(feature_names) + 1,      # +1 for the ASC slot
    "transformations": ["linear", "log", "box-cox"],
    "taste": ["generic"],                    # non-specific branch
    "covariates": [],
}

def estimator(state):
    # 1. decode Delphos state -> DSLStructure (var=0 is ASC, dropped)
    #    Matches old_pipeline/src/baselines/delphos.py:742-773 with
    #    feature_names replacing ALL_TERMS.
    structure = _state_to_structure(state, feature_names)
    # 2. build (n_events, J, n_terms) features via build_structure_features
    feats_list = [build_structure_features(structure, X, feature_names)
                  for X in train.base_features_list]
    # 3. flat MNL fit
    try:
        w = fit_weights_no_hierarchy(structure, feats_list,
                                     train.chosen_indices, sigma=sigma)
    except Exception:
        return pd.DataFrame([{"successfulEstimation": False, ...}])
    # 4. compute LL0, LL, LLout, AIC, BIC
    # 5. return a single-row DataFrame matching _REWARD_COLUMNS
    return pd.DataFrame([row])

learner = DQNLearner(state_space_params, n_episodes, estimator, ...)
for episode in range(n_episodes):
    state = []
    while not done:
        a = epsilon_greedy(policy_net(encode(state)), valid_actions(state))
        state, done = apply_action(state, a)
    reward, outcome_row = estimator(state)   # cached by encoded-state key
    update_candidate_tracker(state, outcome_row)
    distribute_reward(episode_steps, reward, mode)   # exponential by default
    experience_replay_step()
    if episode % target_update_freq == 0:
        target_net <- policy_net
    epsilon = max(epsilon_min, epsilon - 1/n_episodes)
    if past min_percentage AND rolling-window plateau: break
return best_state_from_candidate_tracker()
```

### 4.2 Scoring pseudocode

```
def score_events(self, batch):
    names = batch.base_feature_names
    scores = []
    for feats in batch.base_features_list:                    # (J, n_terms_base)
        struct_feats = build_structure_features(
            self.best_structure, feats, names)               # (J, k)
        scores.append(struct_feats @ self.best_weights)      # (J,)
    return scores
```

The scoring path ignores `customer_ids` / `categories` because the flat
MNL has no per-customer branch. This matches the adapter's
documentation that baselines share one coefficient vector.

## 5. Test strategy

`tests/baselines/test_delphos.py` covers:

1. **Protocol conformance** — `isinstance(fitted, FittedBaseline)` after
   fitting on `make_synthetic_batch(n_events=200)` (the synthetic helper
   at `src/baselines/_synthetic.py:37`).
2. **Shapes** — `score_events(test)` returns `len(test.n_events)`
   arrays each of shape `(J,)` with finite values.
3. **Beats chance on synthetic** — with `signal_strength=2.0`, top-1
   accuracy on a held-out synthetic batch exceeds `1/J + 0.05` after
   150 episodes. Guards against a silent no-op.
4. **Monotonic AIC tracking** — the best-candidate AIC recorded in
   `learner.best_candidates["AIC"]` at end-of-training is <= any AIC
   logged during training. Guards against the candidate tracker
   regressing.
5. **Cache behavior** — after fit, `learner.cache_misses` <= number of
   unique encoded states visited, and `cache_hits + cache_misses` equals
   the total number of estimator invocations. Detects cache-key drift.
6. **Determinism on seed** — two `Delphos(seed=42)` fits on the same
   synthetic batch yield identical `best_structure.term_names` and
   `best_weights` within 1e-6. Catches torch/numpy seeding regressions.
7. **Early stopping fires** — with `n_episodes=2000` on a 50-event toy
   batch where the search space has ~4 reachable specs, `learner.train()`
   returns fewer than 2000 episodes. Guards against plateau-detector
   regressions.
8. **Empty-fallback structure** — if the DQN terminates immediately on
   episode 0 (forced via `epsilon=0` and a patched policy that always
   emits `terminate`), `DelphosFitted.best_structure` falls back to a
   non-empty structure so `score_events` never divides by zero.
9. **Adapter round-trip** — fit on a batch produced by
   `records_to_baseline_batch` on 3 synthetic PO-LEU records and assert
   that `fitted.base_feature_names == BUILTIN_FEATURE_NAMES`.
10. **NaN / degenerate fit handled** — seed a batch where one feature
    is all-zero; assert the estimator returns `successfulEstimation=True`
    with finite AIC (not `-inf`) and that fitting completes.
11. **`n_params` matches the structure** — after fit,
    `fitted.n_params == len(fitted.best_structure.terms)`.
12. **Registry wiring** — import
    `src.baselines.run_all.BASELINE_REGISTRY`, assert the entry
    `("Delphos", "src.baselines.delphos", "Delphos")` is present once.

## 6. Registry entry

Append one line to `_build_registry()` in
`/Users/wesleylu/Desktop/structure_descent/src/baselines/run_all.py:133-147`,
next to `DUET`:

```python
("Delphos",          "src.baselines.delphos",       "Delphos"),
```

No changes to `LLM_CLIENT_FACTORIES` (Delphos is LLM-free). The
`cls(**kwargs)` path at
`src/baselines/run_all.py:249` accepts zero kwargs because all Delphos
constructor args have defaults.

## 7. Cost and wall-clock estimates

| Scale             | n_events (train) | n_episodes | cold fits | fit wall-clock | peak RAM |
|-------------------|-----------------:|-----------:|----------:|---------------:|---------:|
| 20-customer pilot |           ~3,000 |        300 |      ~150 |         3-5 min |   500 MB |
| 50-customer paper |           ~9,000 |       1500 |      ~400 |        30-60 min |   1.5 GB |

Numbers assume 1 CPU core, no GPU, flat MNL inner loop with L-BFGS-B
maxiter=500. Scaling is dominated by `n_events × n_terms²` in the
inner-loop objective evaluation; doubling either linearly doubles
cold-fit time. No API costs.

## 8. Known limitations

1. **Seed variance.** Small-budget DQN runs exhibit large variance in
   the final spec; run `n_seeds >= 3` and report the best-BIC spec.
2. **Shallow search space.** With 4 variables, Delphos has 12 atoms and
   ~800 reachable specs; this understates the method's value vs a
   larger DSL. Documented as the Option B trade-off.
3. **No interactions explored.** The Delphos action space only emits
   `add(var, trans) / change(var, trans) / terminate`. Combinators
   (`interaction`, `ratio`, `split_by`) from
   `/Users/wesleylu/Desktop/structure_descent/old_pipeline/src/dsl.py:99-112`
   are not reachable. A later extension could add `interact(var_a,
   var_b)` actions, but that's a second project.
4. **DQN hyperparameter sensitivity.** `gamma`, `hidden_layers`,
   `target_update_freq`, `epsilon_min` are taken from the paper and
   not re-tuned. Empirically the method is relatively robust, but the
   20-customer pilot should spot-check one alternate `hidden_layers`
   setting.
5. **Cache is not persisted.** The AIC-cache lives in memory per fit;
   re-running a second seed re-fits all previously seen specs. If total
   wall-clock becomes a problem at the paper scale, we can pickle the
   cache keyed on `(batch_hash, structure_string)` — out of scope for
   the initial port.
6. **No behavioral sign check.** Delphos itself does not implement
   Eq. 17; neither do we. If a term comes back with an
   economically-implausible sign (e.g. positive `price` coefficient),
   the baseline still reports it. Consistent with the paper, documented
   for reviewers.
