# PO-LEU: Implementation Specification

**Perceived-Outcome, LLM-generated, Embedding-based Utility**
*A per-component implementation reference for the attribute-decomposed narrative choice model.*

---

## 0. Scope and Conventions

This document specifies what a developer needs to build PO-LEU end-to-end. It assumes the v2.0 pipeline (data cleaning, survey join, state features, temporal split, choice-set construction, leverage-score subsampling, evaluation harness) is already in place; those components are unchanged and are referenced here only at their interfaces.

**Notation used throughout.**

| Symbol | Meaning | Shape / type |
|---|---|---|
| $d$ | decision-maker (customer) index | scalar |
| $t$ | choice event index for a customer | scalar |
| $j \in \{0, \dots, J{-}1\}$ | alternative index within a choice set | $J = 10$ |
| $k \in \{0, \dots, K{-}1\}$ | outcome index within an alternative | $K = 3$ (default) |
| $m \in \{1, \dots, M\}$ | attribute index | $M = 5$ (default) |
| $z_d$ | person feature vector | $\mathbb{R}^{p}$, $p = 10$ |
| $c_d$ | person context string (LLM-readable) | string |
| $x_j$ | alternative attributes (price, category, title, …) | struct |
| $o_k^{(j)}$ | generated outcome narrative for $(d, j, k)$ | string |
| $e_k^{(j)}$ | encoded outcome embedding | $\mathbb{R}^{d_e}$, $d_e = 768$ |
| $u_m(e)$ | attribute head $m$ | scalar |
| $w_m(z_d)$ | attribute importance weight | scalar, $\sum_m w_m = 1$ |
| $s_k^{(j)}$ | outcome salience within alternative $j$ | scalar, $\sum_k s_k^{(j)} = 1$ |
| $U_k^{(j)}$ | outcome utility | scalar |
| $V(a_j)$ | alternative value | scalar |
| $c^*_t$ | index of the chosen alternative in event $t$ | int in $\{0, \dots, J{-}1\}$ |

Batch dimension $B$ is prepended to all tensors in the training code. Unless stated otherwise, all neural modules use ReLU activations, Xavier-uniform init for linear weights, and zero init for biases.

---

## 1. Input Tuple per Choice Event

A single training/evaluation instance is the tuple

$$\big(z_d,\ c_d,\ \{x_j\}_{j=0}^{J-1},\ \{e_k^{(j)}\}_{j,k},\ c^*_t\big).$$

The first four fields are produced by the data-prep pipeline; $c^*_t$ is the ground-truth chosen index. Downstream networks consume only $z_d$ and the embedding tensor $E \in \mathbb{R}^{B \times J \times K \times d_e}$. Context $c_d$ is consumed only by the outcome generator; alternative attributes $x_j$ are consumed only by the outcome generator.

---

## 2. Person Feature Vector $z_d$ and Context String $c_d$

### 2.1 $z_d$ (for the weight net and salience net)

$p = 10$ features, all derived from columns available at time $t$ with no future leakage:

| # | Feature | Type | Encoding |
|---|---|---|---|
| 1 | `age_bucket` | categorical (6 bins) | one-hot (6 dims) |
| 2 | `income_bucket` | categorical (5 bins) | one-hot (5 dims) |
| 3 | `household_size` | int | standardized |
| 4 | `has_kids` | binary | 0/1 |
| 5 | `city_size` | categorical (4 bins) | one-hot (4 dims) |
| 6 | `education` | ordinal (5 levels) | standardized scalar |
| 7 | `health_rating` | ordinal (5 levels) | standardized scalar |
| 8 | `risk_tolerance` | derived scalar | standardized |
| 9 | `purchase_frequency` | continuous | $\log(1 + \text{prior events})$, then standardized |
| 10 | `novelty_rate` | continuous ∈ [0,1] | rate of `novelty=1` over prior events |

After encoding, the actual dimensionality is $p = 26$ (6 + 5 + 1 + 1 + 4 + 1 + 1 + 1 + 1 + 1). Report $p$ in configs but keep the logical "10 features" language in the paper.

**Standardization.** Fit $\mu, \sigma$ on the training split only; apply to val/test.

### 2.2 $c_d$ (for the LLM generator prompt)

Plain text, 5–8 short lines, built from the same columns as $z_d$ but rendered humanly. Template:

```
Person profile
- Age: mid-30s; household of 4 (two children, ages 6 and 9).
- Income: about $55k/year.
- Lives in a large U.S. city; college-educated.
- Self-reports good health; tends to be cautious with risk.
- Buys on Amazon roughly twice per week; a third of orders are first-time products.
Recent purchases (last 30 days): kids' backpacks, a coffee maker, batteries.
Current time: Tuesday evening, late April.
```

Two strict rules:
1. Never include numeric feature values that would leak test-time truth (e.g. "will buy headphones next").
2. Paraphrase rather than dump columns; column dumps make the LLM generate generic outcomes.

---

## 3. Outcome Generation (Component 1 of the LLM pipeline)

### 3.1 Role

Given $(c_d, x_j)$, produce $K$ first-person outcome narratives the person would imagine experiencing if they chose alternative $j$. Narratives should be **consequential, not descriptive** — not "this product is X" but "if I bought this, Y would happen."

### 3.2 Prompt spec

Fixed system prompt + dynamic user block.

**System prompt (verbatim, frozen):**

> You generate short first-person outcome narratives for a decision-maker considering an alternative. Produce exactly K sentences, each 10–25 words, each describing a *different type* of consequence: financial, health/physical, convenience/time, emotional/identity, social/relational. Sentences are in the first person, present or near-future tense, and grounded in the person's specific context. Do not describe the product. Do not hedge ("might", "could" are fine; "I think" is not). Do not number the sentences; separate them with newlines.

**User block template:**

```
CONTEXT:
{c_d}

ALTERNATIVE:
- Name/title: {x_j.title}
- Category: {x_j.category}
- Price: ${x_j.price}
- Popularity: {x_j.popularity_rank}
{optional domain-specific fields}

Generate K={K} outcome sentences.
```

### 3.3 Decoding settings

- Model: frozen; recommended default is a ~7B–70B instruction-tuned model accessible by API or local inference.
- Temperature: 0.8.
- Top-p: 0.95.
- Max tokens: 180 (accommodates $K=3$ at ~25 words + newlines).
- Seed: fixed per (customer, asin) pair (see caching).
- Single call per (customer, asin); parse the completion by splitting on newlines, stripping whitespace, keeping the first $K$ non-empty lines. If fewer than $K$ are produced, pad with a single-token sentinel outcome ("no additional consequence.") and log the event.

### 3.4 Caching

Cache key: `sha256(customer_id || asin || seed || prompt_version)`. Value: the $K$ strings plus generation metadata (temperature, model id, finish reason). Cache format: SQLite (single file, fast random access) or Parquet + RocksDB. The pipeline never calls the LLM at inference time once the cache is warm.

### 3.5 Diversity check (optional but recommended)

Post-hoc filter: discard and regenerate any $o_k$ that has cosine similarity $> 0.9$ with any other $o_{k'}$ for $k \ne k'$ in the same alternative. At most 2 retries; after that, accept. This prevents the salience network from being trained on paraphrase sets.

---

## 4. Outcome Encoding (Component 2 of the LLM pipeline)

### 4.1 Encoder choice

Frozen sentence encoder producing $d_e = 768$-dim vectors. Default: a modern sentence-transformers model (e.g. `sentence-transformers/all-mpnet-base-v2` or equivalent; MTEB scores are fine for this purpose). Alternatives for ablation: a 1024-dim larger encoder; a multilingual encoder. No fine-tuning in the main experiments.

### 4.2 Encoding procedure

1. Collect all $J \cdot K$ strings for a batch (or the whole dataset, offline).
2. Batched forward pass, max token length 64 (outcomes are short).
3. Mean-pool over token embeddings (or use the encoder's recommended pooling).
4. $\ell_2$-normalize each embedding to unit norm. Normalization is on by default — this stabilizes the scale entering the attribute heads.
5. Reshape to $E \in \mathbb{R}^{J \times K \times d_e}$ and serialize.

### 4.3 Caching

Cache key: `sha256(outcome_string || encoder_id)`. Value: the 768-dim vector. Independent of the generation cache — regenerating outcomes triggers re-encoding only for strings that actually changed.

---

## 5. Attribute Heads $u_m(e_k)$

### 5.1 Specification

$M$ independent small MLPs, one per attribute. Each maps $\mathbb{R}^{d_e} \to \mathbb{R}$.

| Layer | Input dim | Output dim | Activation |
|---|---|---|---|
| Linear 1 | 768 | 128 | ReLU |
| Linear 2 | 128 | 1 | — (linear) |

Parameter count per head: $768 \cdot 128 + 128 + 128 \cdot 1 + 1 = 98{,}433$. Total across $M=5$ heads: 492,165.

### 5.2 Attribute naming (for $M=5$)

1. **Financial** — monetary costs and savings
2. **Health** — physical health, access to care
3. **Convenience** — time, effort, logistics
4. **Emotional** — affect, stress, peace of mind, identity
5. **Social** — relationships, status, family

These names are fixed a priori. Heads are **not** told which attribute they are; the names are assigned post-hoc by inspecting what each head scores high (§12). The ordering is by convention.

### 5.3 Person-independence

By default, $u_m$ is **not** conditioned on $z_d$. Rationale: "what this outcome is about on dimension $m$" should be a property of the outcome, not the person. Person-dependent heads $u_m(e_k, z_d)$ are reported as an ablation (§11).

### 5.4 Output

For a batch of shape $(B, J, K, d_e)$, the attribute-head stack returns a tensor of shape $(B, J, K, M)$: the attribute score of each outcome.

---

## 6. Weight Network $w_m(z_d)$

### 6.1 Specification

One small MLP: $\mathbb{R}^{p} \to \mathbb{R}^{M}$, followed by a softmax.

| Layer | Input dim | Output dim | Activation |
|---|---|---|---|
| Linear 1 | 26 | 32 | ReLU |
| Linear 2 | 32 | 5 | — |
| Softmax | 5 | 5 | over $M$ |

Parameter count: $26 \cdot 32 + 32 + 32 \cdot 5 + 5 = 1{,}029$.

### 6.2 Why softmax

Interpretability: $w_m(z_d)$ is read as the **share of attentional budget** person $d$ allocates to attribute $m$. Weights lie on the $M$-simplex. Ablation: softplus normalization ($w_m = \text{softplus}(\text{raw}_m) / \sum_{m'} \text{softplus}(\text{raw}_{m'})$) permits unnormalized magnitudes; reported in §11.

### 6.3 Output

For a batch of shape $(B, p)$, returns $w \in \mathbb{R}^{B \times M}$. Broadcast to $(B, J, K, M)$ when combined with attribute scores.

### 6.4 Combination

$$U_k^{(j)} = \sum_{m=1}^{M} w_m(z_d) \cdot u_m(e_k^{(j)})$$

produces a tensor of shape $(B, J, K)$.

---

## 7. Salience Network $s_k^{(j)}(e_k, z_d)$

### 7.1 Specification

MLP taking concatenated $(e_k, z_d)$ and returning a scalar per outcome, softmaxed over $k$ within each alternative.

| Layer | Input dim | Output dim | Activation |
|---|---|---|---|
| Linear 1 | 768 + 26 = 794 | 64 | ReLU |
| Linear 2 | 64 | 1 | — |
| Softmax | $K$ | $K$ | over $k$ within each $(b, j)$ |

Parameter count: $794 \cdot 64 + 64 + 64 \cdot 1 + 1 = 50{,}881$.

### 7.2 Semantics

$s_k^{(j)}$ is **mental prominence**, not probability. It says "among the $K$ imagined outcomes of this alternative, which ones dominate this person's thinking." Interpretability claim: for risk-averse individuals, catastrophic outcomes should receive disproportionate salience relative to their generator-implied probability.

### 7.3 Ablation: uniform salience

Set $s_k^{(j)} = 1/K$ for all $k, j$. Reports predictive loss when the model cannot allocate attention. Required to distinguish "PO-LEU is better because of outcome narratives" from "PO-LEU is better because of personalized attention over narratives."

### 7.4 Output

$(B, J, K)$ tensor, with `softmax(raw, dim=-1)` applied over $K$.

---

## 8. Value and Choice

### 8.1 Alternative value

$$V(a_j) = \sum_{k=1}^{K} s_k^{(j)}(e_k^{(j)}, z_d) \cdot U_k^{(j)}.$$

Implementation: elementwise multiply $(B, J, K)$ tensors, sum over last dim, result $(B, J)$.

### 8.2 Choice probability

$$P(a_j \mid \text{context}) = \frac{\exp(V(a_j) / \tau)}{\sum_{j'=0}^{J-1} \exp(V(a_{j'}) / \tau)}.$$

Default $\tau = 1.0$. $\tau$ is **not** trained; it is a hyperparameter. A learned $\tau$ is an ablation.

---

## 9. Training

### 9.1 Loss

Per-event cross-entropy:

$$\ell_t = -\log P(a_{c^*_t} \mid \text{context}_t) = -\log \frac{\exp(V(a_{c^*_t})/\tau)}{\sum_{j} \exp(V(a_j)/\tau)}.$$

Batch loss with optional importance weights $\omega_t$ from leverage-score subsampling (see Appendix C for the construction of $\omega_t$):

$$\mathcal{L}_{\text{data}} = \frac{1}{\sum_{t \in B} \omega_t} \sum_{t \in B} \omega_t \cdot \ell_t.$$

If subsampling is off, $\omega_t = 1$ for all $t$.

### 9.2 Regularizers

Four, with small coefficients. Each has a justified purpose and a principled default.

| Term | Formula | Default $\lambda$ | Purpose |
|---|---|---|---|
| Weight-net L2 | $\|\phi_w\|_2^2$ | $10^{-4}$ | Prevent the weight net from overfitting to rare demographic combinations. |
| Salience entropy | $-\mathbb{E}_{b,j}\!\left[\sum_k s_k^{(j)} \log s_k^{(j)}\right]$ | $10^{-3}$ (*minimize negative entropy*, i.e. encourage spread) | Prevent salience collapse onto a single outcome (degenerate solution). |
| Monotonicity (optional) | $\sum_m \text{mean}_{b,j,k}\!\left[\max(0, -\partial_{p_j} u_m(e_k))\right]^2$ | $10^{-3}$ | Price-monotonicity: financial head should be non-increasing in price. Applied only if domain prior holds. |
| Diversity (optional) | $\text{mean}_{b,j}\!\left[\max_{k\ne k'} \cos(e_k, e_{k'})\right]$ | $10^{-4}$ | Penalize near-duplicate outcomes from the generator that slipped past §3.5. |

All $\lambda$ values are tuned once on the validation set; they are not per-experiment knobs.

Attribute heads and salience network have **no direct regularization**: their capacity is already small and the softmax constraints indirectly cap their influence.

### 9.3 Optimizer and schedule

- Optimizer: Adam, $\beta_1 = 0.9$, $\beta_2 = 0.999$.
- Initial LR: $10^{-3}$.
- Schedule: cosine annealing over the full training horizon to $10^{-4}$.
- Batch size: 128 choice events.
- Gradient clipping: $\ell_2$ norm $\le 1.0$.
- Max epochs: 30.
- Early stopping: val NLL, patience 5 epochs.
- Seed: fixed per experiment; all reported numbers averaged over 3 seeds with std reported.

### 9.4 Forward pass (reference sequence)

1. Load batch: $(z_d, E, \omega, c^*)$ with $E \in \mathbb{R}^{B \times J \times K \times d_e}$.
2. Compute attribute scores $A \in \mathbb{R}^{B \times J \times K \times M}$ by applying each $u_m$ to $E$.
3. Compute weights $w \in \mathbb{R}^{B \times M}$ from $z_d$ via the weight net + softmax.
4. Broadcast $w$ to $(B, 1, 1, M)$ and compute $U = (A \cdot w).\text{sum}(\text{dim}=-1) \in \mathbb{R}^{B \times J \times K}$.
5. Compute salience $S \in \mathbb{R}^{B \times J \times K}$ from $(E, z_d)$, softmaxed over $K$.
6. Compute $V = (S \cdot U).\text{sum}(\text{dim}=-1) \in \mathbb{R}^{B \times J}$.
7. Compute $\ell_t = \text{cross\_entropy}(V / \tau, c^*)$.
8. Add regularizers; take a step.

Total trainable parameter count (defaults: $M=5$, $p=26$, $d_e=768$, all hidden sizes as above): **544,075** — 492,165 (heads) + 1,029 (weight net) + 50,881 (salience net). This is the $k$ value used in AIC/BIC.

### 9.5 What gets frozen

- LLM generator: frozen. Never trained.
- Sentence encoder: frozen. Never trained.
- Temperature $\tau$: frozen.

Only the three small networks are trained. Deliberate design choice: keeps the model cheap, makes the LLM-side pipeline reproducible, and means a new dataset only requires re-caching outcomes + re-training ~0.5M parameters.

---

## 10. Data Flow Diagram (Implementation View)

```
              ┌──────────────────────┐
              │ Survey + Purchase    │
              │       Data           │
              └──────────┬───────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             │             ▼
     z_d ∈ ℝ^26          │       x_j (per alternative)
           │             │             │
           │             ▼             │
           │         c_d (string)      │
           │             │             │
           │             └──────┬──────┘
           │                    ▼
           │           ┌──────────────────────┐
           │           │ LLM Generator        │
           │           │ → K narratives       │
           │           │ (frozen, cached)     │
           │           └──────────┬───────────┘
           │                      │
           │                      ▼
           │           ┌──────────────────────┐
           │           │ Encoder              │
           │           │ → e_k ∈ ℝ^768        │
           │           │ (frozen)             │
           │           └──────────┬───────────┘
           │                      │
           │                      ▼
           │              E ∈ ℝ^(B×J×K×768)
           │                      │
           │          ┌───────────┴───────────┐
           │          ▼                       ▼
  ┌──────────────────────┐       ┌──────────────────────────┐
  │ weight_net           │       │ attribute_heads          │
  │ z_d → w ∈ Δ^M        │       │ (×M, shared across       │
  └──────────┬───────────┘       │  people)                 │
             │                   │ E → H ∈ ℝ^(B×J×K×M)      │
             │                   │ where H_{jkm} = u_m(e_jk)│
             │                   └──────────┬──────────────┘
             │                              │
             └───────────────┬──────────────┘
                             ▼
               U_jk = Σ_m w_m · H_jkm
                             │
                             ▼
                  ┌──────────────────────────┐
                  │ salience_net             │
                  │ (person-specific         │
                  │  attention from z_d)     │
                  │ → S ∈ ℝ^(B×J×K)          │
                  └──────────┬───────────────┘
                             │
                             ▼
                 V(a_j) = Σ_k s_jk · U_jk
                             │
                             ▼
                       softmax over J
                             │
                             ▼
                  cross-entropy vs. chosen c*
```

---

## 11. Ablations (What the Paper Must Report)

Eight variants, each training-expensive but cache-reusable. Run 3 seeds each.

| # | Name | What changes | Hypothesis |
|---|---|---|---|
| A0 | **PO-LEU (default)** | $M=5$, softmax weights, shared heads, salience on | Baseline to beat. |
| A1 | $M=3$ | Attributes: financial, health, other | Interpretability gain, predictive loss small. |
| A2 | $M=10$ | Latent factors, un-named | Fit improves, interpretability degrades. |
| A3 | $M=20$ | Latent factors | Marginal fit gain vs A2. |
| A4 | Softplus weights | Un-normalized importance | Interpretation harder; fit neutral. |
| A5 | Person-dependent heads | $u_m(e_k, z_d)$ | Fit improves modestly; heads lose universality. |
| A6 | Uniform salience | $s_k = 1/K$ | Fit drops sharply when outcomes are heterogeneous. |
| A7 | Concatenation utility | $U_k = h_\psi([e_k; z_d])$, no decomposition | Fit similar; interpretability lost — **central ablation**. |
| A8 | FiLM utility | $U_k = h_\psi(e_k; \theta_d)$, $\theta_d = g(z_d)$ | Fit between A7 and A0; modulation params harder to interpret than $(w, u_m)$. |

A7 and A8 are the critical comparisons: they demonstrate that the attribute decomposition buys interpretability without sacrificing predictive accuracy. If A7 wins on NLL, the paper's interpretability claim needs to be downweighted; if A0 matches or beats A7, the paper has both halves of its pitch.

**Secondary ablations** (single seed each):

- **Encoder swap.** all-mpnet-base-v2 vs. larger 1024-dim encoder. Expect small sensitivity.
- **Generator swap.** Two different LLMs of similar size. Bounds the "generator quality" variance in the method.
- **K = 1, 3, 5, 7.** The paper's default $K=3$ is justified only if $K=5$ doesn't help meaningfully.
- **Temperature $\tau \in \{0.5, 1.0, 2.0\}$.** Softmax temperature sensitivity.
- **No subsampling.** Full-data training to verify the leverage-score subsample isn't distorting results.

---

## 12. Interpretability Protocol

This is what the paper uses to defend "interpretable" in a non-hand-wavy way.

### 12.1 Attribute-head naming

For the trained model, collect the top-100 outcome strings (across all events) by $u_m(e_k)$ for each $m$. Inspect manually. Each head should yield a coherent semantic cluster; report the cluster + a proposed name. If two heads cluster around the same concept, report it — that's evidence the decomposition is weaker than claimed.

### 12.2 Per-decision decomposition

For any held-out event, produce a three-panel report:
1. The $K$ generated outcomes (strings) for each alternative.
2. The $(K \times M)$ matrix of attribute scores per alternative.
3. The $(M)$ person weights and $(K)$ salience weights per alternative.
4. The $V(a_j)$ values and the choice probability.

Every number in the probability is traceable to a named dimension and a named narrative.

### 12.3 Dominant-attribute evaluation

For each test event, compute the attribute $m^* = \arg\max_m w_m \cdot |u_m(e_{c^*})|$ — the attribute that most determines the model's prediction. Bucket test events by $m^*$ and report top-1 accuracy within each bucket. This probes whether the decomposition's claimed drivers are the same as the model's actual drivers.

### 12.4 Counterfactual sensitivity

Pick a held-out event. Perturb one component of $z_d$ (e.g., add `+1` child). Re-score without regenerating outcomes. Report the change in $w(z_d)$, the change in $s_k$ for each outcome, and the change in $P(a_{c^*})$. Narrate three such perturbations in the paper; they are the most concrete evidence the model is using person features the way the decomposition claims.

---

## 13. Evaluation

Unchanged from the v2.0 harness. Reported on the held-out test split.

- **Top-1 accuracy**: $\mathbb{1}[\text{rank}(c^*) = 0]$, averaged.
- **Top-5 accuracy**: $\mathbb{1}[\text{rank}(c^*) < 5]$.
- **MRR**: $1 / (\text{rank} + 1)$.
- **NLL**: per-event $-\log P(a_{c^*})$.
- **AIC**: $2k + 2 n_{\text{train}} \cdot \text{NLL}_{\text{test}}$, $k = 544{,}075$ for the default config.
- **BIC**: $k \log n_{\text{train}} + 2 n_{\text{train}} \cdot \text{NLL}_{\text{test}}$.

Stratified breakdowns: by category, repeat/novel, activity tertile, time-of-day, and dominant attribute.

Baselines compared against: LASSO-MNL, Bayesian ARD, RF, GB, MLP, Paz VNS, DUET, Delphos. All use the 102-dim feature pool; PO-LEU does not.

---

## 14. Suggested Repository Layout

```
po-leu/
├── configs/
│   ├── default.yaml               # M=5, K=3, softmax weights, salience on
│   ├── ablation_M3.yaml           # ... etc., one per row of Table in §11
│   └── ...
├── src/
│   ├── data/
│   │   ├── load.py                # v1, unchanged
│   │   ├── clean.py               # v1, unchanged
│   │   ├── survey_join.py         # v1, unchanged
│   │   ├── state_features.py      # v1, unchanged
│   │   ├── split.py               # v1, unchanged
│   │   ├── choice_sets.py         # v2 addition: emits z_d and c_d
│   │   ├── person_features.py     # NEW: z_d builder + standardization
│   │   └── context_string.py      # NEW: c_d builder
│   ├── outcomes/
│   │   ├── generate.py            # NEW: LLM call + cache
│   │   ├── prompts.py             # NEW: system + user templates
│   │   ├── encode.py              # NEW: encoder call + cache
│   │   ├── diversity_filter.py    # NEW: §3.5
│   │   └── cache.py               # NEW: SQLite-backed KV store
│   ├── model/
│   │   ├── attribute_heads.py     # §5
│   │   ├── weight_net.py          # §6
│   │   ├── salience_net.py        # §7
│   │   ├── po_leu.py              # end-to-end forward pass (§9.4)
│   │   └── ablations.py           # A7 (concat), A8 (FiLM)
│   ├── train/
│   │   ├── loop.py                # §9.3
│   │   ├── regularizers.py        # §9.2
│   │   └── subsample.py           # v1, retained
│   ├── eval/
│   │   ├── metrics.py             # v1, retained (§13)
│   │   ├── strata.py              # v1, retained + dominant-attribute add
│   │   └── interpret.py           # NEW: §12 reports
│   └── baselines/                 # v1, retained (LASSO-MNL, ARD, etc.)
├── outcomes_cache/                # generated artifacts (gitignored)
├── embeddings_cache/              # generated artifacts (gitignored)
├── checkpoints/
└── reports/
```

---

## 15. Reproducibility Checklist

- [ ] LLM id, temperature, top-p, seed, and prompt version logged with every cached outcome.
- [ ] Encoder id and pooling method logged with every cached embedding.
- [ ] `z_d` standardization stats ($\mu, \sigma$) saved with the split artifact.
- [ ] Every training run emits: config YAML, three seeds' final metrics, training curves (loss + val NLL per epoch), trained weights for the best-val-NLL checkpoint.
- [ ] Interpretability report (§12.1, §12.3) re-run automatically after every full training.
- [ ] Cache versioning: bumping the prompt template invalidates the generation cache only, not the encoding cache (since strings change).

---

## 16. Known Failure Modes and What to Check First

| Symptom | Likely cause | First check |
|---|---|---|
| Val NLL plateaus near $\log J = 2.30$ | Weights not learning | Confirm `z_d` is standardized; inspect `w` distribution — all-uniform means weight net isn't receiving gradient. |
| All salience mass on one outcome | Salience collapse | Raise salience-entropy $\lambda$; verify diversity filter (§3.5) ran. |
| One attribute head always dominates | Head imbalance | Check whether softmax weights are near one-hot; inspect raw weight-net output; consider $\tau > 1$ on the weight softmax. |
| Attribute heads don't cluster semantically | Heads redundant | Reduce $M$; add orthogonality regularizer between heads as a quick diagnostic; may indicate $M$ is too large. |
| Test NLL much worse than val NLL | Temporal distribution shift | Confirm train-popularity is train-only; check whether the train/test gap is concentrated in specific categories or time periods. |
| Generator produces paraphrases | Prompt too narrow | Tighten system prompt's "different type of consequence" language; enable diversity filter. |
| Training throughput low | Redundant encoding | Confirm embedding cache is hit ≥ 99%; re-encoding is the usual bottleneck. |

---

## Appendix A. Pseudocode for the End-to-End Forward Pass

```
input:  z_d  ∈ ℝ^(B × p)
        E    ∈ ℝ^(B × J × K × d_e)
        c*   ∈ ℤ^B

# 1. Attribute scores
A ← stack_m [ u_m(E) ]                      # (B, J, K, M)

# 2. Person weights
w ← softmax( weight_net(z_d) , dim=-1 )     # (B, M)
w ← w[:, None, None, :]                     # (B, 1, 1, M)

# 3. Outcome utility
U ← (A * w).sum(dim=-1)                     # (B, J, K)

# 4. Salience
S_raw ← salience_net( concat(E, broadcast(z_d)) )   # (B, J, K)
S     ← softmax( S_raw, dim=-1 )                    # (B, J, K)

# 5. Alternative value
V ← (S * U).sum(dim=-1)                     # (B, J)

# 6. Loss
ℓ ← cross_entropy( V / τ , c* )
ℒ ← ℓ + λ_w * ||φ_w||² + λ_s * H(S) + ...
```

---

## Appendix B. Minimal Hyperparameter Table

| Hyperparameter | Default | Source |
|---|---|---|
| $M$ | 5 | §5.2 |
| $K$ | 3 | §3 |
| $d_e$ | 768 | §4.1 |
| $p$ (effective) | 26 | §2.1 |
| Attribute-head hidden | 128 | §5.1 |
| Weight-net hidden | 32 | §6.1 |
| Salience-net hidden | 64 | §7.1 |
| Softmax temp $\tau$ | 1.0 | §8.2 |
| LR | $10^{-3}$ → $10^{-4}$ | §9.3 |
| Batch size | 128 | §9.3 |
| Max epochs | 30 | §9.3 |
| Patience | 5 | §9.3 |
| $\lambda_{\text{wL2}}$ | $10^{-4}$ | §9.2 |
| $\lambda_{\text{entropy}}$ | $10^{-3}$ | §9.2 |
| $\lambda_{\text{mono}}$ | $10^{-3}$ | §9.2 |
| $\lambda_{\text{div}}$ | $10^{-4}$ | §9.2 |
| Generation temperature | 0.8 | §3.3 |
| Generation top-p | 0.95 | §3.3 |
| Seeds per experiment | 3 | §9.3 |

---

## Appendix C. Customer Subsampling via PCA + KMeans + Leverage Scores

Inherited from the v2.0 pipeline (`old_pipeline/src/subsample.py`). The training loop is expensive — LLM generation + encoder passes + the three small nets — so rather than training on every customer, the pipeline picks a behaviorally diverse subset of customers and attaches importance weights so the empirical risk still approximates the full-population risk.

### C.1 Customer behavior profile

Per customer, build a profile vector on the training split only:

1. **Category distribution.** Pivot `(customer_id × category)` purchase counts, row-normalize to fractions.
2. **PCA compression.** Fit PCA on the category-fraction matrix; keep `n_pca_components = min(20, n_categories − 1, n_customers − 1)` components. Default is 20. This collapses the high-dimensional category simplex into a dense low-rank summary.
3. **Scalar features** (5, computed over the customer's prior events):
   - `repeat_rate` — fraction of events with `routine > 0`.
   - `mean_recency` — mean of $1 / (1 + \text{recency\_days})$ across events.
   - `novelty_rate` — fraction of events with `novelty == 1`.
   - `purchase_freq` — $\log(1 + \text{n\_events})$.
   - `cat_entropy` — Shannon entropy of the customer's category distribution.
4. **Concatenate** the PCA components with the 5 scalars, giving a profile matrix $X \in \mathbb{R}^{n \times d}$ with $d = $ `n_pca_components + 5` (default 25).
5. **Column-standardize** $X$ (subtract column means, divide by column stds; replace zero-std columns with 1.0). Standardization is required for leverage scores to be scale-free.

### C.2 Leverage scores

Compute the thin SVD $X = U \Sigma V^\top$, mask out singular values $\le 10^{-6}$, and set

$$h_i = \sum_{r:\ \sigma_r > 10^{-6}} U_{i,r}^2.$$

These are the diagonals of the hat matrix and measure how much customer $i$'s profile "spans a direction no other customer covers." Clip to $\ge 10^{-10}$ for numerical safety.

### C.3 KMeans-stratified selection

Raw leverage-proportional sampling tends to oversample a few extreme customers and miss whole behavioral regions. The pipeline adds coverage via KMeans stratification:

1. **Cluster** customers into $n_{\text{clusters}} = \max(n_{\text{customers}} / 5,\ 2)$ groups using KMeans on the standardized profile matrix (`n_init=10`, seeded).
2. **Guaranteed coverage.** Within each cluster, sort members by descending leverage and include the top `min_per_category_cluster = 2` into the selected set. This guarantees every behavioral cluster contributes at least two customers.
3. **Proportional fill.** Fill the remaining budget $n_{\text{customers}} - |\text{selected}|$ by sampling without replacement from the unselected customers, with probabilities $p_i \propto h_i$.

(The algorithm is KMeans-based stratified leverage sampling, not kNN.)

### C.4 Importance weights

Let $h_i$ be the leverage of customer $i$ and $q_i = h_i / \sum_{i'} h_{i'}$ the intended proportional probability. For each selected customer:

$$\tilde{\omega}_i = \frac{1}{n_{\text{customers}} \cdot q_i}, \qquad \omega_i = \tilde{\omega}_i \cdot \frac{n_{\text{total}}}{\sum_{i' \in S} \tilde{\omega}_{i'}},$$

where $n_{\text{total}}$ is the full-customer count and $S$ the selected set. The second factor rescales so $\sum_{i \in S} \omega_i = n_{\text{total}}$, which keeps the loss on the same scale as full-data training.

Every choice event inherits its customer's weight, i.e. $\omega_t = \omega_{d(t)}$. This is the $\omega_t$ consumed by §9.1.

### C.5 Typical numbers

The v2.0 Amazon run used:

- `n_customers = 500`, `n_pca_components = 20`, `min_per_category_cluster = 2`, `seed = 42`.
- Reduced **4,986 customers / 1,484,093 events → 500 customers / ~119,299 events** (≈ 8%).
- Weight range roughly $[0.12, 70.79]$; the tail reflects customers with rare behavioral profiles.

### C.6 API (from `old_pipeline/src/subsample.py`)

```python
from subsample import (
    build_customer_profiles,    # (df, n_pca_components=20) -> (profiles_df, X_standardized)
    compute_leverage_scores,    # (X) -> h  (thin-SVD hat diagonals)
    subsample_customers,        # (df, n_customers=500, n_pca_components=20,
                                #  min_per_category_cluster=2, seed=42)
                                # -> (selected_ids, customer_weights)
    apply_subsample,            # (df, selected_ids, weights)
                                # -> (filtered_df, per_event_weights)
)
```

`subsample_customers` returns `(selected_ids, weights)` aligned one-to-one. `apply_subsample` filters the training dataframe to those customers and broadcasts per-customer weights to per-event weights, ready to feed into the batch loss of §9.1. Set `n_customers = None` (or skip the call) to train on the full population with $\omega_t = 1$.

### C.7 Diagnostics

When the subsample is active, two plots are produced during data prep:

1. **Leverage histogram** with selected customers overlaid — visual check that selection is not concentrated in the low-leverage bulk.
2. **Category coverage bar** comparing unique categories present in the full vs. subsampled training set — catches cases where the subsample drops whole product categories.

---

**End of specification.**