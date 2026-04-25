# PO-LEU Predictive Improvements — Deferred Ideas

Strategies researched on the `predictive-improv` branch (2026-04-25) but **not** scheduled for the current implementation pass. Each entry preserves the research findings so we can return to them later without redoing the literature search.

Currently scheduled for implementation: **A (free wins bundle)** and **B (tabular residual branch)**.

Deferred for future iterations: **D, E, F** below.

---

## D — Widen the M·K=15 scalar bottleneck

### Summary
PO-LEU's attribute heads project `(B, J, K=3, d_e=768)` down to `(B, J, K, M=5)` — i.e. `M·K=15` scalars per alternative before person info `z_d` enters via the weight network. ST-MLP keeps a 64-d hidden state to its final linear, which is why a 49K-param ST-MLP outperforms a 544K-param PO-LEU at smoke-10. Widening the per-(m,k) channel from 1 scalar to a `d=64` vector recovers expressiveness while keeping the M-named-attribute structure.

### Recommended design (from research)
**Vector-valued heads with learned per-head read-out vectors:**

```
A: (B, J, K=3, M=5, d=64)             # was (B, J, K, M)
r: (M=5, d=64)                        # learned, one read-out direction per head
A_scalar = einsum("bjkmd,md->bjkm", A, r)   # legacy view for §12 interpretability
```

The `A_scalar` derived view preserves the existing interpretability tensor contract; downstream (`U`, `S`, `V`) is unchanged. M head names survive because each head is still its own MLP referenced by index `m`; the read-out `r_m` is "the named direction in d-space".

### Expected impact
**+8-12 pts top-1 (speculative).** Anchored to FiLM CLEVR results (~2× error reduction when conditioning channels widen) and Set Transformer pooling-attention gains. Not directly measured on choice tasks.

### Files to touch
- `src/model/attribute_heads.py:47-68` — widen `nn.Linear(hidden, 1)` → `nn.Linear(hidden, 64)`; add `nn.Parameter(M, 64)` read-out
- `src/model/po_leu.py:184-202` — recompute `A_scalar` for intermediates; rest of forward is unchanged
- `src/train/regularizers.py:141-231` — **monotonicity rewrite required** (~5 lines): project `fin_head(E)` along `r_fin` before applying alt-level surrogate
- `configs/default.yaml` — add `head_d: 64`, optional `lambda_orthogonality: 1e-4`
- `tests/model/` — new shape-contract tests

### Param count change
+40,315 params (~+7.4% on existing 544K).

### Risks
- **Head entanglement.** Without per-attribute supervision (only price-monotonicity provides any), M=5 heads can collapse to redundant features. Mitigation: head-orthogonality regularizer `λ·sum_{m≠m'}(r_m · r_{m'})²` to prevent collapse.
- **Attention degeneracy at smoke scale.** Prior literature reports up to 50% of attention heads in pretrained transformers collapse to lazy/uniform. Smoke-10's 3,372 records is well below the data threshold where wider heads have been validated.
- **Overfit on smoke-10.** Already 161:1 param-to-data ratio. Going to 175:1 worsens that. Defer until data scale is larger or until A+B leave a residual gap.

### Why deferred
The bottleneck argument is sound architecturally, but at smoke-10 scale the additional capacity is more likely to overfit than help. A+B are predicted to close most of the gap with zero/+3 params; revisit D after measuring residual error and ideally at n≥50 customer scale.

### Key sources
- [FiLM (Perez et al. 2017)](https://arxiv.org/abs/1709.07871)
- [Set Transformer (Lee et al. 2019)](https://arxiv.org/abs/1810.00825)
- [α-entmax / sparsemax (Peters et al. 2019)](https://arxiv.org/abs/1905.05702)
- [Stabilizing Transformer Training (2023)](https://arxiv.org/abs/2303.06296) — attention entropy collapse evidence
- [Disentangling via Multi-task Learning (2024)](https://arxiv.org/abs/2407.11249)

---

## E — Knowledge distillation from ST-MLP

### Summary
ST-MLP (top-1 56.2%) is structurally simpler than PO-LEU but predicts better. The hypothesis is that PO-LEU's interpretability priors (M-head bottleneck, salience entropy, monotonicity) constrain optimization and force it into suboptimal local minima. KD with ST-MLP as teacher could pull PO-LEU toward better predictions while keeping its architecture intact.

### Recommended design (from research)
**Soft-target output KD:**

```
L = α · CE(z_S, c_star, ω)
  + (1-α) · T² · KL(softmax(z_T/T) || softmax(z_S/T))
  + λ_reg · R(model, intermediates, E, prices)
```

Hyperparams: **α=0.5, T=4** (Hinton 2015 standard). Tuning grid: α∈{0.3, 0.5, 0.7}, T∈{2, 4}.

Use ST-MLP **as-is** (no retraining). Cache `teacher_logits` once per dataset split — teacher is deterministic at eval, so a single `(N_events, J)` numpy artifact avoids recomputation.

### Critical input mismatch
- ST-MLP consumes 1 metadata sentence per alt → `(B, J, 768)`
- PO-LEU consumes K=3 narratives per alt → `(B, J, K=3, 768)`

**Output-space KD is fine** — both produce logits over the same J alternatives. Feature-level KD (FitNets-style) requires either retraining ST-MLP on narratives or a learned projector; not recommended for the first pass.

### Expected impact
**+3-8 pts top-1, NLL drop 0.2-0.5, ECE drop 0.18 → ~0.10.** The strongest theoretical argument comes from Stanton & Izmailov 2021 "Does KD Really Work?" — KD often acts as regularization/curriculum even when fidelity to the teacher is incomplete, which fits PO-LEU's "stuck in bad local minima" hypothesis.

### Files to touch
- `src/train/loop.py:394-424` (`_compute_batch_loss`) — add KD term alongside regularizers
- `src/train/loop.py:516-594` — pre-compute and cache teacher logits at fit time
- `iter_batches` data path — attach `teacher_logits` field to batch dict
- `configs/default.yaml` — add `kd_alpha`, `kd_temperature`, `kd_teacher_path`

### Param count change
0 (no architecture change).

### Risks
- **KD overrides interpretability priors.** If `(1-α)·KL` dominates `λ·R_salience + λ·R_monotonicity`, M heads can collapse to homogeneous representations and salience can flatten. Mitigation: start α=0.7 (favor CE), monitor head cosine similarity and salience entropy before/after KD.
- **Backstop (Proposal C from research):** if interpretability metrics degrade, switch to Relational KD (Park 2019) which only constrains pairwise structure of scores across J alternatives, leaving internals free to keep their structured priors.
- **Teacher calibration unknown.** ST-MLP's ECE hasn't been measured. KD typically improves student ECE only when teacher is well-calibrated. Measure ST-MLP ECE before KD as a sanity check.

### Why deferred
KD is sound and zero-param, but it depends on B (or D) succeeding first. KD without B forces PO-LEU to mimic ST-MLP's behavior using a strict subset of ST-MLP's input information (no numeric price). Apply B first, see if PO-LEU can leverage the new pathway under standard CE loss; only add KD if there's a gap left.

### Key sources
- [Hinton et al. 2015 — Soft-target KD](https://arxiv.org/abs/1503.02531)
- [Park 2019 — Relational KD (backstop)](https://arxiv.org/abs/1904.05068)
- [Stanton & Izmailov 2021 — Does KD Really Work? (motivation)](https://arxiv.org/abs/2106.05945)
- [Reddi et al. AISTATS 2021 — RankDistil](http://proceedings.mlr.press/v130/reddi21a/reddi21a.pdf)
- [Furlanello 2018 — Born-Again Networks](https://arxiv.org/abs/1805.04770)

---

## F — Cross-attention between alternatives

### Summary
PO-LEU currently scores each alternative independently; alternatives only interact at the final softmax (textbook MNL / IIA). On Amazon e-commerce, relative context matters: a $5 item next to a $50 item is more attractive than next to other $5 items. Cross-attention over the J-axis lets the model see all J alternatives jointly when computing each alternative's utility.

### Recommended design (from research)
**Light cross-attention on V (post-heads, pre-softmax)** — attend on V *after* heads compute, NOT on E before, to preserve the `(A, w, U, S)` decomposition:

```
V = (S * U).sum(dim=-1)                    # (B, J)
V_emb = V.unsqueeze(-1) + alt_summary(E)   # (B, J, d_attn=64)
V_attn, _ = MHA(V_emb, V_emb, V_emb)
V' = sigmoid(gate) · proj(V_attn) + (1 - sigmoid(gate)) · V
logits = V' / τ
```

Config: 2-head MHA d=64, dropout=0.2, no positional encoding (preserves permutation invariance), gated residual init at 0 (gate scalar starts at large negative bias so initial behavior = baseline PO-LEU).

### Expected impact
**+3-8 pts top-1 (speculative).** Anchored to Pobrotyn et al. 2020 (+2.3% NDCG@5 on MSLR-WEB30K) and RankFormer (KDD 2023) Amazon Search results. Could regress with bad hyperparameters.

### Files to touch
- `src/model/po_leu.py:197-200` — insert cross-attention block + gate parameter
- `tests/model/` — add permutation-invariance test (random-permute J, assert chosen-alt logit follows the permutation)
- `configs/default.yaml` — add `cross_attn_d`, `cross_attn_heads`, `cross_attn_dropout`, `gate_init_bias`

### Param count change
~+70K params (~+13% on existing 544K).

### Theoretical implications
- **IIA violation:** Yes, by design. RUM consistency is preserved (cross-attention over deterministic features + iid Gumbel errors is a valid GEV-class model), just not MNL. Most behavioral evidence (Benson 2016) shows IIA is empirically false for product choice anyway, so framing this as "controlled IIA relaxation" is honest.
- **Monotonicity becomes set-conditional.** Raising B's price could change A's V. The current alt-level price-monotonicity penalty becomes a "given the rest of the set" statement. Plan: replace with a per-alt finite-difference probe (`perturb only alt j's price by ε in the embedding, check dV_j/dp_j ≤ 0`).
- **Counterfactuals reframe** from "what if alt j had attribute X?" to "what if alt j had attribute X, holding rest of set fixed?" — set-conditional. Still well-defined but a different query.

### Risks
- **Smoke-10 overfit risk is highest of all 5 strategies.** Ai et al. 2019 (GSF) found gains peak at group size m=2-3, *degrade* at larger groups. J=10 with 3,372 records is in the danger zone. Set Transformer (full encoder) was explicitly NOT recommended at this scale.
- **Attention rank collapse.** Up to 50% of attention heads in pretrained transformers collapse to rank-1 patterns. At smoke-10 scale this risk is amplified by limited gradient signal.
- **Permutation invariance must be guaranteed and tested.** MHA without positional encodings is naturally equivariant; downstream softmax is invariant. But a regression test is non-negotiable.

### Why deferred
Highest overfit risk and the most invasive change to the interpretability story. Defer until: (a) larger data scale (n≥50 customers), (b) A+B+E have been measured and a residual gap remains, (c) the paper's core claim is finalized — if the paper claims "RUM-consistent interpretable choice", F is off the table; if the paper claims "interpretable neural choice for e-commerce", F is defensible as a controlled IIA relaxation.

### Key sources
- [Pobrotyn et al. 2020 — Context-Aware LTR with Self-Attention (closest analogue, +2.3% NDCG@5)](https://arxiv.org/abs/2005.10084)
- [RankFormer KDD 2023 — Amazon Search](https://arxiv.org/abs/2306.05808)
- [Modeling Choice via Self-Attention (theoretical: low-rank attention choice = O(m) sample complexity vs Halo-MNL Ω(m²))](https://arxiv.org/abs/2311.07607)
- [Set Transformer (Lee et al. 2019) — full encoder; NOT recommended at smoke scale](https://arxiv.org/abs/1810.00825)
- [Groupwise Scoring Functions (Ai et al. 2019) — gains peak at m=2-3, degrade after](https://arxiv.org/abs/1811.04415)
- [Benson et al. 2016 — IIA is empirically false in product choice](https://www.cs.cornell.edu/~arb/papers/iia-www2016.pdf)

---

## When to revisit

Trigger conditions for promoting any of D, E, F to active implementation:

| Strategy | Trigger |
|---|---|
| **D** | A+B yields top-1 < 50% AND a real run at n≥50 customers is available (smoke-10 overfit risk too high otherwise) |
| **E** | A+B yields top-1 < 52% AND ST-MLP ECE measurement confirms it's well-calibrated enough to act as a teacher |
| **F** | A+B+E has plateaued AND data scale is n≥100 customers AND the paper narrative accepts "controlled IIA relaxation" |

Each can be implemented independently of the others, so revisit order is determined by whichever residual gap is most urgent at the time.
