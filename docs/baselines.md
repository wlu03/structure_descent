# Baselines and Related Work

This document describes the eight baselines implemented for empirical comparison against Structure Descent. Each baseline represents a distinct methodological family, and the set together spans the full landscape of prior approaches to fitting or searching for discrete choice utility specifications: classical statistical shrinkage, Bayesian shrinkage, flexible black-box machine learning, classical combinatorial metaheuristics, evolutionary search, and deep reinforcement learning.

Every baseline in this document is implemented as a conforming `FittedBaseline` in `src/baselines/` and is evaluated through the same shared harness (`src/baselines/evaluate.py`). Predictive metrics, information criteria, and runtime are reported on the same train/val/test splits and the same per-alternative feature matrices.

---

## 1. Methodological taxonomy

The baselines decompose into three families by methodological role:

### 1.1 Statistical shrinkage (no outer search)

These baselines fit a single global coefficient vector over a fixed, richly expanded feature pool and rely on a regularization mechanism to perform automatic variable selection. They are the natural "floor": the simplest non-trivial methods that one must beat to justify any outer loop at all.

- **LASSO-MNL** — L1-regularized conditional logit (Tibshirani, 1996, applied to the McFadden conditional-logit likelihood).
- **Bayesian ARD** — Automatic Relevance Determination priors on utility coefficients (Rodrigues et al., 2020; MacKay, 1992).

### 1.2 Flexible black-box machine learning (predictive ceiling)

These baselines abandon the utility-maximization framework in favor of off-the-shelf supervised classifiers. They treat each (event, alternative) pair as a binary classification example and rank alternatives within an event by predicted propensity. They serve as *predictive ceilings* — they are not interpretable, but they bound how much raw predictive signal is in the data.

- **Random Forest** (Breiman, 2001)
- **Gradient Boosting** (Friedman, 2001; specifically the histogram-based variant of Ke et al., 2017)
- **Multi-Layer Perceptron** (Rumelhart et al., 1986)

### 1.3 Structured search and constrained neural utility models

These baselines are direct methodological competitors to Structure Descent: each performs either an *outer* search over the combinatorial space of utility function structures (with an inner coefficient fit per candidate), or an end-to-end constrained neural utility model. They differ in how the outer search is conducted — classical metaheuristic vs. deep RL policy — or, in the DUET case, by replacing the search entirely with a single constrained ANN fit.

- **Paz VNS** — Variable Neighborhood Search (Paz, Arteaga, & Cobos, 2019)
- **DUET** — Economically-consistent ANN-based DCM (Han et al., 2024, arXiv:2404.13198)
- **Delphos** — Deep Q-Network for sequential specification decisions (Nova, Hess, & van Cranenburgh, 2025)

Structure Descent occupies a fourth point in this design space: the outer loop is an LLM proposer conditioned on the current specification and its diagnostic residuals, while the inner loop is an identical hierarchical conditional logit fit to every candidate. Paz VNS and Delphos are the *tightest* search-side controls; DUET controls for whether a constrained neural model would have made the structural search unnecessary in the first place.

---

## 2. Individual baselines

### 2.1 LASSO-MNL

**Citation.** Tibshirani, R. (1996). *Regression shrinkage and selection via the Lasso*. Journal of the Royal Statistical Society: Series B, 58(1), 267–288. Applied to the conditional-logit likelihood of McFadden (1974).

**Method.** LASSO-MNL fits a single global coefficient vector $\mathbf{w}$ over an expanded feature pool by minimising the $L_1$-penalised conditional-logit negative log-likelihood:

$$\mathcal{L}(\mathbf{w}) = \sum_{e=1}^{E} -\log \frac{\exp(\mathbf{x}_{e,c_e}^\top \mathbf{w})}{\sum_{j=1}^{J} \exp(\mathbf{x}_{e,j}^\top \mathbf{w})} + \alpha \|\mathbf{w}\|_1$$

where $e$ indexes choice events, $c_e$ is the chosen alternative, $\mathbf{x}_{e,j}$ is the feature vector for alternative $j$ in event $e$, and $\alpha$ is a non-negative regularisation strength. The $L_1$ penalty drives coefficients of irrelevant features to exactly zero, producing a sparse, automatically-selected utility specification.

**Feature pool.** The expanded pool contains the 12 base DSL features, their signed $\log(1+|x|)$ transforms, their signed squares, and all $\binom{12}{2} = 66$ pairwise interaction products, for a total of 102 candidate terms. Both LASSO-MNL and Bayesian ARD consume this identical pool so the two shrinkage baselines are strictly comparable.

**Hyperparameter selection.** The regularization strength $\alpha$ is selected by a log-spaced grid search on the validation split, using the unregularized conditional-logit NLL as the selection criterion.

**Relation to Structure Descent.** LASSO-MNL is the statistical floor: it performs no outer search and no structural reasoning. It answers the question *"do we need a search loop at all, or would regularized shrinkage over a rich pre-specified feature pool suffice?"* — a question every reviewer of a search-based method will ask.

**Implementation notes.** The baseline uses a faithful Fast Iterative Shrinkage-Thresholding Algorithm (FISTA; Beck & Teboulle, 2009) with the exact $L_1$ proximal operator (soft-thresholding) and Nesterov momentum with backtracking line search. The NLL gradient is the analytic softmax-minus-one-hot form $\sum_e \mathbf{X}_e^\top (\mathrm{softmax}(\mathbf{X}_e \mathbf{w}) - \mathbf{e}_{c_e})$. We deliberately avoid the common shortcut of using L-BFGS-B on a smoothed $\sqrt{w^2 + \varepsilon}$ surrogate, which does not produce true sparsity. The regularization path is traversed from large to small $\alpha$ with warm-starting.

---

### 2.2 Bayesian ARD

**Citation.** Rodrigues, F., Ortelli, N., Bierlaire, M., & Pereira, F. C. (2020). *Bayesian Automatic Relevance Determination for Utility Function Specification in Discrete Choice Models*. Journal of Choice Modelling. arXiv:1906.03855. Building on MacKay (1992) and Neal (1996) for the ARD prior structure.

**Method.** Rather than using a point-estimated $L_1$ penalty, Bayesian ARD places a hierarchical Gaussian prior with per-coefficient precision on the utility weights:

$$\alpha_j \sim \mathrm{Gamma}(a_0, b_0), \qquad w_j \sim \mathcal{N}(0, 1/\alpha_j)$$

with vague hyperparameters $a_0 = b_0 = 10^{-3}$. The coefficient precisions $\alpha_j$ are themselves inferred jointly with the weights. Under the data likelihood, irrelevant coefficients develop a posterior over $\alpha_j$ that concentrates at large values, driving the posterior mean of $w_j$ toward zero — this is the "automatic relevance determination" property.

The likelihood is the conditional-logit over choice sets:

$$p(c_e \mid \mathbf{X}_e, \mathbf{w}) = \mathrm{softmax}(\mathbf{X}_e \mathbf{w})[c_e]$$

and we approximate the joint posterior over $(\mathbf{w}, \boldsymbol{\alpha})$ via **doubly-stochastic variational inference** — the methodological core of Rodrigues et al. (2020).

**Feature pool.** Identical to LASSO-MNL: the 102-dimensional expanded pool over 12 base DSL features. This ensures the two shrinkage baselines differ only in their regularization mechanism, not their feature space.

**Relation to Structure Descent.** Bayesian ARD answers a subtler version of the LASSO-MNL question: *"does uncertainty-aware Bayesian shrinkage beat frequentist $L_1$ shrinkage when both consume the same feature pool?"* It also provides natural uncertainty quantification on the selected specification, which Structure Descent and the other search-based baselines do not.

**Implementation notes.** Inference defaults to NumPyro SVI with an explicit mean-field guide: $w_j \sim \mathcal{N}(\mu_j, \sigma_j)$ and $\alpha_j \sim \mathrm{LogNormal}(m_j, s_j)$. LogNormal is a positive-support stand-in for the paper's Gamma variational family — numerically stable in NumPyro's SVI without losing the support or moment range that the Gamma guide provides. The ELBO uses 4 reparameterized particles per step (`num_particles=4`) and Adam at $10^{-2}$. The No-U-Turn Sampler (NUTS; Hoffman & Gelman, 2014) is exposed as an opt-in `inference="nuts"` backend for comparison only — it is not what the paper runs and targets the notoriously ill-conditioned Gaussian-Gamma ARD funnel.

**Pruning rule.** Following the Tipping (2001) / MacKay (1994) ARD convention, coefficients are pruned by their *posterior precision*, not their posterior mean: features with $\mathbb{E}[\alpha_j] > \alpha_{\mathrm{thresh}}$ (default $10^{3}$) are forced to exactly zero, and `n_params` reports the count of surviving features for AIC/BIC accounting. A previous implementation pruned by $|\mathbb{E}[w_j]|$ instead; the current rule matches the paper. The fit is strictly single-shot — no outer search loop — matching Rodrigues et al.'s formulation.

---

### 2.3 Random Forest (Bagging + Trees)

**Citation.** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

**Method.** A Random Forest is an ensemble of decision trees, each trained on a bootstrap resample of the training data with randomised feature subsets at each split. Predictions are made by averaging class probabilities across all trees. The variance-reducing effect of bagging and the decorrelation induced by feature subsampling make RF a robust, low-tuning classifier that handles nonlinear interactions and categorical features without explicit feature engineering.

**Adaptation to choice data.** Choice events are flattened into a long-format dataset: each of the $E \cdot J$ event-alternative pairs becomes a row, with $y = 1$ for the chosen alternative and $y = 0$ otherwise. The classifier learns $P(y = 1 \mid \mathbf{x})$, and at prediction time the scores returned for a held-out choice event are $\{\log P(y = 1 \mid \mathbf{x}_{e,j})\}_{j=1}^{J}$. The shared evaluation harness applies a softmax over these log-scores, producing a valid choice-set probability $P(\text{alt } i \mid e) = p_i / \sum_j p_j$.

This "binary-classification-over-alternatives" conversion does not satisfy the axioms of a formal random utility model — in particular it lacks invariance to rescaling of alternatives' features and does not produce interpretable marginal utilities — but it is the standard way ML ceilings are reported in choice-modelling papers.

**Feature pool.** The raw 12 base DSL features, not the expanded pool. Tree ensembles learn axis-aligned nonlinearities and their products directly, so pre-expansion adds only noise.

**Class balance.** The minority-class imbalance (1 chosen alternative per $J$) is corrected via per-row `sample_weight = 1/J`, so each *event* contributes mass 1.0 to the loss regardless of choice-set size. This is preferable to sklearn's `class_weight='balanced'` because the latter warps predicted probabilities away from the base rate, which biases NLL/AIC/BIC after the harness re-softmaxes — ranking metrics (top-1/top-5/MRR) would be unaffected, but the calibrated-fit columns would be silently wrong.

**Relation to Structure Descent.** Random Forest is one of three predictive ceilings in the suite. Its role in the comparison is to bound the total predictive signal in the data: if Structure Descent recovers, say, 95% of RF's top-1 accuracy while using an order of magnitude fewer parameters and producing economically interpretable coefficients, that is the headline interpretability-vs-accuracy trade-off.

---

### 2.4 Gradient Boosting (Boosting + Trees)

**Citation.** Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189–1232. Histogram-based variant: Ke, T., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS.

**Method.** Gradient Boosting fits an additive ensemble of shallow decision trees, where each successive tree is fit to the pseudo-residuals (negative gradient of the loss) of the ensemble so far. Unlike bagging, boosting reduces bias aggressively and typically achieves the strongest raw predictive performance among tabular ML methods. We use sklearn's `HistGradientBoostingClassifier`, a histogram-based variant equivalent in spirit to LightGBM and XGBoost.

**Adaptation to choice data.** Identical to Random Forest: long-format flattening, binary classification, log-probability scoring, shared harness softmax normalization, and the same per-row `sample_weight = 1/J` correction for choice-set imbalance.

**Feature pool.** Same as Random Forest — raw 12 base features.

**Relation to Structure Descent.** Boosting is the second predictive ceiling, representing the state of the art among off-the-shelf tabular classifiers. It is typically the strongest pure-predictive baseline in tabular benchmarks. If Gradient Boosting and Structure Descent achieve comparable top-1 accuracy, that is evidence the interpretable parametric form is not costing us much predictive power on this task.

---

### 2.5 Multi-Layer Perceptron

**Citation.** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323, 533–536.

**Method.** A fully-connected feed-forward neural network with ReLU activations, trained via stochastic gradient descent with early stopping on a held-out validation fraction. We use a small architecture (two hidden layers, 32 and 16 units respectively) that roughly matches the parameter budget of the shrinkage baselines.

**Adaptation to choice data.** Identical long-format binary classification framing. The MLP is wrapped in a sklearn `Pipeline` with a `StandardScaler` so that inputs are standardised before feeding into the network — neural networks are sensitive to feature scale, and the raw DSL features have heterogeneous scales (log-transformed counts, binary flags, unnormalised prices).

**Feature pool.** Raw 12 base features. MLPs learn continuous nonlinear transformations directly through their hidden layers.

**Relation to Structure Descent.** MLP is the third predictive ceiling, included for completeness across the major ML families (bagging, boosting, neural). In tabular choice modelling, MLPs are rarely competitive with tree ensembles unless the dataset is very large, so the MLP row in the empirical comparison is expected to lag behind RF and Gradient Boosting. Its value is in documenting the full ML-ceiling landscape rather than in providing the strongest baseline.

---

### 2.6 Paz VNS

**Citation.** Paz, A., Arteaga, C., & Cobos, C. (2019). *Specification of mixed logit models assisted by an optimization framework*. Journal of Choice Modelling, 30, 50–60. Extended formulation in Paz et al. (2021), Transportation Research Part B.

**Method.** Paz, Arteaga, and Cobos formulate discrete choice model specification as a multi-objective combinatorial optimization problem and solve it with a Variable Neighborhood Search (VNS; Hansen & Mladenović, 1997) metaheuristic. Following the paper's preference for goodness-of-fit indicators over the raw NLL, the two objectives are:

- minimize **BIC**, $2 \cdot \mathrm{NLL}(S) + K(S) \log E$, where $K(S)$ is the effective hierarchical parameter count and $E$ is the number of training events (parsimony-penalized fit);
- maximize **adjusted ${\bar\rho}^2$**, $1 - (\mathrm{NLL}(S) - K(S)) / \mathrm{LL}_0$, where $\mathrm{LL}_0 = \sum_i \log(1 / |A(s_i)|)$ is the equal-probability null summed per-event over actual choice-set sizes (fit quality relative to the null, with parameter-count adjustment).

The VNS proceeds in three phases per iteration. First, a **shake** step draws one perturbation from the $k$-th neighborhood $N_k$, where $k$ indexes *which* neighborhood to draw from (not the magnitude of the perturbation). Second, a **local improvement** step hill-climbs $S$ across $N_1 \cup N_2$ using strict Pareto domination on the (BIC, $-{\bar\rho}^2$) objective pair, capped by a hard local-search budget. Third, an **acceptance** rule updates the Pareto front of discovered specifications: if the improved candidate extends or dominates the front, it is accepted and $k$ is reset to 1; otherwise $k$ is incremented until $k > k_\max$, after which the search restarts from a random front member. The algorithm terminates on an evaluation budget or after a fixed number of unproductive restarts.

**Behavioral sign check.** Paz et al. reject specifications whose coefficients violate economic priors (e.g. positive price coefficient). We replicate this as a configurable expected-sign filter `expected_signs` (default `{"price_sensitivity": -1}` for the Amazon e-commerce domain). Solutions whose global $\theta_g$ for any listed feature has the wrong sign are excluded from the Pareto archive but logged on `PazVnsFitted.sign_violations` for inspection.

The output of the method is the *full Pareto front* of sign-valid discovered specifications, not a single "best" model — reflecting the multi-objective nature of the optimization and the analyst's freedom to pick a preferred trade-off point.

**Neighborhoods.** Our implementation defines three neighborhoods over `DSLStructure`, structurally analogous to the paper's variable-inclusion / random-parameter / distribution neighborhoods but adapted to the project's DSL:

- **N1** (atomic simple edits): add a base term, remove a simple term, or swap one simple term for another.
- **N2** (compound edits): add or remove a compound term constructed via one of the DSL's Layer-3 combinators (`interaction`, `ratio`, `log_transform`, `threshold`, `power`, `decay`, `split_by`, `difference`).
- **N3** (composed multi-edit, used only for shake): a fixed-size composition of two atomic N1/N2 edits.

**Inner loop.** For each candidate structure $S$, features are computed via `dsl.build_structure_features` and the hierarchical conditional logit fitter `inner_loop.fit_weights` estimates the coefficients. BIC and $\bar\rho^2$ are then computed from the fitted log-likelihood and the *effective* hierarchical parameter count $K(S) = |\mathcal{T}| \cdot (1 + |\mathcal{C}| + |\mathcal{I}|)$ — not just $|\mathcal{T}|$ — so the parsimony objective sees the real model size.

**Relation to Structure Descent.** Paz VNS is the canonical pre-LLM automated DCM specification baseline. It serves as the "before" half of the "before and after LLMs" story: it isolates the question *"does LLM world knowledge outperform a classical combinatorial metaheuristic operating over the same DSL and the same inner loop?"* Because Structure Descent also uses a fit-vs-complexity Pareto lens internally, Paz VNS provides a directly comparable Pareto front.

**Documented deviations.** Combinator hyperparameters (e.g. `threshold` cutoff, `power` exponent, `decay` half-life) are drawn from a small discrete grid rather than tuned continuously — this matches Paz et al., who also restrict their hyperparameter search to a finite set. The neighborhoods are translated to the project's DSL, so $N_1$/$N_2$/$N_3$ are not literally the paper's variable-inclusion/random-parameter/distribution neighborhoods but their structural analogues over the same kind of additive utility space.

---

### 2.7 DUET

**Citation.** Han, Y. et al. (2024). *DUET: Economically-consistent ANN-based discrete choice model.* arXiv:2404.13198. https://arxiv.org/abs/2404.13198

**Method.** DUET is a constrained neural utility model that fuses two parallel branches into a single utility:

$$u(x_{ia} \mid \theta) = \beta^\top x_{ia} + f_\theta(x_{ia})$$

where the first term is a directly inspectable linear ("interpretable") branch with one coefficient per input feature, and the second is a small `tanh` MLP ("flexible" branch) that captures residual nonlinearities. Choice probabilities are computed by softmax over the per-event choice set. Training minimizes the conditional-logit NLL with a small $L_2$ penalty on the neural weights and a **soft monotonicity penalty** that enforces economic consistency on specific input gradients:

$$\mathcal{L}(\beta, \theta) = -\sum_i \log \mathrm{softmax}(u_i(x))[c_i] + \ell_2 \|\theta_{\mathrm{NN}}\|^2 + \lambda_{\mathrm{mono}} \sum_{j \in \mathcal{M}} \sum_{i, a} \mathrm{ReLU}\!\left(-s_j \cdot \tfrac{\partial u_{ia}}{\partial x_{ia,j}}\right)^2$$

Here $\mathcal{M}$ is the set of constrained features and $s_j \in \{-1, +1\}$ is each feature's expected sign. Gradients $\partial u / \partial x$ are computed at every training point via `torch.autograd.grad`, so the constraint is enforced *at every observation* rather than only at training-set means. The penalty is soft (not a hard projection), matching the paper's §3 formulation.

**Domain adaptation (Amazon e-commerce).** The default monotonicity targets resolve against `BaselineEventBatch.base_feature_names`:

- `price_sensitivity` — expected sign $-1$ (negative price sensitivity)
- `rating_signal` — expected sign $+1$ (positive rating sensitivity)

If either feature name is absent in the batch, the corresponding constraint is silently dropped with a warning at fit time. Custom targets can be passed via the `mono_targets` constructor kwarg.

**Architecture defaults.** The flexible branch is a 2-hidden-layer MLP with 32 units per layer and `tanh` activation. Adam at $5 \times 10^{-3}$, mini-batches of 64 events, $L_2 = 10^{-4}$, $\lambda_{\mathrm{mono}} = 1.0$, early stopping on val NLL with patience 15.

**Relation to Structure Descent.** DUET is the constrained-neural-network ceiling: it answers the question *"would an end-to-end ANN with hard-coded economic priors make the structural search unnecessary?"* If Structure Descent and DUET reach comparable predictive accuracy with Structure Descent producing a strictly more interpretable additive utility, that is direct evidence the structural search adds interpretability without giving up predictive ground.

**Documented deviations.** None substantive. The `_ga` suffix in the filename `src/baselines/duet_ga.py` is legacy from a prior (incorrect) GA-over-DSL implementation that previously occupied this slot; back-compat aliases `DuetGA` / `DuetGAFitted` are retained so callers and notebooks continue to import cleanly. The current implementation is the faithful arXiv:2404.13198 model.

---

### 2.8 Delphos

**Citation.** Nova, G., Hess, S., & van Cranenburgh, S. (2025). *Delphos: A reinforcement learning framework for assisting discrete choice model specification*. arXiv:2506.06410. Public code: [github.com/TUD-CityAI-Lab/Delphos](https://github.com/TUD-CityAI-Lab/Delphos), CC BY-NC-SA 4.0.

**Method.** Delphos frames DCM specification as a Markov Decision Process and trains a Deep Q-Network (DQN; Mnih et al., 2015) to learn a policy over specification-building actions. The state is the current partial utility specification, encoded as a multi-hot feature vector. The action space comprises three operation types:

- `('terminate')`: finalise the current specification and submit it to the inner estimator.
- `('add', var, trans)`: add a new term with the given variable and transformation.
- `('change', var, trans)`: replace the current transformation on a variable.

A dynamic action mask prevents the agent from taking invalid actions (e.g. adding a variable already in the spec, applying a non-linear transformation to the alternative-specific constant).

Episodes proceed by starting from an empty specification and applying epsilon-greedy actions chosen by the DQN until a `terminate` action is selected. The terminal state is then estimated via the inner loop, and a scalar reward is computed as

$$r = \sum_m \omega_m \cdot \widetilde{M}_m \cdot \mathbb{1}_{\text{converged}}$$

where $\omega_m$ is the user-specified weight on metric $m$ (by default, AIC receives weight 1), $\widetilde{M}_m$ is the metric normalised by reference to the equal-probability null log-likelihood, and the indicator ensures failed estimates receive zero reward. The terminal reward is then distributed across all transitions in the episode using an exponential discount $\gamma^{L - \ell - 1}$, so that actions taken earlier in a successful trajectory receive less credit than actions taken near the terminal state.

Training proceeds via standard DQN experience replay: transitions are stored in a replay buffer, mini-batches are sampled uniformly, and the policy network's Q-values are regressed against a target network that is hard-copied from the policy network every $k$ episodes. Epsilon decays linearly from 1.0 to 0.01 over training, and early stopping triggers when rolling-window reward improvement drops below a threshold for a patience window.

**Inner loop.** In the original paper, Delphos uses the R package Apollo (Hess & Palma, 2019) as its inner MNL estimator, invoked from Python via the `rpy2` bridge. Our implementation replaces this with our hierarchical conditional-logit fitter (`inner_loop.fit_weights`), wrapped in an adapter that produces a pandas DataFrame with the exact column names Delphos's reward function expects (`LL0`, `LLC`, `LLout`, `rho2_0`, `adjRho2_0`, `rho2_C`, `adjRho2_C`, `AIC`, `BIC`, `numParams`, `successfulEstimation`). The metrics are computed honestly:

- `LL0` is the equal-probability null log-likelihood summed per-event over actual choice-set sizes, $\sum_i \log(1 / |A(s_i)|)$, not a constant $E \cdot \log(1/J)$ (the choice-set size can vary between events).
- `LLC` is the constants-only MNL log-likelihood. Because our DSL carries no alternative-specific constants, `LLC` reduces to `LL0`.
- `LLout` is the log-likelihood evaluated on the validation batch, which is threaded through `Delphos.fit(train, val)` into the estimator. It is *not* aliased to the training LL.
- `rho2_C` and `adjRho2_C` are derived from `LLC` rather than aliased to `rho2_0`.
- `numParams` uses the *effective* hierarchical parameter count $|\mathcal{T}| \cdot (1 + |\mathcal{C}| + |\mathcal{I}|)$, not just the DSL term count, so AIC/BIC reflect the true model size.

**Relation to Structure Descent.** Delphos is the most recent direct competitor in the automated DCM specification literature, published in 2025 and using modern deep reinforcement learning. It is the "novelty vs recent work" baseline: any 2026 paper on automated DCM specification must address why its approach is preferable to Delphos. The comparison is particularly interesting because Delphos and Structure Descent both use search methods that incorporate *sequential* reasoning about which term to add next, but they differ on the knowledge source: Delphos learns the policy from training experience (data-driven, tabula rasa), while Structure Descent draws on an LLM's embedded knowledge of economic behaviour (pre-trained, zero-shot).

**Documented deviations.** Several simplifications are required to port Delphos to our data format:

- **Alternative-specific constants** (the `var = 0` slot in Delphos's action space) are dropped because our hierarchical fitter does not carry ASCs as a separate parameter.
- **Alternative-specific taste** (Delphos's `spec` dimension in the tuple representation) is dropped because our choice events share coefficients across alternatives by design.
- **Covariate interactions** (Delphos's `cov` dimension) are dropped in this port; a future extension could map them onto our DSL's Layer-3 binary combinators.
- **Box-Cox transformation** is approximated by a fixed-exponent `power(exponent=0.5)` term, since our DSL does not include a learned $\lambda$ parameter.
- **The disk-based `rewards.csv` cache** is replaced with an in-memory dict keyed on the encoded state string.

The RL core (DQNetwork, ReplayBuffer, StateManager, action-space definition, masking logic, reward function, normalization, experience replay, target network sync, epsilon schedule, early stopping, reward distribution across trajectories) is preserved line-for-line from the upstream `agent.py`. Only the inner estimator and the disk-persistence layer are rewritten. An independent verification agent confirmed that upstream Delphos itself does not implement the paper's "behavioural plausibility" sign-check indicator (the $\mathbb{1}_{\text{behavioural}}$ term in Eq. 17), so our omission is faithful to the reference implementation rather than a new deviation.

---

## 3. Comparison matrix

The table below summarises the key axes of each baseline. All entries are filled based on the reference paper and our implementation.

| # | Baseline | Family | Outer search | Inner estimator | Feature pool | Parametric form | Interpretable |
|---|---|---|---|---|---|---|---|
| 1 | LASSO-MNL | shrinkage | none | $L_1$-penalised conditional logit | expanded (102) | linear | yes |
| 2 | Bayesian ARD | shrinkage | none | doubly-stochastic VI on ARD prior | expanded (102) | linear | yes |
| 3 | Random Forest | bagging ML | none | RF classifier | raw (12) | non-parametric | no |
| 4 | Gradient Boosting | boosting ML | none | HistGB classifier | raw (12) | non-parametric | no |
| 5 | MLP | neural ML | none | MLP classifier | raw (12) | non-parametric | no |
| 6 | Paz VNS | combinatorial | Variable Neighborhood Search (BIC, $\bar\rho^2$) | hierarchical MNL | DSL structure | linear | yes |
| 7 | DUET | constrained neural | none (single ANN fit) | linear branch + tanh MLP, sign-constrained | raw (12) | linear + small MLP | partially |
| 8 | Delphos | reinforcement learning | Deep Q-Network | hierarchical MNL | DSL structure | linear | yes |
| — | **Structure Descent** | LLM-guided search | LLM proposer | hierarchical MNL | DSL structure | linear | yes |

### What each baseline rules out

| Baseline | Null hypothesis it rules out if Structure Descent beats it |
|---|---|
| LASSO-MNL | "Regularised shrinkage over a rich pool would have sufficed; no search is needed." |
| Bayesian ARD | "A Bayesian treatment of the same pool would have sufficed." |
| Random Forest / Gradient Boosting | "The interpretable parametric form is free — a black-box ML ceiling isn't much higher than our method." (Recovered fraction of the ML top-1 is the headline metric for the interpretability–accuracy trade-off.) |
| MLP | Same as above, for the neural family. |
| Paz VNS | **"Classical combinatorial search over the same DSL already solves the problem; LLM reasoning isn't needed."** This is the tightest classical-search control. |
| DUET | "A constrained ANN with hard-coded economic priors would have made the structural search unnecessary." (Tests the interpretability-vs-flexibility frontier on the same input features.) |
| Delphos | "A learned RL policy is the state of the art, and our approach cannot beat the 2025 best." |

---

## 4. Evaluation protocol

Every baseline is evaluated through the same shared harness (`src/baselines/evaluate.py`), which produces a `BaselineReport` with the following metric panel on a held-out test split:

- **Predictive accuracy**: top-1, top-5, mean reciprocal rank (MRR)
- **Calibrated fit**: test-set negative log-likelihood per event
- **Information criteria**: AIC and BIC, computed using the training set size and the baseline's `n_params` attribute
- **Structural breakdowns**: top-1 accuracy per category, top-1 accuracy for repeat-vs-novel purchases
- **Runtime**: fit wall-clock in seconds

The `n_params` convention varies by family:

- Shrinkage methods report the count of non-zero coefficients after hard-thresholding.
- Tree ensembles report the total leaf count across the ensemble (bagged or boosted).
- MLPs report the total count of weights and biases.
- Structured-search methods and Structure Descent report $|\mathcal{T}| \cdot (1 + |\mathcal{C}| + |\mathcal{I}|)$, where $|\mathcal{T}|$ is the number of terms in the final structure, $|\mathcal{C}|$ is the number of distinct categories, and $|\mathcal{I}|$ is the number of distinct individuals — reflecting the flat parameter layout of the hierarchical fit.

AIC and BIC values are therefore only comparable within families. Cross-family comparison should focus on the predictive accuracy metrics (top-1, top-5, MRR, NLL) and on the structural quality of the fitted model (parsimony, coefficient plausibility).

---

## 5. Positioning statement

Structure Descent sits at the intersection of two literatures. From the LLM-guided program search tradition (FunSearch, Romera-Paredes et al. 2023; LLM-SR, Shojaee et al. 2025; LaSR, Grayeli et al. 2024), it inherits the outer-loop architecture of an LLM proposer iteratively improving candidates against an evaluator. From the automated DCM specification tradition (Paz et al. 2019; Rodrigues et al. 2020; Han et al. 2024; Nova et al. 2025), it inherits the domain framing and the evaluation protocol: a hierarchical conditional logit inner loop, behavioural-plausibility constraints, and economic interpretability as a primary goal.

The baselines in this document populate both of these literatures. Collectively, they let the empirical section of a Structure Descent paper make four distinct claims:

1. **Search is necessary** (vs LASSO-MNL and Bayesian ARD).
2. **Interpretable beats black-box on the parsimony-adjusted frontier** (vs Random Forest, Gradient Boosting, and MLP).
3. **Structural search beats a constrained-ANN end-to-end fit** (vs DUET — the constrained-neural ceiling).
4. **We are competitive with or superior to recent direct competitors** (vs Delphos, the 2025 state of the art; and vs Paz VNS, the canonical pre-LLM automation).

No single baseline is "best" in isolation; each controls for a distinct null hypothesis, and the full suite is the relevant comparison.

---

## References

- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Grayeli, A., Sehgal, A., Costilla-Reyes, O., Cranmer, M., & Chaudhuri, S. (2024). Symbolic regression with a learned concept library. *NeurIPS 2024*. arXiv:2409.09359.
- Han, Y. et al. (2024). DUET: Economically-consistent ANN-based discrete choice model. arXiv:2404.13198. https://arxiv.org/abs/2404.13198
- Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. *Journal of Machine Learning Research*, 1, 211–244.
- Hansen, P., & Mladenović, N. (1997). Variable neighborhood search. *Computers & Operations Research*, 24(11), 1097–1100.
- Hess, S., & Palma, D. (2019). Apollo: A flexible, powerful and customisable freeware package for choice model estimation and application. *Journal of Choice Modelling*, 32, 100170.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.
- MacKay, D. J. C. (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415–447.
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105–142). Academic Press.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Neal, R. M. (1996). *Bayesian learning for neural networks*. Springer.
- Nova, G., Hess, S., & van Cranenburgh, S. (2025). Delphos: A reinforcement learning framework for assisting discrete choice model specification. arXiv:2506.06410.
- Paz, A., Arteaga, C., & Cobos, C. (2019). Specification of mixed logit models assisted by an optimization framework. *Journal of Choice Modelling*, 30, 50–60.
- Rodrigues, F., Ortelli, N., Bierlaire, M., & Pereira, F. C. (2020). Bayesian Automatic Relevance Determination for Utility Function Specification in Discrete Choice Models. *Journal of Choice Modelling*. arXiv:1906.03855.
- Romera-Paredes, B., et al. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468–475.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Shojaee, P., et al. (2025). LLM-SR: Scientific equation discovery via programming with large language models. *ICLR 2025 Oral*. arXiv:2404.18400.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
