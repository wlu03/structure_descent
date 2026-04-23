"""
Bayesian ARD (Automatic Relevance Determination) baseline for discrete choice.

Reference
---------
Rodrigues, F., Ortelli, N., Bierlaire, M., Pereira, F. (2020).
    "Bayesian Automatic Relevance Determination for Utility Function
    Specification in Discrete Choice Models." arXiv:1906.03855.

Key idea: place a hierarchical Gaussian-Gamma ARD prior over a LARGE pool of
candidate utility terms. Each coefficient w_j has its own precision alpha_j
learned jointly with the data; as alpha_j -> infinity the posterior mean of
w_j collapses to zero, so irrelevant terms are "pruned" automatically --
without any discrete search or outer loop.

Prior specification (exact, per Rodrigues et al.):

    alpha_j ~ Gamma(a0=1e-3, b0=1e-3)                 [vague hyperprior]
    w_j     ~ Normal(0, sigma_j)   with sigma_j = 1 / sqrt(alpha_j)

Likelihood (conditional logit):

    log p(data | w) = sum_e  log softmax(X_e @ w)[chosen_e]

where X_e is the shared expanded feature pool from feature_pool.expand_batch
(apples-to-apples with LASSO-MNL).

Inference — default is VI, matching the paper
----------------------------------------------
The paper's core contribution is **doubly-stochastic variational inference**
(mean-field VI with reparameterized sampling and mini-batch subsampling of
the data). We default to VI accordingly.

Guide family:
  - w_j     ~ Normal(loc_j, softplus(scale_j))   [mean-field Gaussian]
  - alpha_j ~ LogNormal(mu_j, softplus(sigma_j)) [positive-support proxy
              for the paper's Gamma variational family; LogNormal is
              numerically stable in NumPyro's SVI while supporting the
              same range and shape]

An opt-in 'nuts' mode runs the NumPyro NUTS sampler against the same model.
NUTS is not what the paper runs and targets the notoriously ill-conditioned
Gaussian-Gamma ARD funnel, so it is kept only as a comparison backend.

Pruning rule — alpha-based, matching Tipping 2001 / MacKay 1994
---------------------------------------------------------------
Rodrigues et al. (and the broader ARD/RVM literature) prune coefficients
whose posterior precision has grown large: ``alpha_j > alpha_threshold``
means the coefficient is effectively forced to zero by the prior. We use
the posterior mean of alpha_j for this test.

    alpha_threshold  — default 1e3. Features with posterior_mean(alpha_j)
                       above this are pruned to exactly 0.
    n_pruned         — count of pruned coefficients.
    n_params         — count of surviving (non-pruned) coefficients, so
                       AIC/BIC reward the sparsity ARD induces.

No outer search: ARD is single-shot. One expanded pool, one inference pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import BaselineEventBatch, FittedBaseline
from .feature_pool import expand_batch


# -----------------------------------------------------------------------------
# NumPyro model definition
# -----------------------------------------------------------------------------


def _build_stacked_design(
    expanded_list: List[np.ndarray],
    chosen_indices: List[int],
) -> "tuple[np.ndarray, np.ndarray]":
    """Stack a list of per-event design matrices into one (E, A, F) tensor."""
    n_events = len(expanded_list)
    if n_events == 0:
        raise ValueError("Bayesian ARD received an empty batch")
    n_alts, n_features = expanded_list[0].shape
    stacked = np.zeros((n_events, n_alts, n_features), dtype=np.float32)
    for e, feats in enumerate(expanded_list):
        if feats.shape != (n_alts, n_features):
            raise ValueError(
                f"event {e} has design shape {feats.shape}, "
                f"expected ({n_alts}, {n_features}). "
                "Bayesian ARD requires a uniform choice-set size."
            )
        stacked[e] = feats
    chosen = np.asarray(chosen_indices, dtype=np.int32)
    return stacked, chosen


def _ard_model(X: "np.ndarray", y: "np.ndarray", a0: float = 1e-3, b0: float = 1e-3):
    """NumPyro model: ARD prior + conditional-logit likelihood."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_events, n_alts, n_features = X.shape

    alpha = numpyro.sample(
        "alpha",
        dist.Gamma(a0 * jnp.ones(n_features), b0 * jnp.ones(n_features)),
    )
    sigma = 1.0 / jnp.sqrt(alpha)
    w = numpyro.sample(
        "w",
        dist.Normal(jnp.zeros(n_features), sigma),
    )

    logits = jnp.einsum("eaf,f->ea", X, w)
    numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)


def _ard_guide(
    X: "np.ndarray",
    y: "np.ndarray",
    a0: float = 1e-3,
    b0: float = 1e-3,
):
    """Explicit mean-field guide: Normal on w, LogNormal on alpha.

    LogNormal is used instead of Gamma because NumPyro's SVI handles
    LogNormal's reparameterized gradient robustly across feature pool
    sizes, whereas a learnable Gamma guide is numerically fragile.
    LogNormal has positive support and can match any Gamma moments, so
    it is a standard stand-in for ARD VI in practice (see the
    pyro-ppl examples for BayesianLinearRegression).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints

    n_features = X.shape[2]

    w_loc = numpyro.param(
        "w_loc", jnp.zeros(n_features),
    )
    w_scale = numpyro.param(
        "w_scale", 0.1 * jnp.ones(n_features),
        constraint=constraints.positive,
    )
    numpyro.sample("w", dist.Normal(w_loc, w_scale))

    alpha_loc = numpyro.param(
        "alpha_loc", jnp.zeros(n_features),  # exp(0) = 1 -> matches Gamma(1e-3, 1e-3) prior mean region
    )
    alpha_scale = numpyro.param(
        "alpha_scale", 0.5 * jnp.ones(n_features),
        constraint=constraints.positive,
    )
    numpyro.sample("alpha", dist.LogNormal(alpha_loc, alpha_scale))


# -----------------------------------------------------------------------------
# Fitted object
# -----------------------------------------------------------------------------


@dataclass
class BayesianARDFitted:
    """Fitted Bayesian ARD, conforming to the FittedBaseline protocol."""

    name: str
    posterior_mean_weights: np.ndarray   # after pruning (zeros for pruned coefs)
    posterior_alpha: np.ndarray          # posterior mean precisions, length F
    feature_names: List[str]
    include_interactions: bool
    n_pruned: int
    inference: str
    alpha_threshold: float
    n_warmup: int
    n_samples: int

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        expanded_list, _ = expand_batch(batch, self.include_interactions)
        return [feats @ self.posterior_mean_weights for feats in expanded_list]

    @property
    def n_params(self) -> int:
        return int(np.sum(self.posterior_mean_weights != 0.0))

    @property
    def description(self) -> str:
        total = int(self.posterior_mean_weights.shape[0])
        nz = self.n_params
        max_alpha = float(np.max(self.posterior_alpha)) if self.posterior_alpha.size else 0.0
        return (
            f"Bayesian-ARD inference={self.inference} "
            f"{nz}/{total} nonzero n_pruned={self.n_pruned} "
            f"alpha_thresh={self.alpha_threshold:g} "
            f"max_alpha={max_alpha:.2g}"
        )


# -----------------------------------------------------------------------------
# Fit-time class
# -----------------------------------------------------------------------------


class BayesianARD:
    """Bayesian ARD baseline (Rodrigues et al. 2020, arXiv:1906.03855).

    Parameters
    ----------
    include_interactions : bool
        Forwarded to expand_batch(); toggles pairwise x_i * x_j terms.
    n_warmup : int
        VI optimization steps (or NUTS warmup steps if inference='nuts').
    n_samples : int
        Posterior samples drawn from the fitted guide (or NUTS samples).
    inference : {'vi', 'nuts'}
        Posterior inference method. Default 'vi' matches the paper. 'nuts'
        is provided for comparison only and is not what the paper runs.
    seed : int
        PRNG seed fed to JAX.
    alpha_threshold : float
        Features with posterior_mean(alpha_j) above this value are pruned
        to zero. Default 1e3 follows the Tipping/MacKay ARD convention.
    a0, b0 : float
        Gamma(a0, b0) hyperprior on each per-coefficient precision alpha_j.
    num_particles : int
        VI ELBO particles for doubly-stochastic variance reduction.
    """

    name = "Bayesian-ARD"

    def __init__(
        self,
        include_interactions: bool = True,
        n_warmup: int = 2000,
        n_samples: int = 500,
        inference: str = "vi",
        seed: int = 0,
        alpha_threshold: float = 1e3,
        a0: float = 1e-3,
        b0: float = 1e-3,
        num_particles: int = 4,
    ):
        if inference not in ("nuts", "vi"):
            raise ValueError(f"inference must be 'vi' or 'nuts', got {inference!r}")
        self.include_interactions = include_interactions
        self.n_warmup = int(n_warmup)
        self.n_samples = int(n_samples)
        self.inference = inference
        self.seed = int(seed)
        self.alpha_threshold = float(alpha_threshold)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.num_particles = int(num_particles)

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> BayesianARDFitted:
        train_exp, feature_names = expand_batch(train, self.include_interactions)
        X_np, y_np = _build_stacked_design(train_exp, train.chosen_indices)

        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
        import numpyro.optim as optim

        X = jnp.asarray(X_np)
        y = jnp.asarray(y_np)
        rng = jax.random.PRNGKey(self.seed)

        if self.inference == "nuts":
            kernel = NUTS(_ard_model)
            mcmc = MCMC(
                kernel,
                num_warmup=self.n_warmup,
                num_samples=self.n_samples,
                num_chains=1,
                progress_bar=False,
            )
            mcmc.run(rng, X=X, y=y, a0=self.a0, b0=self.b0)
            samples = mcmc.get_samples()
            w_samples = np.asarray(samples["w"])
            alpha_samples = np.asarray(samples["alpha"])
        else:  # 'vi' — doubly-stochastic VI with explicit Normal/LogNormal guide
            svi = SVI(
                _ard_model,
                _ard_guide,
                optim.Adam(1e-2),
                Trace_ELBO(num_particles=self.num_particles),
            )
            svi_result = svi.run(
                rng, self.n_warmup, X=X, y=y, a0=self.a0, b0=self.b0,
                progress_bar=False,
            )
            params = svi_result.params
            w_loc = np.asarray(params["w_loc"])
            w_scale = np.asarray(params["w_scale"])
            alpha_loc = np.asarray(params["alpha_loc"])
            alpha_scale = np.asarray(params["alpha_scale"])

            rng_sample = jax.random.PRNGKey(self.seed + 1)
            rng_w, rng_a = jax.random.split(rng_sample)
            w_samples = (
                w_loc
                + w_scale
                * np.asarray(
                    jax.random.normal(rng_w, shape=(self.n_samples, w_loc.shape[0]))
                )
            )
            alpha_samples = np.exp(
                alpha_loc
                + alpha_scale
                * np.asarray(
                    jax.random.normal(rng_a, shape=(self.n_samples, alpha_loc.shape[0]))
                )
            )

        posterior_mean_w = w_samples.mean(axis=0)
        posterior_mean_alpha = alpha_samples.mean(axis=0)

        # Paper-faithful pruning: large posterior alpha_j => coefficient is
        # effectively zeroed by the prior. (Tipping 2001; MacKay 1994.)
        prune_mask = posterior_mean_alpha > self.alpha_threshold
        pruned = np.where(prune_mask, 0.0, posterior_mean_w).astype(np.float64)
        n_pruned = int(prune_mask.sum())

        return BayesianARDFitted(
            name=self.name,
            posterior_mean_weights=pruned,
            posterior_alpha=posterior_mean_alpha.astype(np.float64),
            feature_names=feature_names,
            include_interactions=self.include_interactions,
            n_pruned=n_pruned,
            inference=self.inference,
            alpha_threshold=self.alpha_threshold,
            n_warmup=self.n_warmup,
            n_samples=self.n_samples,
        )
