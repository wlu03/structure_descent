"""Tests for :mod:`src.model.po_leu` (redesign.md §8, §9.4, Appendix A).

All tests use the session-scoped ``synthetic_batch`` fixture
(B=4, J=10, K=3, d_e=768, p=26) from ``tests/conftest.py``.
"""

from __future__ import annotations

import torch

from src.model.po_leu import (
    EXPECTED_PARAM_COUNT_DEFAULT,
    POLEU,
    POLEUIntermediates,
    choice_probabilities,
    cross_entropy_loss,
)


# ---------------------------------------------------------------------------
# Parameter-count invariants (§9.4, NOTES.md reconciliation)
# ---------------------------------------------------------------------------


def test_param_count_default() -> None:
    """Orchestrator-reconciled total: 492_805 + 1_029 + 50_945 = 544_779."""
    model = POLEU()
    assert model.num_params() == EXPECTED_PARAM_COUNT_DEFAULT == 544_779


def test_param_count_uniform_salience() -> None:
    """A6 ablation: salience contributes 0, so total = heads + weight_net."""
    model = POLEU(uniform_salience=True)
    assert model.num_params() == 492_805 + 1_029


# ---------------------------------------------------------------------------
# Forward shapes (Appendix A)
# ---------------------------------------------------------------------------


def test_forward_shapes(synthetic_batch) -> None:
    """logits (B,J); A (B,J,K,M); w (B,M); U (B,J,K); S (B,J,K); V (B,J)."""
    sb = synthetic_batch
    model = POLEU()
    logits, inter = model(sb.z_d, sb.E)

    assert logits.shape == (sb.B, sb.J)
    assert inter.A.shape == (sb.B, sb.J, sb.K, 5)
    assert inter.w.shape == (sb.B, 5)
    assert inter.U.shape == (sb.B, sb.J, sb.K)
    assert inter.S.shape == (sb.B, sb.J, sb.K)
    assert inter.V.shape == (sb.B, sb.J)


def test_intermediates_dict_keys(synthetic_batch) -> None:
    """``to_dict`` returns exactly {"A","w","U","S","V"}."""
    sb = synthetic_batch
    model = POLEU()
    _, inter = model(sb.z_d, sb.E)
    d = inter.to_dict()
    assert set(d.keys()) == {"A", "w", "U", "S", "V"}
    assert isinstance(inter, POLEUIntermediates)


# ---------------------------------------------------------------------------
# Softmax invariants (§6.1, §7.1)
# ---------------------------------------------------------------------------


def test_softmax_invariants(synthetic_batch) -> None:
    """w rows sum to 1 over M; S rows sum to 1 over K (within each (B, J))."""
    sb = synthetic_batch
    model = POLEU()
    _, inter = model(sb.z_d, sb.E)

    w_sums = inter.w.sum(dim=-1)
    assert torch.allclose(w_sums, torch.ones_like(w_sums), atol=1e-6)

    s_sums = inter.S.sum(dim=-1)
    assert torch.allclose(s_sums, torch.ones_like(s_sums), atol=1e-6)


def test_choice_probabilities_sum_to_one(synthetic_batch) -> None:
    """softmax(logits) over J gives rows summing to 1."""
    sb = synthetic_batch
    model = POLEU()
    logits, _ = model(sb.z_d, sb.E)
    probs = choice_probabilities(logits)

    assert probs.shape == (sb.B, sb.J)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(sb.B), atol=1e-6)


# ---------------------------------------------------------------------------
# Temperature (§8.2, §9.5)
# ---------------------------------------------------------------------------


def test_temperature_is_not_trainable() -> None:
    """τ never appears in model.parameters(); no param named ``temperature``."""
    tau = 1.23456789
    model = POLEU(temperature=tau)

    named = list(model.named_parameters())
    # No parameter is named "temperature".
    assert all(name != "temperature" for name, _ in named)
    assert all(not name.endswith(".temperature") for name, _ in named)

    # No parameter holds the (distinctive) τ value.
    for _, param in named:
        assert not torch.any(torch.isclose(param.detach(), torch.tensor(tau))), (
            "Temperature must not appear in model.parameters()."
        )


def test_temperature_scales_logits(synthetic_batch) -> None:
    """With τ=2.0, logits equal half of τ=1.0 logits on the same (z_d, E)."""
    sb = synthetic_batch

    torch.manual_seed(42)
    model_tau1 = POLEU(temperature=1.0)
    torch.manual_seed(42)
    model_tau2 = POLEU(temperature=2.0)

    # Sanity: weights identical at init under the same seed.
    for p1, p2 in zip(model_tau1.parameters(), model_tau2.parameters()):
        assert torch.equal(p1, p2)

    with torch.no_grad():
        logits1, _ = model_tau1(sb.z_d, sb.E)
        logits2, _ = model_tau2(sb.z_d, sb.E)

    assert torch.allclose(logits2, logits1 / 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


def test_backward_finite_grads(synthetic_batch) -> None:
    """CE loss.backward() produces finite grads on every trainable param."""
    sb = synthetic_batch
    model = POLEU()
    logits, _ = model(sb.z_d, sb.E)
    loss = cross_entropy_loss(logits, sb.c_star)

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert torch.all(torch.isfinite(param.grad)), f"{name} has non-finite grad"


# ---------------------------------------------------------------------------
# Cross-entropy loss contract (§9.1)
# ---------------------------------------------------------------------------


def test_cross_entropy_omega_equiv_ones(synthetic_batch) -> None:
    """loss(logits, c*) == loss(logits, c*, ones(B)) to 1e-6."""
    sb = synthetic_batch
    model = POLEU()
    with torch.no_grad():
        logits, _ = model(sb.z_d, sb.E)

    loss_plain = cross_entropy_loss(logits, sb.c_star)
    loss_ones = cross_entropy_loss(logits, sb.c_star, torch.ones(sb.B))
    assert torch.allclose(loss_plain, loss_ones, atol=1e-6)


def test_cross_entropy_omega_weighting(synthetic_batch) -> None:
    """Non-uniform omega scales the loss as sum(omega*ℓ)/sum(omega)."""
    sb = synthetic_batch
    model = POLEU()
    with torch.no_grad():
        logits, _ = model(sb.z_d, sb.E)

    # Manually compute per-event CE.
    per_event = torch.nn.functional.cross_entropy(
        logits, sb.c_star, reduction="none"
    )

    omega = torch.tensor([0.25, 0.5, 2.0, 3.0])
    expected = (omega * per_event).sum() / omega.sum()
    got = cross_entropy_loss(logits, sb.c_star, omega)
    assert torch.allclose(got, expected, atol=1e-6)

    # Also verify it differs from the plain mean when omega is non-uniform.
    plain = cross_entropy_loss(logits, sb.c_star)
    assert not torch.allclose(got, plain, atol=1e-6)


# ---------------------------------------------------------------------------
# Ablation A6 — uniform salience
# ---------------------------------------------------------------------------


def test_uniform_salience_wiring(synthetic_batch) -> None:
    """With uniform_salience=True, S is constant = 1/K everywhere."""
    sb = synthetic_batch
    model = POLEU(uniform_salience=True)
    _, inter = model(sb.z_d, sb.E)

    expected = torch.full((sb.B, sb.J, sb.K), 1.0 / float(sb.K))
    assert torch.allclose(inter.S, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_under_seed(synthetic_batch) -> None:
    """Same seed two constructions → identical logits on the same batch."""
    sb = synthetic_batch

    torch.manual_seed(123)
    model_a = POLEU()
    torch.manual_seed(123)
    model_b = POLEU()

    with torch.no_grad():
        logits_a, _ = model_a(sb.z_d, sb.E)
        logits_b, _ = model_b(sb.z_d, sb.E)

    assert torch.equal(logits_a, logits_b)
