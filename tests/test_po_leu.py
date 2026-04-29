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
    """Group-2 total: 492_805 + 1_029 + 51_465 = 545_299.

    Salience grew by 520 parameters (+512 from the d_cat=8 widening of
    fc1, +8 from the 1-row category embedding).
    """
    model = POLEU()
    assert model.num_params() == EXPECTED_PARAM_COUNT_DEFAULT == 545_299


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


# ---------------------------------------------------------------------------
# Strategy B — Sifringer feature-partition tabular residual
# ---------------------------------------------------------------------------
#
# Four contract tests covering:
#   A.1  zero-init residual produces logits identical to baseline.
#   A.2  non-zero x_tab + simple loss propagates gradient to beta_tab.
#   A.3  backward-compat: forward(z_d, E) with no x_tab still works.
#   A.4  shape-contract violations on x_tab raise a clear ValueError.


def _make_x_tab(B: int, J: int, F: int = 3, seed: int = 7) -> torch.Tensor:
    """Synthesize a deterministic (B, J, F) tabular feature tensor."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, J, F, generator=g)


def test_residual_zero_init_matches_baseline(synthetic_batch) -> None:
    """A.1: with β=0, forward(z_d, E, x_tab) == forward(z_d, E) bitwise.

    Critical interpretability invariant: enabling the residual must not
    perturb initial logits, otherwise downstream ablation comparisons
    would conflate "structural change at init" with "trained effect".
    """
    sb = synthetic_batch
    x_tab = _make_x_tab(sb.B, sb.J, F=3)

    torch.manual_seed(999)
    base = POLEU()
    torch.manual_seed(999)
    residual = POLEU(
        tabular_residual_enabled=True,
        tabular_features=("price", "log1p_price", "price_rank"),
    )

    # β must be zero at init (this is the invariant the test protects).
    assert torch.equal(residual.beta_tab.detach(), torch.zeros(3))

    with torch.no_grad():
        logits_base, inter_base = base(sb.z_d, sb.E)
        logits_res, inter_res = residual(sb.z_d, sb.E, x_tab)

    assert torch.allclose(logits_base, logits_res, atol=0.0, rtol=0.0), (
        "Zero-init residual must produce bit-identical logits."
    )
    assert inter_base.V_residual is None
    assert inter_res.V_residual is not None
    assert torch.equal(inter_res.V_residual, torch.zeros_like(inter_res.V))
    assert torch.equal(inter_res.V_total, inter_res.V)


def test_residual_gradient_propagates_to_beta(synthetic_batch) -> None:
    """A.2: a nonzero x_tab + CE loss yields a nonzero gradient on β."""
    sb = synthetic_batch
    x_tab = _make_x_tab(sb.B, sb.J, F=3)

    torch.manual_seed(2024)
    model = POLEU(tabular_residual_enabled=True)
    logits, _ = model(sb.z_d, sb.E, x_tab)
    loss = cross_entropy_loss(logits, sb.c_star)
    loss.backward()

    assert model.beta_tab.grad is not None
    assert torch.all(torch.isfinite(model.beta_tab.grad))
    assert model.beta_tab.grad.norm().item() > 0.0, (
        "β must receive nonzero gradient from a nontrivial loss."
    )


def test_residual_backward_compat_no_x_tab(synthetic_batch) -> None:
    """A.3: forward(z_d, E) without x_tab still works on a residual model.

    Plumbed callers that haven't yet started passing x_tab must continue
    to function — the residual is opt-in per call as well as per
    constructor flag.
    """
    sb = synthetic_batch

    torch.manual_seed(2024)
    model = POLEU(tabular_residual_enabled=True)
    logits, inter = model(sb.z_d, sb.E)  # no x_tab arg
    assert logits.shape == (sb.B, sb.J)
    assert inter.V_residual is None
    assert inter.V_total is None

    # And explicit None should match the no-arg form.
    torch.manual_seed(2024)
    model_b = POLEU(tabular_residual_enabled=True)
    logits_explicit, _ = model_b(sb.z_d, sb.E, None)
    assert torch.equal(logits, logits_explicit)


def test_residual_shape_contract_raises(synthetic_batch) -> None:
    """A.4: wrong-shape x_tab raises ValueError; passing x_tab when the
    residual is disabled also raises (config-drift trap)."""
    sb = synthetic_batch
    model_off = POLEU()
    model_on = POLEU(
        tabular_residual_enabled=True,
        tabular_features=("price", "log1p_price", "price_rank"),
    )

    # x_tab supplied when residual disabled -> error
    x_tab_ok = _make_x_tab(sb.B, sb.J, F=3)
    try:
        model_off(sb.z_d, sb.E, x_tab_ok)
    except ValueError as exc:
        assert "tabular_residual is disabled" in str(exc)
    else:
        raise AssertionError("expected ValueError when residual disabled")

    # Wrong feature dim (F=4 when constructor expects 3).
    x_tab_wrong_F = _make_x_tab(sb.B, sb.J, F=4)
    try:
        model_on(sb.z_d, sb.E, x_tab_wrong_F)
    except ValueError as exc:
        assert "x_tab shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on wrong F")

    # Wrong batch dim.
    x_tab_wrong_B = _make_x_tab(sb.B + 1, sb.J, F=3)
    try:
        model_on(sb.z_d, sb.E, x_tab_wrong_B)
    except ValueError as exc:
        assert "x_tab shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on wrong B")

    # Wrong J.
    x_tab_wrong_J = _make_x_tab(sb.B, sb.J + 1, F=3)
    try:
        model_on(sb.z_d, sb.E, x_tab_wrong_J)
    except ValueError as exc:
        assert "x_tab shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on wrong J")

    # 2-D instead of 3-D.
    try:
        model_on(sb.z_d, sb.E, torch.zeros(sb.B, sb.J))
    except ValueError as exc:
        assert "x_tab shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on 2-D x_tab")


def test_residual_param_count_increment() -> None:
    """Enabling residual adds exactly F trainable parameters."""
    base = POLEU()
    residual = POLEU(
        tabular_residual_enabled=True,
        tabular_features=("price", "log1p_price", "price_rank"),
    )
    assert residual.num_params() - base.num_params() == 3


def test_residual_excluded_from_weight_l2(synthetic_batch) -> None:
    """β must NOT be touched by the §9.2 weight-net L2 penalty.

    Sanity check: ``weight_net_l2(model.weight_net)`` walks the
    weight_net submodule only, so β (a model-level Parameter) is never
    in scope. This guards against a future refactor that walks all
    parameters by name and forgets to exclude β.
    """
    from src.train.regularizers import weight_net_l2

    model = POLEU(tabular_residual_enabled=True)
    # Set β to a large value and confirm weight_net_l2 is unaffected.
    with torch.no_grad():
        model.beta_tab.fill_(100.0)
    base_l2 = weight_net_l2(model.weight_net).item()
    with torch.no_grad():
        model.beta_tab.fill_(0.0)
    zeroed_l2 = weight_net_l2(model.weight_net).item()
    assert base_l2 == zeroed_l2, (
        "weight_net_l2 must be independent of beta_tab; got differing values "
        f"({base_l2} vs {zeroed_l2})."
    )
    assert "beta_tab" in dict(model.named_parameters())
    assert "beta_tab" not in dict(model.weight_net.named_parameters())


def test_residual_standardization_buffers(synthetic_batch) -> None:
    """``set_tabular_feature_stats`` installs train-set mean/std.

    After fit, with a fixed β!=0 the residual contribution should equal
    ``((x_tab - mean) / std) @ β`` element-wise.
    """
    sb = synthetic_batch
    x_tab = _make_x_tab(sb.B, sb.J, F=3, seed=11)

    model = POLEU(tabular_residual_enabled=True)
    flat = x_tab.reshape(-1, 3)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    model.set_tabular_feature_stats(mean, std)

    # Manually set β to a known nonzero vector.
    with torch.no_grad():
        model.beta_tab.copy_(torch.tensor([0.1, -0.2, 0.3]))

    with torch.no_grad():
        _, inter = model(sb.z_d, sb.E, x_tab)

    expected_std = (x_tab - mean) / std
    expected_residual = (expected_std * model.beta_tab).sum(dim=-1)
    assert torch.allclose(inter.V_residual, expected_residual, atol=1e-6)


def test_residual_intermediates_to_dict(synthetic_batch) -> None:
    """``to_dict`` includes V_residual + V_total only when populated."""
    sb = synthetic_batch
    x_tab = _make_x_tab(sb.B, sb.J, F=3, seed=22)

    base = POLEU()
    _, inter_base = base(sb.z_d, sb.E)
    keys_base = set(inter_base.to_dict().keys())
    assert keys_base == {"A", "w", "U", "S", "V"}

    residual = POLEU(tabular_residual_enabled=True)
    _, inter_res = residual(sb.z_d, sb.E, x_tab)
    keys_res = set(inter_res.to_dict().keys())
    assert keys_res == {"A", "w", "U", "S", "V", "V_residual", "V_total"}



# ---------------------------------------------------------------------------
# Group-2 SalienceNet category injection plumbing
# ---------------------------------------------------------------------------


def test_forward_accepts_c_none(synthetic_batch) -> None:
    """Legacy callers that don't pass c still work."""
    sb = synthetic_batch
    model = POLEU()
    logits, inter = model(sb.z_d, sb.E)
    assert logits.shape == (sb.B, sb.J)
    logits2, _ = model(sb.z_d, sb.E, c=None)
    assert torch.equal(logits, logits2)


def test_forward_with_int64_c(synthetic_batch) -> None:
    """Per-event category code threads through SalienceNet."""
    sb = synthetic_batch
    model = POLEU(n_categories=5, d_cat=8)
    c = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    logits, inter = model(sb.z_d, sb.E, c=c)
    assert logits.shape == (sb.B, sb.J)
    assert inter.S.shape == (sb.B, sb.J, sb.K)


def test_forward_c_changes_output(synthetic_batch) -> None:
    """Different category codes produce different logits given same (E, z_d)."""
    sb = synthetic_batch
    torch.manual_seed(0)
    model = POLEU(n_categories=4, d_cat=8)
    E_pair = sb.E[:1].repeat(2, 1, 1, 1)
    z_pair = sb.z_d[:1].repeat(2, 1)
    c0 = torch.tensor([0, 0], dtype=torch.long)
    c1 = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        logits_a, _ = model(z_pair, E_pair, c=c0)
        logits_b, _ = model(z_pair, E_pair, c=c1)
    assert torch.equal(logits_a[0], logits_b[0])
    assert not torch.allclose(logits_a[1], logits_b[1], atol=1e-7)
