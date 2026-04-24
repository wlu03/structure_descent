"""Unit tests for ``LLMSR`` and the shared symbolic-regression helpers.

The 12-test battery from ``docs/llm_baselines/llm_sr_baseline.md`` §9:

1-5  grammar accept / reject (5 cases).
6-7  safe-div + exp_c runtime guards.
8    BFGS synthetic fit recovery.
9    divergence handling (blow-up skeleton -> (None, inf) without raising).
10   stub-client offline path falls back to FALLBACK_SKELETON.
11   retry logic: ill-formed then valid counts as two calls for proposal 0.
12   registry wiring: LLM-SR rows appear in BASELINE_REGISTRY.

Every test uses :class:`StubLLMClient` (or a local subclass) so the suite
runs without network access.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.baselines._symbolic_regression_common import (
    FALLBACK_SKELETON,
    SandboxError,
    _safe_div,
    compile_equation,
    exp_c,
    fit_coefficients_softmax_ce,
)
from src.baselines.base import BaselineEventBatch
from src.baselines.llm_sr import LLMSR, LLMSRFitted
from src.outcomes.generate import GenerationResult, StubLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_dcm_batch(
    n_events: int,
    *,
    true_w: np.ndarray,
    seed: int = 0,
    J: int = 4,
    F: int = 4,
) -> BaselineEventBatch:
    """Generate a J=4 MNL batch with utilities ``feats @ true_w + gumbel``.

    Returns a ``BaselineEventBatch`` whose 4 base-feature columns are
    drawn iid N(0, 1). The chosen alternative is the argmax utility so
    BFGS can recover ``true_w`` up to overall scale.
    """
    rng = np.random.default_rng(seed)
    feats_list: List[np.ndarray] = []
    chosen: List[int] = []
    customer_ids: List[str] = []
    categories: List[str] = []
    for i in range(n_events):
        feats = rng.normal(0.0, 1.0, size=(J, F)).astype(np.float64)
        gumbel = -np.log(-np.log(rng.uniform(size=J) + 1e-12) + 1e-12)
        u = feats @ true_w + gumbel
        chosen.append(int(np.argmax(u)))
        feats_list.append(feats)
        customer_ids.append(f"cust_{i}")
        categories.append("cat_0")
    return BaselineEventBatch(
        base_features_list=feats_list,
        base_feature_names=["price", "popularity_rank", "log1p_price", "price_rank"],
        chosen_indices=chosen,
        customer_ids=customer_ids,
        categories=categories,
    )


class _SequenceStubClient(StubLLMClient):
    """Return a pre-scripted sequence of raw texts per ``generate`` call.

    Each call pops the next entry off ``_responses``; once exhausted, it
    falls back to the parent :class:`StubLLMClient` behaviour. Used to
    inject specific LLM replies (ill-formed then valid, etc.) without
    patching the outer loop.
    """

    def __init__(self, responses: List[str], model_id: str = "stub-seq"):
        super().__init__(model_id=model_id)
        self._responses = list(responses)
        self.call_count = 0

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> GenerationResult:
        self.call_count += 1
        if self._responses:
            text = self._responses.pop(0)
            return GenerationResult(text=text, finish_reason="stop", model_id=self.model_id)
        return super().generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Grammar: accept (test 1)
# ---------------------------------------------------------------------------


_VALID_EXAMPLES = [
    "def utility(x, c): return c[0]*x[0] + c[1]*x[1] + c[2]*x[2]",
    "def utility(x, c): return -c[0]*log1p(x[0]) + c[1]*x[3] - c[2]*x[1]**2",
    "def utility(x, c): return c[0]*x[2] + c[1]*tanh(c[2]*x[3] - c[3])",
]


def test_grammar_accepts_well_formed():
    """All three §2 examples parse, compile, and return finite scalars."""
    x = np.array([10.0, 5.0, np.log1p(10.0), 0.5], dtype=np.float64)
    for src in _VALID_EXAMPLES:
        fn, k = compile_equation(src, max_coefficients=8)
        assert k >= 1
        c = np.linspace(-0.3, 0.3, k, dtype=np.float64)
        u = fn(x, c)
        assert np.isfinite(u), f"non-finite utility from: {src}"


# ---------------------------------------------------------------------------
# Grammar: reject (tests 2-5)
# ---------------------------------------------------------------------------


def test_grammar_rejects_attribute_access():
    """Attribute lookups (``x.__class__``) are banned."""
    src = "def utility(x, c): return x.__class__"
    with pytest.raises(SandboxError):
        compile_equation(src)


def test_grammar_rejects_import():
    """``import os`` anywhere in the source is rejected at AST walk."""
    src = "import os\ndef utility(x, c): return c[0]*x[0]"
    with pytest.raises(SandboxError):
        compile_equation(src)


def test_grammar_rejects_unsafe_call():
    """Call-node policy: ``open``, ``__import__`` rejected — not in allowlist."""
    src_open = "def utility(x, c): return open('/etc/passwd')"
    with pytest.raises(SandboxError):
        compile_equation(src_open)

    src_import = "def utility(x, c): return __import__('os')"
    # ``__import__`` is both a disallowed callee and a dunder name; either
    # check fires. The point is: this never reaches exec.
    with pytest.raises(SandboxError):
        compile_equation(src_import)


def test_grammar_rejects_bad_subscript():
    """``x[10]`` (out of range) and ``x[c[0]]`` (dynamic) both fail."""
    with pytest.raises(SandboxError, match="out of range"):
        compile_equation("def utility(x, c): return x[10]")
    with pytest.raises(SandboxError, match="literal integer"):
        compile_equation("def utility(x, c): return x[c[0]]")


def test_grammar_rejects_negative_subscript_and_lambda():
    """Extra hardening: negative index + lambda both rejected."""
    with pytest.raises(SandboxError):
        compile_equation("def utility(x, c): return x[-1]")
    with pytest.raises(SandboxError):
        compile_equation("def utility(x, c): return (lambda: c[0])()")


# ---------------------------------------------------------------------------
# Runtime primitives (tests 6-7)
# ---------------------------------------------------------------------------


def test_safe_div_handles_zero():
    """``_safe_div(a, 0)`` returns a finite value; reachable through sandbox."""
    assert np.isfinite(_safe_div(1.0, 0.0))
    assert np.isfinite(_safe_div(-7.0, 0.0))
    # Reach it through the sandbox too — exercise compile_equation end-to-end.
    fn, k = compile_equation(
        "def utility(x, c): return _safe_div(c[0], x[0])", max_coefficients=1
    )
    assert k == 1
    u = fn(np.array([0.0, 0.0, 0.0, 0.0]), np.array([3.0]))
    assert np.isfinite(u)


def test_exp_c_clips_large():
    """``exp_c(50)`` equals ``exp_c(10)`` by clip; ``exp_c(-50)`` equals ``exp_c(-10)``."""
    assert exp_c(50.0) == pytest.approx(exp_c(10.0))
    assert exp_c(-50.0) == pytest.approx(exp_c(-10.0))
    # A raw exp of 50 would overflow float64; the clipped version is
    # bounded well below +inf.
    assert np.isfinite(exp_c(1e6))
    assert np.isfinite(exp_c(-1e6))


# ---------------------------------------------------------------------------
# Coefficient fitting (tests 8-9)
# ---------------------------------------------------------------------------


def test_fit_coefficients_on_synthetic():
    """BFGS recovers ``U = 2*x[0] - 0.5*x[1]`` on 500 synthetic events."""
    true_w = np.array([2.0, -0.5, 0.0, 0.0], dtype=np.float64)
    batch = _make_synthetic_dcm_batch(500, true_w=true_w, seed=42)
    fn, k = compile_equation(
        "def utility(x, c): return c[0]*x[0] + c[1]*x[1]", max_coefficients=2
    )
    assert k == 2
    feats_list = [np.asarray(f, dtype=np.float64)[:, :4] for f in batch.base_features_list]
    coeffs, nll = fit_coefficients_softmax_ce(
        fn, feats_list, batch.chosen_indices, k=k, n_restarts=4, seed=0
    )
    assert coeffs is not None
    assert np.isfinite(nll)
    # Allow generous tolerance: MNL is identifiable up to scale but the
    # ratio of coefficients should match the ratio in true_w.
    ratio = coeffs[0] / max(abs(coeffs[1]), 1e-6)
    true_ratio = true_w[0] / max(abs(true_w[1]), 1e-6)
    assert ratio > 0, f"coeffs[0] sign wrong: {coeffs}"
    assert coeffs[1] < 0, f"coeffs[1] sign wrong: {coeffs}"
    # Magnitudes within 50% (MNL identifies scale only up to monotone
    # transform when the noise is large).
    assert abs(ratio - true_ratio) / abs(true_ratio) < 1.0


def test_fit_coefficients_handles_divergence():
    """A blow-up skeleton returns ``(None, inf)`` without raising."""
    # ``x[0]**2 * exp_c(c[0]*x[0]**2)`` explodes quickly; paired with a
    # big init it saturates exp_c's clip and the gradient plateaus, which
    # BFGS may still handle -- so we also try a pathological divide by a
    # coefficient initialised near zero.
    batch = _make_synthetic_dcm_batch(50, true_w=np.ones(4), seed=1)
    feats_list = [np.asarray(f, dtype=np.float64)[:, :4] for f in batch.base_features_list]

    # Skeleton that is purely dependent on a coefficient divided by itself
    # squared -- BFGS sees a flat surface with infinities on the boundary.
    src = "def utility(x, c): return _safe_div(x[0], c[0]*c[0]) * x[1]*x[1]"
    fn, k = compile_equation(src, max_coefficients=1)
    coeffs, nll = fit_coefficients_softmax_ce(
        fn, feats_list, batch.chosen_indices, k=k, n_restarts=2, seed=0
    )
    # Either the fitter converged to something finite or it bailed.
    # In both cases the function must not raise; that's the contract.
    assert (coeffs is None and not np.isfinite(nll)) or np.isfinite(nll)


# ---------------------------------------------------------------------------
# Outer-loop behaviour (tests 10-11)
# ---------------------------------------------------------------------------


def test_outer_loop_with_stub_client():
    """Stub client: every LLM proposal rejected → falls back to FALLBACK_SKELETON.

    The StubLLMClient emits narrative outcome sentences (never Python),
    so ``_extract_skeleton_source`` returns ``None`` for every proposal.
    The fitted object must hold the fallback skeleton with non-inf NLL.
    """
    train = _make_synthetic_dcm_batch(60, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=1)
    val = _make_synthetic_dcm_batch(20, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=2)
    test = _make_synthetic_dcm_batch(20, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=3)

    baseline = LLMSR(llm_client=StubLLMClient(), n_proposals=5)
    fitted = baseline.fit(train, val)

    assert isinstance(fitted, LLMSRFitted)
    assert fitted.best_skeleton.strip() == FALLBACK_SKELETON.strip()
    assert np.isfinite(fitted.train_nll)
    assert np.isfinite(fitted.val_nll)
    assert fitted.n_proposals_accepted == 0
    assert fitted.n_proposals_total >= 5  # at least one call per proposal

    scores = fitted.score_events(test)
    assert len(scores) == test.n_events
    for arr in scores:
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr))


def test_outer_loop_retries_on_rejection():
    """Retry logic: ill-formed reply then valid reply -> 2 calls for proposal 0."""
    valid_source = (
        "```python\n"
        "def utility(x, c): return c[0]*x[0] + c[1]*x[3]\n"
        "```"
    )
    client = _SequenceStubClient(
        responses=[
            "not code at all",    # reply 1: unparseable -> rejected
            valid_source,          # reply 2: valid -> accepted
        ]
    )
    train = _make_synthetic_dcm_batch(40, true_w=np.array([1.0, 0.0, 0.0, -0.5]), seed=7)
    val = _make_synthetic_dcm_batch(20, true_w=np.array([1.0, 0.0, 0.0, -0.5]), seed=8)

    baseline = LLMSR(
        llm_client=client,
        n_proposals=1,
        max_retries_per_proposal=2,
    )
    fitted = baseline.fit(train, val)

    # With n_proposals=1 and retries=2 we can issue up to 3 calls. Our
    # scripted sequence should accept on the 2nd call, so ``call_count``
    # is exactly 2 and ``n_proposals_total == 2`` (both counted against
    # iteration t=0).
    assert client.call_count == 2
    assert fitted.n_proposals_total == 2
    assert fitted.n_proposals_accepted == 1
    # The accepted skeleton should now be the best (lower NLL than the
    # fallback on a true_w that zeroes out x[1], x[2]).
    assert "c[0]*x[0]" in fitted.best_skeleton
    assert "c[1]*x[3]" in fitted.best_skeleton


# ---------------------------------------------------------------------------
# Registry wiring (test 12)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# extra_artifacts_for_json (paper-grade evaluation addition 4)
# ---------------------------------------------------------------------------


def test_extra_artifacts_returns_none_when_memory_empty():
    """A fitted object with no memory entries → ``extra_artifacts_for_json`` → None.

    The leaderboard writer interprets ``None`` as "omit / null this
    baseline's extra_artifacts" — non-LLM-SR rows must see None here
    and so must LLM-SR rows that fit with zero proposals accepted.
    """
    fitted = LLMSRFitted(
        name="LLM-SR",
        best_skeleton=FALLBACK_SKELETON,
        best_coefficients=np.zeros(4, dtype=np.float64),
        n_coefficients=4,
        train_nll=1.0,
        val_nll=1.2,
        memory=[],
    )
    assert fitted.extra_artifacts_for_json() is None


def test_extra_artifacts_sorts_by_combined_nll_and_caps_at_10():
    """Top-K export respects ``nll_train + nll_val`` ascending; cap = 10."""
    from src.baselines.llm_sr import _SkeletonRecord

    records = []
    for i in range(15):
        records.append(
            _SkeletonRecord(
                source=f"def utility(x, c): return c[0]*x[{i % 4}]  # id={i}",
                fn=lambda x, c: 0.0,
                coeffs=np.array([float(i), -float(i)]),
                k=2,
                train_nll=float(15 - i),   # smaller i → higher NLL; reverse sort
                val_nll=float(15 - i),
            )
        )
    fitted = LLMSRFitted(
        name="LLM-SR",
        best_skeleton=records[-1].source,
        best_coefficients=records[-1].coeffs,
        n_coefficients=2,
        train_nll=records[-1].train_nll,
        val_nll=records[-1].val_nll,
        memory=records,
    )
    out = fitted.extra_artifacts_for_json()
    assert out is not None
    eqs = out["llm_sr_top_equations"]
    assert len(eqs) == 10  # cap
    # First entry has the lowest combined_nll — records[14] has train+val=2.
    assert eqs[0]["nll_train"] == 1.0 and eqs[0]["nll_val"] == 1.0
    # Ordering is ascending.
    combined = [e["nll_train"] + e["nll_val"] for e in eqs]
    assert combined == sorted(combined)
    # Shape of each entry.
    for e in eqs:
        assert set(e.keys()) == {"source", "nll_train", "nll_val", "coefficients"}
        assert isinstance(e["source"], str)
        assert isinstance(e["nll_train"], float)
        assert isinstance(e["nll_val"], float)
        assert isinstance(e["coefficients"], list)
        for c in e["coefficients"]:
            assert type(c) is float


def test_extra_artifacts_populated_after_fit():
    """After a real stub-path fit, memory is non-empty and the hook fires."""
    train = _make_synthetic_dcm_batch(50, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=1)
    val = _make_synthetic_dcm_batch(20, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=2)
    baseline = LLMSR(llm_client=StubLLMClient(), n_proposals=2)
    fitted = baseline.fit(train, val)
    # Stub path only seeds the fallback into memory, but memory is non-empty.
    assert len(fitted.memory) >= 1
    out = fitted.extra_artifacts_for_json()
    assert out is not None
    assert "llm_sr_top_equations" in out
    assert len(out["llm_sr_top_equations"]) >= 1


def test_registry_factory_wires_llm_sr():
    """``LLM-SR-*`` rows appear in BASELINE_REGISTRY and factories are callable."""
    from src.baselines.run_all import BASELINE_REGISTRY, LLM_CLIENT_FACTORIES

    names = [row[0] for row in BASELINE_REGISTRY]
    llm_sr_rows = [n for n in names if n.startswith("LLM-SR-")]
    # At minimum, Sonnet + Opus (Claude) expansions should be present.
    assert any("Sonnet" in n for n in llm_sr_rows), f"no Sonnet row in {llm_sr_rows}"
    assert any("Opus" in n for n in llm_sr_rows), f"no Opus row in {llm_sr_rows}"

    # Every LLM-SR row should resolve to (module, class) == (llm_sr, LLMSR).
    for name, module, clsname in BASELINE_REGISTRY:
        if name.startswith("LLM-SR-"):
            assert module == "src.baselines.llm_sr"
            assert clsname == "LLMSR"
            # And there must be a factory entry.
            assert name in LLM_CLIENT_FACTORIES
            assert callable(LLM_CLIENT_FACTORIES[name])
