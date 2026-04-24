"""Unit tests for the LaSR baseline (design doc §9 claims 1-8).

Coverage map:

* Seed concepts compile under ``compile_equation``.
* ``canonicalize`` + ``extract_subexpression_candidates`` stability.
* Promotion fires at ``freq >= 3`` (auto) and under explicit LLM
  nomination (freq >= 1).
* Cap + TTL eviction on :class:`ConceptLibrary`.
* Library transfer: a concept found on customer A is available when
  fitting customer B.
* Stub-fallback path with an offline client (LaSR returns the
  ``FALLBACK_SKELETON`` record).
* Registry wiring: all 5 LaSR rows visible in ``BASELINE_REGISTRY``.
* End-to-end fit on ``_make_synthetic_dcm_batch`` with a small budget.

Every test uses :class:`StubLLMClient` (or a scripted subclass) so the
suite runs without network access.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.baselines._symbolic_regression_common import (
    FALLBACK_SKELETON,
    SEED_CONCEPTS,
    canonicalize,
    compile_equation,
    extract_subexpression_candidates,
)
from src.baselines.base import BaselineEventBatch
from src.baselines.lasr import (
    Concept,
    ConceptLibrary,
    LaSR,
    LaSRFitted,
    _is_trivial_concept,
    _seed_concepts_to_library,
)
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
    """MNL fixture identical to the one in ``test_llm_sr``."""
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
    """Replay a pre-scripted sequence of raw texts per ``generate`` call."""

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
            return GenerationResult(
                text=text, finish_reason="stop", model_id=self.model_id
            )
        return super().generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# 1. Seed concepts compile under compile_equation
# ---------------------------------------------------------------------------


def test_seed_concepts_compile_and_evaluate():
    """Every entry in ``SEED_CONCEPTS`` compiles and returns a finite float."""
    x = np.array([25.0, 0.7, np.log1p(25.0), 0.3], dtype=np.float64)
    for entry in SEED_CONCEPTS:
        src = f"def utility(x, c): return {entry['body']}"
        fn, k = compile_equation(src, max_coefficients=2)
        assert k >= 1
        c = np.array([0.3, 0.2], dtype=np.float64)[: max(k, 1)]
        u = fn(x, c)
        assert np.isfinite(u), f"non-finite utility for seed {entry['name']}"


def test_seed_concepts_to_library_roundtrips():
    """``_seed_concepts_to_library`` produces frozen ``Concept`` rows."""
    concepts = _seed_concepts_to_library()
    assert len(concepts) == len(SEED_CONCEPTS)
    for c in concepts:
        assert isinstance(c, Concept)
        with pytest.raises(Exception):
            # frozen=True -> setattr raises
            c.usage_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. canonicalize / extract_subexpression_candidates stability
# ---------------------------------------------------------------------------


def test_canonicalize_collapses_variable_renaming():
    """``c[0]*x[0] + c[1]*x[1]`` and ``c[3]*x[2] + c[5]*x[0]`` collapse."""
    import ast as _ast
    src_a = "def utility(x, c): return c[0]*x[0] + c[1]*x[1]"
    src_b = "def utility(x, c): return c[3]*x[2] + c[5]*x[0]"

    def _body(src: str) -> str:
        t = _ast.parse(src, mode="exec")
        for node in _ast.walk(t):
            if isinstance(node, _ast.Return):
                return canonicalize(node.value)
        raise RuntimeError("no return")

    assert _body(src_a) == _body(src_b)


def test_extract_subexpression_candidates_returns_stable_canonicals():
    """Identical sub-expressions across two survivors hash to the same key."""
    src1 = "def utility(x, c): return -c[0]*x[0] + c[1]*x[3]"
    src2 = "def utility(x, c): return -c[2]*x[1] + c[3]*x[3]"
    subs1 = extract_subexpression_candidates(src1)
    subs2 = extract_subexpression_candidates(src2)
    # ``-c[0]*x[0]`` and ``-c[2]*x[1]`` canonicalise identically.
    canonicals1 = {s.canonical for s in subs1}
    canonicals2 = {s.canonical for s in subs2}
    assert canonicals1 & canonicals2


# ---------------------------------------------------------------------------
# 3. Promotion: frequency >= 3 AND explicit LLM nomination
# ---------------------------------------------------------------------------


def _survivor_from(src: str):
    """Fit a minimal 1-event "survivor" so promotion logic has data.

    The promotion scanner only reads ``rec.source`` from each survivor,
    so we don't need real coefficients / NLLs — just the AST.
    """
    # Build a lightweight shim: promotion only touches ``source``.
    from src.baselines.lasr import _SurvivorRecord

    return _SurvivorRecord(
        source=src,
        fn=lambda x, c: 0.0,
        coeffs=np.zeros(1),
        k=1,
        train_nll=1.0,
        val_nll=1.0,
    )


def test_promotion_fires_at_frequency_threshold():
    """3 survivors sharing a non-trivial sub-expression → auto-promoted."""
    library = ConceptLibrary(max_size=20, ttl=5, promotion_threshold=3)
    library.seed([])  # empty library

    # Synthetic baseline instance only to get _update_library as a method.
    lasr = LaSR(llm_client=StubLLMClient(), n_iters=1, proposals_per_iter=1)
    # The shared sub-expression is ``log1p(x[1]) * c[0]`` — depth 3, non-
    # trivial (has a function call) so the collapse-guard leaves it alone.
    survivors = [
        _survivor_from("def utility(x, c): return log1p(x[1])*c[0] + c[1]*x[2]"),
        _survivor_from("def utility(x, c): return log1p(x[1])*c[0] + c[1]*x[3]"),
        _survivor_from("def utility(x, c): return log1p(x[1])*c[0] - c[1]*x[0]"),
    ]
    lasr._update_library(
        library,
        survivors,
        nominations=[],
        iter_index=1,
        protect=set(),
    )
    names = [c.name for c in library.as_list()]
    bodies = [c.source for c in library.as_list()]
    assert any("log1p(x[0]) * c[0]" in b for b in bodies), (
        f"no log1p-concept promoted; got {bodies}"
    )
    assert any(n.startswith("concept_1_") for n in names), names


def test_promotion_fires_below_threshold_only_with_nomination():
    """2 survivors would not auto-promote, but NOMINATE: admits the concept."""
    library = ConceptLibrary(max_size=20, ttl=5, promotion_threshold=3)
    library.seed([])
    lasr = LaSR(llm_client=StubLLMClient(), n_iters=1, proposals_per_iter=1)
    # Non-trivial shared sub-expression (log1p on a feature).
    survivors = [
        _survivor_from("def utility(x, c): return log1p(x[2])*c[0] + c[1]*x[3]"),
        _survivor_from("def utility(x, c): return log1p(x[2])*c[0] + c[1]*x[0]"),
    ]
    # Without nomination: no promotion at freq=2 < threshold=3.
    lasr._update_library(
        library, survivors, nominations=[], iter_index=1, protect=set()
    )
    assert len(library) == 0
    # With nomination: admitted even at freq=2 < threshold=3.
    lasr._update_library(
        library,
        survivors,
        nominations=["my_log_term"],
        iter_index=2,
        protect=set(),
    )
    assert "my_log_term" in library


def test_promotion_filters_trivial_concepts():
    """A canonical form matching the bare ``c[0]*x[0]`` identity is filtered.

    The collapse-guard per design doc §11 keeps the library from
    degenerating into a single linear-scaling concept.
    """
    library = ConceptLibrary(max_size=20, ttl=5, promotion_threshold=1)
    library.seed([])
    lasr = LaSR(llm_client=StubLLMClient(), n_iters=1, proposals_per_iter=1)
    survivors = [
        _survivor_from("def utility(x, c): return c[0]*x[0]"),
    ]
    lasr._update_library(
        library, survivors, nominations=[], iter_index=1, protect=set()
    )
    for c in library.as_list():
        assert not _is_trivial_concept(c), c


# ---------------------------------------------------------------------------
# 4. LRU + TTL + cap eviction on ConceptLibrary
# ---------------------------------------------------------------------------


def test_library_cap_evicts_lowest_usage():
    """Fill library past cap; the min-usage entry is evicted."""
    library = ConceptLibrary(max_size=3, ttl=5, promotion_threshold=3)
    library.add(
        Concept(name="c0", source="x[0]", nl_summary="a", usage_count=10, discovered_at=0)
    )
    library.add(
        Concept(name="c1", source="x[1]", nl_summary="b", usage_count=9, discovered_at=1)
    )
    # c2 is the lowest-usage incumbent.
    library.add(
        Concept(name="c2", source="x[2]", nl_summary="c", usage_count=2, discovered_at=2)
    )
    # Now add a 4th with usage above c2 — c2 must be evicted.
    library.add(
        Concept(name="c3", source="x[3]", nl_summary="d", usage_count=8, discovered_at=3)
    )
    evicted = library.tick()
    assert "c2" in evicted, f"expected lowest-usage c2 evicted; got {evicted}"
    assert len(library) == 3


def test_library_cap_tiebreak_by_discovery_order():
    """Two concepts with equal usage: the older ``discovered_at`` is evicted."""
    library = ConceptLibrary(max_size=2, ttl=5, promotion_threshold=3)
    library.add(
        Concept(name="old", source="x[0]", nl_summary="a", usage_count=1, discovered_at=1)
    )
    library.add(
        Concept(name="new", source="x[1]", nl_summary="b", usage_count=1, discovered_at=5)
    )
    library.add(
        Concept(name="fresh", source="x[2]", nl_summary="c", usage_count=2, discovered_at=6)
    )
    evicted = library.tick()
    assert "old" in evicted, f"expected oldest tied-usage evicted; got {evicted}"


def test_library_ttl_evicts_zero_usage():
    """A concept with 0 usage for ``ttl`` consecutive ticks is evicted."""
    library = ConceptLibrary(max_size=20, ttl=2, promotion_threshold=3)
    library.add(
        Concept(name="lonely", source="x[0]", nl_summary="zzz", usage_count=0, discovered_at=0)
    )
    library.add(
        Concept(name="used", source="x[1]", nl_summary="live", usage_count=3, discovered_at=0)
    )
    # Two ticks of zero usage for "lonely".
    for _ in range(2):
        library.update_usage({"lonely": 0, "used": 5})
    evicted = library.tick()
    assert "lonely" in evicted
    assert "used" in library


def test_library_cap_respects_size_20_by_default():
    """Default cap of 20 is enforced when proliferating."""
    library = ConceptLibrary()  # defaults: cap=20 ttl=5 thr=3
    for i in range(30):
        library.add(
            Concept(
                name=f"c{i}",
                source=f"x[{i % 4}]",
                nl_summary="x",
                usage_count=i,
                discovered_at=i,
            )
        )
    library.tick()
    assert len(library) == 20


# ---------------------------------------------------------------------------
# 5. Library transfer across customers
# ---------------------------------------------------------------------------


def test_library_transfer_available_on_second_fit():
    """A concept seeded via ``transferred_library`` is visible in round 2."""
    transferred = [
        Concept(
            name="ported_price_term",
            source="-c[0] * x[0] * x[0]",
            nl_summary="ported from customer A",
            usage_count=7,
            discovered_at=1,
            n_coeffs=1,
        )
    ]
    train = _make_synthetic_dcm_batch(40, true_w=np.array([1.0, 0.0, 0.0, -0.5]), seed=7)
    val = _make_synthetic_dcm_batch(20, true_w=np.array([1.0, 0.0, 0.0, -0.5]), seed=8)

    baseline = LaSR(
        llm_client=StubLLMClient(),
        n_iters=1,
        proposals_per_iter=1,
        transferred_library=transferred,
    )
    fitted = baseline.fit(train, val)
    names = [c.name for c in fitted.final_concept_library]
    assert "ported_price_term" in names


# ---------------------------------------------------------------------------
# 6. Stub-fallback path
# ---------------------------------------------------------------------------


def test_stub_fallback_returns_fallback_skeleton():
    """Stub LLM rejects every proposal -> fitted holds ``FALLBACK_SKELETON``."""
    train = _make_synthetic_dcm_batch(40, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=1)
    val = _make_synthetic_dcm_batch(20, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=2)
    baseline = LaSR(
        llm_client=StubLLMClient(),
        n_iters=2,
        proposals_per_iter=2,
    )
    fitted = baseline.fit(train, val)
    assert isinstance(fitted, LaSRFitted)
    assert fitted.best_equation.strip() == FALLBACK_SKELETON.strip()
    assert np.isfinite(fitted.val_nll)
    assert np.isfinite(fitted.train_nll)
    assert fitted.n_proposals_accepted == 0
    assert fitted.n_proposals_total >= 4
    # Score contract.
    test = _make_synthetic_dcm_batch(10, true_w=np.array([1.0, -0.5, 0.3, -0.1]), seed=3)
    scores = fitted.score_events(test)
    assert len(scores) == test.n_events
    for arr in scores:
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# 7. Registry wiring
# ---------------------------------------------------------------------------


def test_registry_contains_all_five_lasr_rows():
    """``BASELINE_REGISTRY`` contains five ``LaSR-<model>`` rows with factories."""
    from src.baselines.run_all import BASELINE_REGISTRY, LLM_CLIENT_FACTORIES

    names = [row[0] for row in BASELINE_REGISTRY]
    lasr_rows = [n for n in names if n.startswith("LaSR-")]
    assert len(lasr_rows) == 5, f"expected 5 LaSR rows, got {lasr_rows}"
    for n in lasr_rows:
        assert n in LLM_CLIENT_FACTORIES
        assert callable(LLM_CLIENT_FACTORIES[n])
    for name, module, clsname in BASELINE_REGISTRY:
        if name.startswith("LaSR-"):
            assert module == "src.baselines.lasr"
            assert clsname == "LaSR"


# ---------------------------------------------------------------------------
# 8. End-to-end fit + structured reply acceptance
# ---------------------------------------------------------------------------


def test_end_to_end_fit_with_scripted_proposals():
    """Scripted valid proposal is accepted + appears in equation memory.

    The proposed 2-feature equation may not beat the 4-feature fallback
    on small finite samples (the fallback's extra slots just fit noise),
    so we assert on acceptance + equation-memory membership rather than
    best-equation identity.
    """
    valid_source = (
        "```python\n"
        "def utility(x, c): return c[0]*x[0] + c[1]*x[3]\n"
        "```"
    )
    client = _SequenceStubClient(responses=[valid_source])
    train = _make_synthetic_dcm_batch(200, true_w=np.array([1.0, 0.0, 0.0, -0.5]), seed=11)
    val = _make_synthetic_dcm_batch(60, true_w=np.array([1.0, 0.0, 0.0, -0.5]), seed=12)
    baseline = LaSR(
        llm_client=client,
        n_iters=1,
        proposals_per_iter=1,
        max_retries_per_proposal=0,
    )
    fitted = baseline.fit(train, val)
    assert fitted.n_proposals_accepted >= 1
    assert np.isfinite(fitted.val_nll)
    # The accepted proposal must appear in the equation memory.
    memory_sources = [src for (src, _) in fitted.equation_memory]
    assert any(
        "c[0]*x[0]" in src and "c[1]*x[3]" in src for src in memory_sources
    ), f"scripted proposal missing from memory: {memory_sources}"


# ---------------------------------------------------------------------------
# 9. extra_artifacts_for_json (paper-grade evaluation addition 4)
# ---------------------------------------------------------------------------


def test_extra_artifacts_returns_none_when_both_empty():
    """No records AND no library → hook returns None."""
    fitted = LaSRFitted(
        name="LaSR",
        best_equation=FALLBACK_SKELETON,
        best_coefficients=np.zeros(4, dtype=np.float64),
        n_coefficients=4,
        final_concept_library=[],
        equation_memory=[],
        equation_records=[],
    )
    assert fitted.extra_artifacts_for_json() is None


def test_extra_artifacts_carries_top_equations_and_library():
    """Both ``llm_sr_top_equations`` and ``lasr_final_concept_library`` are present."""
    from src.baselines.lasr import _SurvivorRecord

    # Six records with varying combined NLL so sort ordering is testable.
    records = []
    for i in range(6):
        records.append(
            _SurvivorRecord(
                source=f"def utility(x, c): return c[0]*x[{i % 4}]",
                fn=lambda x, c: 0.0,
                coeffs=np.array([0.1 * i, -0.2 * i], dtype=np.float64),
                k=2,
                train_nll=1.0 + 0.1 * i,
                val_nll=1.0 + 0.1 * i,
            )
        )
    library = [
        Concept(
            name="price_term",
            source="-c[0] * x[0]",
            nl_summary="price sensitivity",
            usage_count=3,
            discovered_at=1,
            n_coeffs=1,
        ),
        Concept(
            name="log_price",
            source="log1p(x[0]) * c[0]",
            nl_summary="log-price term",
            usage_count=1,
            discovered_at=2,
            n_coeffs=1,
        ),
    ]
    fitted = LaSRFitted(
        name="LaSR",
        best_equation=records[0].source,
        best_coefficients=records[0].coeffs,
        n_coefficients=2,
        train_nll=records[0].train_nll,
        val_nll=records[0].val_nll,
        final_concept_library=library,
        equation_records=records,
        equation_memory=[(r.source, r.val_nll) for r in records],
    )
    out = fitted.extra_artifacts_for_json()
    assert out is not None
    # Top equations — ascending by combined, at most 10.
    eqs = out["llm_sr_top_equations"]
    assert 1 <= len(eqs) <= 10
    combined = [e["nll_train"] + e["nll_val"] for e in eqs]
    assert combined == sorted(combined)
    for e in eqs:
        assert set(e.keys()) == {"source", "nll_train", "nll_val", "coefficients"}
    # Library — schema + ordering preserved from final_concept_library.
    lib = out["lasr_final_concept_library"]
    assert len(lib) == 2
    for entry in lib:
        assert set(entry.keys()) == {
            "name", "source", "nl_summary", "usage_count", "discovered_at"
        }
    assert {entry["name"] for entry in lib} == {"price_term", "log_price"}


def test_extra_artifacts_library_only_when_no_records():
    """Library present but no records → ``llm_sr_top_equations`` is empty list."""
    library = [
        Concept(
            name="c0",
            source="x[0]",
            nl_summary="raw price",
            usage_count=1,
            discovered_at=0,
            n_coeffs=0,
        )
    ]
    fitted = LaSRFitted(
        name="LaSR",
        final_concept_library=library,
        equation_records=[],
    )
    out = fitted.extra_artifacts_for_json()
    assert out is not None
    assert out["llm_sr_top_equations"] == []
    assert len(out["lasr_final_concept_library"]) == 1


def test_extra_artifacts_populated_after_fit():
    """End-to-end stub fit populates equation_records so the hook fires."""
    train = _make_synthetic_dcm_batch(30, true_w=np.ones(4), seed=20)
    val = _make_synthetic_dcm_batch(15, true_w=np.ones(4), seed=21)
    baseline = LaSR(llm_client=StubLLMClient(), n_iters=1, proposals_per_iter=1)
    fitted = baseline.fit(train, val)
    out = fitted.extra_artifacts_for_json()
    assert out is not None
    assert "llm_sr_top_equations" in out
    assert "lasr_final_concept_library" in out


def test_fit_records_equation_memory():
    """After a successful fit the ``equation_memory`` is populated."""
    train = _make_synthetic_dcm_batch(30, true_w=np.ones(4), seed=20)
    val = _make_synthetic_dcm_batch(15, true_w=np.ones(4), seed=21)
    baseline = LaSR(
        llm_client=StubLLMClient(),
        n_iters=1,
        proposals_per_iter=1,
    )
    fitted = baseline.fit(train, val)
    # Even under the stub, the fallback skeleton occupies the memory.
    assert len(fitted.equation_memory) >= 1
    for (src, nll) in fitted.equation_memory:
        assert isinstance(src, str)
        assert np.isfinite(nll)
