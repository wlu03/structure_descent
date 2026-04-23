"""Smoke tests for ``scripts/run_dataset.py`` (Wave 10).

One main end-to-end test (``test_end_to_end_with_sdk_mocks_on_amazon_fixture``)
and a handful of argparse edge-case tests. The main test is the CI gate:
it drives the full pipeline on the 100-row Amazon fixture with the
``anthropic`` and ``sentence_transformers`` packages faked at the
``sys.modules`` level so the real-only driver stays hermetic in CI.

The stub/real mutex that used to live on the driver has been removed --
``scripts/run_dataset.py`` now *only* builds real clients, by construction.
Stubs still exist in ``src/outcomes/{generate,encode}.py`` for unit tests
and for the cache-reuse test at the bottom of this file, but the driver
refuses them via an ``isinstance`` tripwire at startup.

The test adapter-YAML override technique (writing a tmp copy of
``configs/datasets/amazon.yaml`` with ``events.path`` / ``persons.path``
pointed at the fixtures) mirrors the pattern used elsewhere in the test
suite (see ``tests/test_adapter.py``).

"""

from __future__ import annotations

import json
import logging
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_fixture_yaml(tmp_path: Path) -> Path:
    """Copy the Amazon YAML into tmp_path, overriding fixture paths."""
    with AMAZON_YAML.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    doc["dataset"]["events"]["path"] = str(EVENTS_FIXTURE)
    doc["dataset"]["persons"]["path"] = str(PERSONS_FIXTURE)
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "amazon_fixture.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


def _base_args(tmp_path: Path, dataset_yaml: Path) -> Namespace:
    """Build a Namespace matching the CLI contract for main()."""
    return Namespace(
        adapter="amazon",
        n_customers=10,
        seed=42,
        config=REPO_ROOT / "configs" / "default.yaml",
        n_epochs=1,
        batch_size=4,
        min_events_per_customer=5,
        output_dir=tmp_path,
        K=3,
        dataset_config=dataset_yaml,
    )


# --------------------------------------------------------------------------- #
# Fakes for the anthropic + sentence_transformers SDKs
# --------------------------------------------------------------------------- #


# Canned outcome text families -- copied (shortened) from
# StubLLMClient._STUB_TEMPLATES so the test's fake LLM produces plausibly
# varied three-line completions.
_CANNED_OUTCOMES: tuple[str, ...] = (
    "I save roughly twelve dollars this month and redirect the difference "
    "toward groceries without juggling my usual weekend spending plans.",
    "I sleep noticeably better across the week because the evening routine "
    "gets simpler and my shoulders stop aching by bedtime.",
    "I shave about ten minutes off the Tuesday errand loop and reach the "
    "school pickup line before it wraps around the block.",
    "I feel quietly proud about the decision tonight and the low hum of "
    "decision fatigue I usually carry finally eases a little.",
    "I earn a small round of thanks from my partner at dinner and the "
    "household logistics feel briefly like a shared project again.",
)


class _FakeAnthropicContentBlock:
    """Mimics the ``content`` block returned by Anthropic's Messages API."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    """Mimics ``anthropic.types.Message`` enough for AnthropicLLMClient."""

    def __init__(self, text: str, model_id: str) -> None:
        self.content = [_FakeAnthropicContentBlock(text)]
        self.stop_reason = "end_turn"
        self.model = model_id
        self.usage = None


class _FakeMessages:
    def __init__(self, model_id: str, counter: dict[str, int]) -> None:
        self._model_id = model_id
        self._counter = counter

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        # Rotate through canned outcomes deterministically; three-line
        # completion so parse_completion accepts K<=3 without padding.
        self._counter["n"] = self._counter.get("n", 0) + 1
        i = self._counter["n"] % len(_CANNED_OUTCOMES)
        lines = [
            _CANNED_OUTCOMES[i],
            _CANNED_OUTCOMES[(i + 1) % len(_CANNED_OUTCOMES)],
            _CANNED_OUTCOMES[(i + 2) % len(_CANNED_OUTCOMES)],
        ]
        return _FakeAnthropicResponse("\n".join(lines), self._model_id)


class _FakeAnthropicClient:
    def __init__(self, api_key: str | None = None, **_: Any) -> None:
        self._api_key = api_key
        self._model_id = "fake-sonnet"
        self._counter: dict[str, int] = {"n": 0}
        self.messages = _FakeMessages(self._model_id, self._counter)


class _FakeBadRequestError(Exception):
    """Minimal stand-in for ``anthropic.BadRequestError``."""


def _build_fake_anthropic_module() -> types.ModuleType:
    mod = types.ModuleType("anthropic")
    mod.Anthropic = _FakeAnthropicClient  # type: ignore[attr-defined]
    mod.BadRequestError = _FakeBadRequestError  # type: ignore[attr-defined]
    return mod


class _FakeSentenceTransformer:
    """Mimics the subset of sentence_transformers.SentenceTransformer the encoder uses."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.max_seq_length = 64
        self._d_e = 768

    def get_sentence_embedding_dimension(self) -> int:
        return self._d_e

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Deterministic canned embedding: hash each text to a seed, draw
        # gaussians, L2-normalize. Matches the real client's shape + norm
        # contract exactly.
        import hashlib

        out = np.zeros((len(texts), self._d_e), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.blake2b(
                t.encode("utf-8"), digest_size=8
            ).digest()
            seed = int.from_bytes(digest, "big", signed=False)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._d_e).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec /= norm
            out[i] = vec
        return out


def _build_fake_sentence_transformers_module() -> types.ModuleType:
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def mocked_llm_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake ``anthropic`` + ``sentence_transformers`` into sys.modules.

    Installing the fakes BEFORE the driver's lazy imports means
    ``AnthropicLLMClient`` / ``SentenceTransformersEncoder`` resolve to
    them transparently. The driver's ``isinstance`` tripwire against
    ``StubLLMClient`` / ``StubEncoder`` still passes because the fakes
    are neither.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    monkeypatch.setitem(
        sys.modules, "anthropic", _build_fake_anthropic_module()
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        _build_fake_sentence_transformers_module(),
    )


# --------------------------------------------------------------------------- #
# Argparse edge cases
# --------------------------------------------------------------------------- #


def test_argparse_requires_adapter():
    """Missing --adapter --> SystemExit."""
    from scripts.run_dataset import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--n-customers", "10"])


def test_argparse_rejects_stub_llm_flag():
    """--stub-llm is gone; argparse must reject it."""
    from scripts.run_dataset import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--adapter", "amazon", "--n-customers", "10", "--stub-llm"]
        )


def test_argparse_rejects_real_llm_flag():
    """--real-llm is gone too; argparse must reject it."""
    from scripts.run_dataset import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--adapter", "amazon", "--n-customers", "10", "--real-llm"]
        )


def test_help_output_does_not_mention_stub_or_real():
    """--help text mentions neither --stub-llm nor --real-llm."""
    from scripts.run_dataset import _build_arg_parser

    parser = _build_arg_parser()
    help_text = parser.format_help()
    assert "--stub-llm" not in help_text
    assert "--real-llm" not in help_text


def test_missing_api_key_exits_early(monkeypatch: pytest.MonkeyPatch):
    """Running the driver without ANTHROPIC_API_KEY exits with code 2."""
    from scripts.run_dataset import _build_llm_and_encoder

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        _build_llm_and_encoder(Namespace(), {})
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# Main end-to-end smoke (SDK-mocked)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not EVENTS_FIXTURE.exists()
    or not PERSONS_FIXTURE.exists()
    or not AMAZON_YAML.exists(),
    reason="Amazon fixtures / config not available",
)
def test_end_to_end_with_sdk_mocks_on_amazon_fixture(
    tmp_path: Path, caplog, mocked_llm_sdks
):
    """End-to-end run on the 100-row Amazon fixture with mocked SDKs.

    The ``anthropic`` and ``sentence_transformers`` packages are replaced
    at ``sys.modules`` level by fakes that return canned outcomes / canned
    embeddings. The driver still builds the REAL ``AnthropicLLMClient`` +
    ``SentenceTransformersEncoder`` -- its ``isinstance`` tripwire passes
    because the fakes are neither stub classes. The test stays hermetic:
    no network calls, no real model downloads.

    Asserts exit code 0, the full set of written reports, finite NLL in
    ``metrics.json``, finite ``train_loss``/``val_nll`` in
    ``smoke_summary.json``, ``llm_mode == "real"``, and at least one INFO
    log line.
    """
    from scripts.run_dataset import main

    dataset_yaml = _write_fixture_yaml(tmp_path / "cfg")
    args = _base_args(tmp_path / "out", dataset_yaml)

    caplog.set_level(logging.INFO)
    exit_code = main(args)
    assert exit_code == 0

    out_dir = Path(args.output_dir)
    expected = {
        "metrics.json",
        "metrics_test.json",
        "head_naming.json",
        "per_decision.json",
        "dominant_attribute.json",
        "counterfactual.json",
        "subsample_diagnostics.json",
        "smoke_summary.json",
    }
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    missing = expected - written
    assert not missing, f"missing artifacts: {missing}; got {written}"

    # metrics.json: §13 fields + finite NLL + Wave-11 diagnostics.
    metrics = json.loads((out_dir / "metrics.json").read_text())
    for field in ("top1", "top5", "mrr_val", "nll_val",
                  "aic_val", "bic_val"):
        assert field in metrics, f"metrics.json missing {field!r}"
    assert _isfinite(metrics["nll_val"]), "nll_val must be finite"

    # Wave-11 diagnostics: effective_p + regularizers_active in metrics.json.
    for field in ("effective_p", "canonical_p", "p_is_dataset_dependent",
                  "p_reduction_reason", "regularizers_active"):
        assert field in metrics, f"metrics.json missing {field!r}"
    assert isinstance(metrics["effective_p"], int)
    assert isinstance(metrics["canonical_p"], int)
    assert isinstance(metrics["p_is_dataset_dependent"], bool)
    assert isinstance(metrics["p_reduction_reason"], str)
    reg_active = metrics["regularizers_active"]
    for name in ("weight_l2", "salience_entropy", "monotonicity", "diversity"):
        assert name in reg_active, f"regularizers_active missing {name!r}"
    # default.yaml has all four λ's > 0 AND monotonicity is now on by
    # default + the driver threads prices through batch_train -- all four
    # terms must fire on Amazon.
    assert reg_active["weight_l2"] is True
    assert reg_active["salience_entropy"] is True
    assert reg_active["monotonicity"] is True, (
        "monotonicity must fire on Amazon (enabled by default + prices "
        "threaded through batch_train)."
    )
    assert reg_active["diversity"] is True

    # metrics_test.json: §13 fields + Wave-11 diagnostics.
    test_metrics = json.loads((out_dir / "metrics_test.json").read_text())
    for field in ("top1", "top5", "mrr_val", "nll_val",
                  "aic_val", "bic_val", "effective_p", "canonical_p",
                  "p_is_dataset_dependent", "p_reduction_reason",
                  "regularizers_active", "n_test_records"):
        assert field in test_metrics, f"metrics_test.json missing {field!r}"

    # smoke_summary.json: train_state with finite train_loss + val_nll.
    summary = json.loads((out_dir / "smoke_summary.json").read_text())
    assert "train_state" in summary
    ts = summary["train_state"]
    assert _isfinite(ts["train_loss"]), "train_state.train_loss must be finite"
    assert ts["val_nll"] is not None and _isfinite(ts["val_nll"]), (
        "train_state.val_nll must be a finite number"
    )

    # llm_mode is now a hard-coded constant.
    assert summary["config"]["llm_mode"] == "real"

    # Wave-11 diagnostics mirrored in smoke_summary.json.
    for field in ("effective_p", "canonical_p", "p_is_dataset_dependent",
                  "p_reduction_reason", "regularizers_active",
                  "metrics_test"):
        assert field in summary, f"smoke_summary.json missing {field!r}"
    sreg = summary["regularizers_active"]
    assert sreg["monotonicity"] is True
    assert sreg["weight_l2"] is True
    assert sreg["salience_entropy"] is True
    assert sreg["diversity"] is True
    assert summary["config"].get("n_test_records") is not None

    # caplog: at least one INFO line.
    info_records = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert len(info_records) > 0, "expected at least one INFO log record"


def _isfinite(x: Any) -> bool:
    try:
        import math
        return math.isfinite(float(x))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Cache-reuse on a second run
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not EVENTS_FIXTURE.exists()
    or not PERSONS_FIXTURE.exists()
    or not AMAZON_YAML.exists(),
    reason="Amazon fixtures / config not available",
)
def test_stub_llm_end_to_end_uses_cache_on_second_run(
    tmp_path: Path, monkeypatch
):
    """Outcomes / embeddings caches persist; second assemble_batch hits cache only.

    Directly exercises the cache wiring that ``main()`` threads into
    :func:`src.data.batching.assemble_batch`: same records, shared
    :class:`OutcomesCache` / :class:`EmbeddingsCache`, counting-wrapped
    stub LLM. The second assemble must produce zero additional LLM
    ``generate()`` calls because all ``(customer_id, asin, seed,
    prompt_version)`` keys already sit in the outcomes cache.

    This test imports :class:`StubLLMClient` directly from
    :mod:`src.outcomes.generate`; it does NOT go through the driver,
    which is real-only.
    """
    from src.data.batching import assemble_batch
    from src.outcomes.cache import EmbeddingsCache, OutcomesCache
    from src.outcomes.encode import StubEncoder
    import src.outcomes.generate as gen_module

    # Counting wrapper around StubLLMClient.generate, applied via
    # monkeypatch to mirror the brief's guidance.
    call_counter = {"n": 0}
    _orig_generate = gen_module.StubLLMClient.generate

    def _counting_generate(self, *a, **kw):
        call_counter["n"] += 1
        return _orig_generate(self, *a, **kw)

    monkeypatch.setattr(
        gen_module.StubLLMClient, "generate", _counting_generate
    )

    # Build three tiny deterministic records (mirrors tests/test_batching.py).
    N, J, p = 3, 4, 26
    rng = np.random.default_rng(0)
    records: list[dict] = []
    for i in range(N):
        z = rng.standard_normal(p).astype(np.float32)
        cs_idx = i % J
        choice_asins = [f"A{i}_{j:02d}" for j in range(J)]
        chosen_asin = choice_asins[cs_idx]
        alt_texts = [
            {"title": f"t {choice_asins[j]}", "category": "c",
             "price": float(5.0 + j), "popularity_rank": "top 50%"}
            for j in range(J)
        ]
        records.append({
            "customer_id": f"C{i:03d}",
            "chosen_asin": chosen_asin,
            "choice_asins": choice_asins,
            "chosen_idx": cs_idx,
            "z_d": z,
            "c_d": f"Person context for C{i:03d}.",
            "alt_texts": alt_texts,
            "order_date": None, "category": "c",
            "chosen_features": {}, "metadata": {},
        })

    o_cache = OutcomesCache(tmp_path / "out.sqlite")
    e_cache = EmbeddingsCache(tmp_path / "emb.sqlite")

    llm = gen_module.StubLLMClient()
    enc = StubEncoder(d_e=64)

    call_counter["n"] = 0
    assemble_batch(
        records, adapter=None, llm_client=llm, encoder=enc,
        outcomes_cache=o_cache, embeddings_cache=e_cache,
        K=3, seed=99,
    )
    first_calls = call_counter["n"]
    assert first_calls == N * J, (
        f"first run: expected {N*J} LLM calls (fresh cache), got {first_calls}"
    )

    call_counter["n"] = 0
    assemble_batch(
        records, adapter=None, llm_client=llm, encoder=enc,
        outcomes_cache=o_cache, embeddings_cache=e_cache,
        K=3, seed=99,
    )
    assert call_counter["n"] == 0, (
        f"second run expected 0 LLM calls (cache hits), got {call_counter['n']}"
    )
