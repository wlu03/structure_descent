# Zero-Shot Claude Ranker with Token-Logprob Scoring and Permutation Debiasing

**Status**: design — implementation deferred to a follow-up agent.
**Target paths**: `src/baselines/zero_shot_claude_ranker.py`,
`tests/baselines/test_zero_shot_claude_ranker.py`.

## 1. Method (what the baseline does)

Given a PO-LEU choice event with J=4 alternatives, we ask a frozen Claude
Sonnet 4.6 model: "Which alternative is this person most likely to buy?"
The person context `c_d` and the J alternatives' `alt_texts` are rendered
once into a single prompt with letters **A / B / C / D** mapped to the
canonical alternative indices 0..3. Following Hou et al.
(arXiv:2305.08845), the LLM acts as a zero-shot ranker. We then read a
probability distribution over the four letter tokens and renormalize to
get `P(j)` per canonical index. Because LLMs exhibit a well-documented
token bias toward "A" (Zheng et al., arXiv:2309.03882), we repeat the
call K=4 times with a Latin-square rotation of the letter↔alternative
map (Pezeshkpour & Hruschka, NAACL 2024, arXiv:2308.11483), un-permute
each returned distribution back to the canonical ordering, and take the
arithmetic mean. The final `log P(j)` is returned as the utility score
consumed by `evaluate.evaluate_baseline`. The baseline is "fit" trivially
(no training): `fit()` just stores the client + hyperparameters.

## 2. Prompt template (exact text)

We adapt the Hou et al. (2023) zero-shot ranking template to PO-LEU's
`c_d` + `alt_texts` schema. The `{c_d}` field is the existing
`record["c_d"]` string; each alternative is rendered from the 7-key
`alt_texts[j]` dict the pipeline already produces.

```text
System:
You are a careful shopping-decision analyst. Given a person's profile and
four product alternatives labeled A, B, C, D, identify which one the
person is most likely to purchase next. Respond with a single capital
letter: A, B, C, or D. Do not explain.

User:
PERSON:
{c_d}

ALTERNATIVES:
(A) Title: {alt_A.title}
    Category: {alt_A.category}
    Price: ${alt_A.price}
    Popularity: {alt_A.popularity_rank}
    Brand: {alt_A.brand}
    Previously purchased by this person: {alt_A.is_repeat}

(B) Title: {alt_B.title}
    ...

(C) Title: {alt_C.title}
    ...

(D) Title: {alt_D.title}
    ...

Which alternative is this person most likely to purchase? Answer with a
single capital letter (A, B, C, or D).

Assistant (prefill):
The answer is (
```

Design notes:

- The `Assistant (prefill):` line uses Anthropic's message-prefill
  mechanism. Anthropic allows the caller to append a leading
  `{"role": "assistant", "content": "The answer is ("}` message; the
  model continues from there, so the **very next generated token** is
  almost certainly one of `A`, `B`, `C`, `D` followed by `)`. This
  mitigates the first-token misalignment issue reported in
  arXiv:2402.14499 (models inserting preambles like "Sure" or "The")
  — we borrow the prefilling-attack idea from arXiv:2505.15323.
- Lines are derived from `alt_texts[j]` using the same keys that
  `src.outcomes.prompts.USER_BLOCK_TEMPLATE` already consumes, so the
  information content matches what PO-LEU's outcome generator sees.
  The field set (`title`, `category`, `price`, `popularity_rank`,
  `brand`, `is_repeat`) is a subset of the 7-key adapter output;
  `state` is dropped because it is a context attribute, not an
  alternative attribute.

## 3. Logprob extraction mechanism

### Primary path: token logprobs on the Messages API

As of April 2026, Anthropic's Messages API exposes an optional
`top_logprobs` integer parameter (0..20) gated by a `logprobs: true`
flag; the response contains a `logprobs` field on each content block
with `.content[i].logprobs.top_logprobs` = list of
`{token, logprob}` entries per position (verified via Sonnet 4.5 / 4.6
model cards; the field is documented as `null` in default responses
until explicitly requested). The mechanism used here is:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
    messages=[
        {"role": "user", "content": [{"type": "text", "text": USER,
                                       "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "content": "The answer is ("},
    ],
    max_tokens=2,
    temperature=0.0,
    logprobs=True,
    top_logprobs=20,
)
```

On the first generated position we scan `top_logprobs` for the four
letter tokens `A`, `B`, `C`, `D` (also the variants ` A`, `A)` if the
tokenizer fuses the closing paren or leading space; we take the max
logprob per *letter* across its token variants). Any of the four
letters missing from the top-20 is assigned `logprob = -inf` and later
`softmax`'d to ~0. We softmax over the 4 letter logprobs (not the
full vocabulary) to get `P(letter) = softmax([lp_A, lp_B, lp_C, lp_D])`.

### Fallback path: verbalized probability JSON elicitation

If the API rejects `logprobs=True` on the chosen model (older models
or region-restricted deployments), we fall back to a verbalized
elicitation: change the assistant instructions to
`Return JSON {"A": p_A, "B": p_B, "C": p_C, "D": p_D} with probabilities
summing to 1.`, parse the JSON, and renormalize. This path is
noticeably worse-calibrated (arXiv:2402.14499 §4) but keeps the
baseline runnable. The class exposes `scoring_mode: Literal["logprob",
"verbalized"] = "logprob"` with automatic fallback on the first
`BadRequestError` mentioning `logprobs`, cached on the instance so we
don't pay the failed call more than once.

### Fallback path 2: multi-sample empirical frequencies

If neither of the above work (e.g. a fully hermetic test environment
with only `StubLLMClient`), the class simulates `P(letter)` by
repeatedly sampling the completion at `temperature=1.0` and counting
the frequency of each starting letter over `n_samples=32` draws. This
is transparently what the stub path exercises.

## 4. Permutation debiasing algorithm

### K=4 Latin-square permutations for J=4

We do not enumerate all 24 permutations — that is 6× the API cost with
diminishing returns (Pezeshkpour & Hruschka show the residual bias
plateaus around K=4). A Latin square over {0,1,2,3} hits every
(alternative, letter-slot) pair exactly once:

```
π_0 : letter A→alt 0, B→alt 1, C→alt 2, D→alt 3   (identity)
π_1 : letter A→alt 1, B→alt 2, C→alt 3, D→alt 0   (left rotate 1)
π_2 : letter A→alt 2, B→alt 3, C→alt 0, D→alt 1   (left rotate 2)
π_3 : letter A→alt 3, B→alt 0, C→alt 1, D→alt 2   (left rotate 3)
```

This guarantees each alternative is shown in each slot exactly once
across the K calls, which is the minimum condition for first-order
positional debiasing.

### Un-permute back to canonical order

Each call `k` yields a distribution `q_k ∈ Δ^4` over letters
A/B/C/D. Under permutation `π_k`, letter-slot `s` is occupied by
canonical alternative `π_k(s)`, so the probability we meant to
express for canonical alternative `j` is

    p_k(j) = q_k(π_k^{-1}(j))

i.e. find the letter-slot that `j` was placed in on call `k`, read off
that slot's probability.

### Aggregation

We take the **arithmetic mean** of the un-permuted distributions:

    p_hat(j) = (1/K) Σ_k p_k(j)

and return `log p_hat(j)` as the score. Justification (log-averaging
vs. arithmetic-averaging):

- Arithmetic averaging of probabilities is the Bayes-optimal ensembler
  when the K calls are treated as noisy observations of a single
  underlying distribution (proper-scoring-rule argument; e.g.
  log-score is strictly proper, and arithmetic mean minimizes expected
  log loss under exchangeable priors).
- Log-averaging (i.e. geometric mean of probabilities) corresponds to
  averaging *logits* and is strictly sharper; it over-weights the
  majority call, which re-introduces the very bias the permutation
  is meant to cancel when one of the K calls happens to hit a
  high-bias layout.

We therefore use arithmetic averaging; this matches the "marginalize
over permutations" formulation in Zheng et al. (2023) §3.3 (PriDe)
and the mitigation in Pezeshkpour & Hruschka §5.

### Pseudocode

```python
def score_event(c_d, alt_texts, client, K=4, temperature=0.0):
    J = len(alt_texts)                         # 4
    permutations = _latin_square_rotations(J)  # list of K perms
    un_permuted = np.zeros((K, J), dtype=np.float64)
    for k, pi in enumerate(permutations):
        letter_to_alt = {i: pi[i] for i in range(J)}    # A..D -> 0..3
        prompt = _render_prompt(c_d, [alt_texts[pi[i]] for i in range(J)])
        letter_probs = _call_llm_for_ranking(client, prompt, temperature)
        # letter_probs[i] = P(letter i | prompt), i in 0..3
        for letter_idx in range(J):
            canonical_j = letter_to_alt[letter_idx]
            un_permuted[k, canonical_j] = letter_probs[letter_idx]
    p_hat = un_permuted.mean(axis=0)  # (J,)
    # guard against all-zeros (all calls missing a letter in top_logprobs)
    p_hat = np.maximum(p_hat, 1e-12)
    return np.log(p_hat)
```

## 5. Class signatures

```python
# src/baselines/zero_shot_claude_ranker.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence

import numpy as np

from src.baselines.base import Baseline, BaselineEventBatch, FittedBaseline
from src.outcomes.generate import LLMClient, StubLLMClient  # reuse Protocol


class ZeroShotClaudeRanker:
    """Frozen-LLM ranker with K-permutation debiasing (see design doc)."""

    name = "ZeroShotClaudeRanker"

    def __init__(
        self,
        client: LLMClient,
        *,
        K: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 2,
        scoring_mode: Literal["logprob", "verbalized", "sampling"] = "logprob",
        n_samples_fallback: int = 32,
        seed: int = 0,
        prompt_version: str = "rank-v1",
        cache: Optional[object] = None,            # optional RankingCache
        verbose: bool = False,
    ) -> None: ...

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> "ZeroShotClaudeRankerFitted":
        """Trivial fit: no training. Stores ``self`` behind a fitted wrapper.

        The only validation done here:
        - asserts ``train.n_alternatives == 4`` (baseline hard-coded to J=4).
        - checks that ``train.raw_events`` exposes ``alt_texts`` + ``c_d``.
        """
        ...


@dataclass
class ZeroShotClaudeRankerFitted:
    name: str = "ZeroShotClaudeRanker"
    client: LLMClient = field(repr=False)
    K: int = 4
    temperature: float = 0.0
    max_tokens: int = 2
    scoring_mode: str = "logprob"
    n_samples_fallback: int = 32
    seed: int = 0
    prompt_version: str = "rank-v1"
    cache: Optional[object] = None
    _model_id: str = "unknown"         # populated from client on first call

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Return list of length n_events, each a (J,) array of log P(j)."""
        ...

    @property
    def n_params(self) -> int:
        """Zero trainable parameters (frozen LLM). AIC/BIC treat it as k=1
        via the harness floor so the IC penalty is non-degenerate."""
        return 0

    @property
    def description(self) -> str:
        return (
            f"ZeroShotClaudeRanker model={self._model_id} "
            f"K={self.K} T={self.temperature} "
            f"mode={self.scoring_mode} prompt={self.prompt_version}"
        )
```

Note: `evaluate_baseline` already uses `k = max(int(fitted.n_params), 1)`
so reporting `n_params=0` is safe.

## 6. LLM client interface

The baseline accepts any `LLMClient` from `src.outcomes.generate`. We
reuse the existing Protocol rather than introduce a second one so
`StubLLMClient` is usable in tests and `AnthropicLLMClient` in
production. However, the existing Protocol has a `generate(messages,
..., seed) -> GenerationResult` signature that does **not** expose
`top_logprobs`. We resolve this in one of two ways (pick option B):

**Option A (rejected)**: extend `LLMClient` with an optional
`generate_with_logprobs` method. Rejected because it pollutes the
Protocol for every outcome-generation caller.

**Option B (chosen)**: add a private `_call_llm_for_ranking(client,
messages, temperature) -> np.ndarray[4]` helper inside
`zero_shot_claude_ranker.py`. It dispatches on `type(client)`:

- `isinstance(client, StubLLMClient)` → run `client.generate` to get
  text, then deterministically derive letter probabilities from the
  text's SHA-256 hash (see §9).
- `getattr(client, "_client", None) is not None` (duck-type for
  `AnthropicLLMClient`) → directly call `client._client.messages.create(
  ..., logprobs=True, top_logprobs=20)` bypassing `client.generate`,
  and extract top_logprobs from the first-position content block.
- else → `scoring_mode == "verbalized"` path: call `client.generate`
  with the JSON-elicitation prompt and parse.

The helper returns a `np.ndarray[4]` of probabilities already softmax'd
and summed to 1.

```python
def _call_llm_for_ranking(
    client: LLMClient,
    messages: list[dict],
    temperature: float,
    *,
    scoring_mode: str,
    seed: int,
    n_samples_fallback: int,
) -> np.ndarray:  # shape (4,)
    ...
```

## 7. Determinism and seeding

- **Temperature**: default `0.0`. Claude at T=0 is not bit-exact
  deterministic (Anthropic documents residual nondeterminism), but the
  variance is small compared to permutation bias, which dominates.
- **Permutation order**: fixed (identity, rotate-1, rotate-2,
  rotate-3), not sampled — the Latin square is a deterministic
  function of J.
- **Seed propagation**: we forward `seed + event_idx * K + k` to each
  call's `client.generate(..., seed=...)`; the stub honors this
  exactly, the real Anthropic client ignores it (SDK limitation) but
  propagates to any cache key.
- **Cache**: a sibling of `OutcomesCache` (either a new lightweight
  `RankingCache` keyed on `(customer_id, event_hash, prompt_version,
  K, model_id, perm_idx)` or a reuse of the same SQLite schema). Out
  of scope for the initial implementation; expose a `cache=None`
  kwarg so it can be wired later without a signature change.

## 8. Test strategy

Tests live in `tests/baselines/test_zero_shot_claude_ranker.py` and use
the synthetic batch generator in `src.baselines._synthetic` for
fixture events. All tests run with `StubLLMClient` — no network.

1. **`test_score_events_shape_and_length`**
   - Build a 10-event, J=4 synthetic batch.
   - Fit `ZeroShotClaudeRanker(client=StubLLMClient())`, score.
   - Assert `len(scores) == 10` and `scores[i].shape == (4,)`.
   - Assert every score vector is finite (no ±inf).

2. **`test_single_permutation_matches_argmax_for_pathological_stub`**
   - With `K=1` and a stub whose deterministic mapping forces
     "alternative index 2 gets the highest probability", assert
     `argmax(scores[i]) == 2` across all events.
   - Guards against an off-by-one in the un-permute step.

3. **`test_permutation_debiasing_cancels_synthetic_bias`**
   - Use a stub that *always* puts 0.7 on letter A and 0.1 on B/C/D.
   - With `K=1`: `argmax` always lands on the alternative currently
     mapped to A (i.e. shifts with permutation order).
   - With `K=4` (full Latin square): argmax distribution should be
     near-uniform over 4 alternatives on a batch of uniformly-random
     "true" alternatives. Concretely: χ² test with p>0.01 across N=200
     events — sanity check that the debiasing actually fires.

4. **`test_fitted_description_and_n_params`**
   - Assert `fitted.n_params == 0` and `fitted.description` starts
     with `"ZeroShotClaudeRanker "` and contains the K, temperature,
     and mode settings.

5. **`test_latin_square_covers_every_slot`** (utility test on the
   permutation builder)
   - For J=4: assert that across the 4 returned permutations, each
     alternative index appears in each letter slot exactly once.

6. **`test_score_events_raises_on_wrong_J`**
   - Build a J=3 batch; `.fit()` should raise `ValueError` with a
     message mentioning `n_alternatives`.

7. **`test_verbalized_fallback_parses_json`**
   - With `scoring_mode="verbalized"` and a stub whose `generate`
     returns `'{"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}'`, assert the
     resulting probabilities match (up to renormalization) and
     `argmax` lands on the alternative currently mapped to B.

## 9. Stub behavior

`StubLLMClient` currently returns *narrative outcome sentences*, not
rankings or JSON. Rather than introduce a second stub class (option
(a)), we go with **option (b)**: `_call_llm_for_ranking`'s stub branch
derives letter probabilities directly from a SHA-256 hash of the stub's
text output.

```python
def _stub_letter_probs(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    logits = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float64)
    logits = logits[:4] / 32.0   # scale to sane range
    return np.exp(logits) / np.exp(logits).sum()
```

This keeps the stub deterministic per `(messages, seed)` (since
`StubLLMClient.generate` already is), lets us write the tests in §8
without modifying `generate.py`, and avoids touching the existing
outcome-generation stub contract.

For the pathological-bias test (§8.3), we inject a thin subclass of
`StubLLMClient` whose `generate` returns a constant string regardless
of `messages`, yielding a fixed letter-probability vector — exactly
what's needed to probe the permutation-debias logic.

## 10. File layout and registry updates

New files:

- `src/baselines/zero_shot_claude_ranker.py`
- `tests/baselines/test_zero_shot_claude_ranker.py`

Edits to existing files:

- `src/baselines/__init__.py`: add
  `from .zero_shot_claude_ranker import ZeroShotClaudeRanker,
  ZeroShotClaudeRankerFitted` and extend `__all__`.
- `src/baselines/run_all.py::BASELINE_REGISTRY`: append
  `("ZeroShotClaudeRanker", "src.baselines.zero_shot_claude_ranker",
  "ZeroShotClaudeRanker")`. Because `run_all._try_load` swallows
  import errors, users without the `llm` extra installed will see
  it flagged `unavailable` instead of crashing the suite.
- `src/baselines/run_all.py::run_all_baselines`: thread
  `baseline_kwargs["ZeroShotClaudeRanker"] = {"client": ...}` via
  the existing `baseline_kwargs` pathway — no code change needed,
  but the caller contract in the docstring should be updated to
  note that this baseline requires a `client` kwarg.

## 11. Cost estimate

Per event (J=4, K=4 calls):

- Prompt length: ~ 500 input tokens (c_d block is ~150 tokens; four
  alternatives × ~60 tokens each = 240 tokens; instructions ~80
  tokens; overhead ~30 tokens).
- Output: 2 tokens per call (`X)` where X ∈ {A,B,C,D}).
- With Anthropic prompt caching enabled on the system prompt and the
  `PERSON:` block (the per-event c_d is stable across the K calls
  since it's the same event), ~400 of the 500 tokens are cache hits
  from call 2 onward.

At Claude Sonnet 4.6 April-2026 pricing ($3 / MTok input, $15 / MTok
output, cache reads at 0.1× = $0.30 / MTok):

- Call 1: 500 input ($0.0015) + 2 output ($0.00003) ≈ **$0.0015**
- Calls 2–4: 100 fresh input ($0.0003) + 400 cache reads ($0.00012)
  + 2 output ($0.00003) ≈ **$0.00046 each**
- **Per-event total**: ~ $0.0015 + 3 × $0.00046 ≈ **$0.0029**

For a 10k-event test set: **~ $29** end-to-end.
With batch API (50% off): **~ $14.50**.

This is cheap enough to run on every leaderboard refresh. Sanity
check against the outcome generator: the outcome pass costs ~$0.01
per `(person, alternative)` pair, so this ranker baseline is roughly
an order of magnitude cheaper per event despite the K=4 multiplier.

## 12. Known failure modes

1. **Top-20 truncation**: on events where the model is extremely
   confident (one letter at > 99.99%), the other three letters may
   fall outside the top-20 logprobs. We assign `-inf`, which softmax
   handles cleanly, but the resulting P(j) for those alternatives is
   exactly zero and `log P(j) = -inf`. `evaluate_baseline` feeds
   these into `log_softmax` over scores → fine, but NLL and AIC can
   explode if the ground-truth alternative happens to be the
   zero-probability one. Mitigation: floor logprobs at `np.log(1e-12)`
   before aggregation (already in the pseudocode).

2. **First-token misalignment** (arXiv:2402.14499): instruction-tuned
   Claude may start its answer with a preamble ("The answer is A")
   rather than the raw letter token, causing the first-position
   logprobs to be concentrated on `The`/`Sure` rather than A/B/C/D.
   **Mitigation**: the assistant prefill `"The answer is ("` forces
   the model's very next token into letter space, as demonstrated by
   arXiv:2505.15323.

3. **Tokenization variants**: Claude's tokenizer may fuse letters
   with neighboring punctuation. We scan for any of `A`, ` A`, `A)`
   and take the max; likewise for B/C/D. If the tokenizer ever
   splits letters differently, a monitoring log line flags events
   where none of the 4 letters appeared in top-20.

4. **Residual position bias after K=4**: Latin-square rotation only
   cancels first-order position bias (mean bias per slot). Second-
   order effects (e.g. pairwise interaction between slots A and B)
   survive. If a future evaluation shows K=4 insufficient, the knob
   to increase K is already plumbed — set `K=8` or `K=24` (full
   enumeration for J=4 is tractable).

5. **Logprobs API gating**: if Anthropic removes or restricts the
   `logprobs` parameter on the chosen model, the verbalized fallback
   kicks in automatically. The fallback is noisier (~2-3× higher NLL
   per arXiv:2402.14499) but keeps the baseline runnable.

6. **Prompt-caching cache miss on c_d variability**: PO-LEU's Wave-12
   per-event `c_d` changes across events but is stable across the K
   permutations within one event. The cache breakpoint should be
   placed after `c_d` so the K calls share a cached 400-token prefix
   and only the `ALTERNATIVES:` block changes. This matches the
   split-point convention in `AnthropicLLMClient._split_user_content`.

7. **Stub non-representativeness**: the hash-based stub
   (§9) produces plausible but arbitrary letter probabilities. It
   validates shapes and permutation-unwind logic but gives **no**
   signal about real-LLM accuracy. Smoke-running the real client on
   a 100-event sample is required before reporting numbers.

## 13. References

- Hou et al. 2023. *Large Language Models are Zero-Shot Rankers for
  Recommender Systems*. arXiv:2305.08845 (ECIR 2024).
- Zheng et al. 2023. *Large Language Models Are Not Robust Multiple
  Choice Selectors*. arXiv:2309.03882 (ICLR 2024); §3.3 introduces
  **PriDe** debiasing.
- Pezeshkpour & Hruschka 2024. *Large Language Models Sensitivity to
  The Order of Options in Multiple-Choice Questions*. NAACL-Findings
  2024, arXiv:2308.11483.
- Wang et al. 2024. *"My Answer is C": First-Token Probabilities Do
  Not Match Text Answers in Instruction-Tuned Language Models*.
  arXiv:2402.14499.
- Cappelletti et al. 2025. *Improving LLM First-Token Predictions
  in MCQA via Prefilling Attack*. arXiv:2505.15323.
- Anthropic Messages API reference (docs.anthropic.com/en/api/messages,
  accessed 2026-04-23): `logprobs` boolean + `top_logprobs` integer
  (0..20); Sonnet 4.5 and 4.6 both support the parameter.
- Anthropic pricing (platform.claude.com/docs/en/about-claude/pricing,
  accessed 2026-04-23): Sonnet 4.6 at $3/$15 per MTok, 0.1× cache
  reads, 50% batch-API discount.
