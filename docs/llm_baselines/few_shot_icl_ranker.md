# Few-Shot ICL Ranker with Prior User Choices

Frozen-LLM baseline for PO-LEU. Companion to `zero_shot_claude_ranker.md`.

## 1. Method

A frozen Claude model picks one of 4 labelled alternatives (`A`/`B`/`C`/`D`) for
a PO-LEU choice event, and is scored by the token log-probability of each
letter. Unlike the Zero-Shot Claude Ranker, **the prompt is prefixed with `N`
in-context demonstrations drawn from the same customer's earlier training
events**, each shown as `(c_d, alternatives, chosen letter)`, in chronological
order. The model is asked to generalise the customer's revealed preferences
from those demonstrations to the test event. This follows Chat-REC (Gao et al.
2023, arXiv:2303.14524) and the few-shot LLMRank configuration in Hou et al.
2023 §5 (arXiv:2305.08845), with prompt-format conventions from Wang et al.
(ICLR 2024, "Large Language Models as Zero-Shot Conversational Recommenders").
It is bounded from above by retrieval-augmented ICL (Lin et al. 2024 "ReLLa",
arXiv:2308.11131), which uses similarity-based selection; we deliberately stop
short of retrieval so this baseline isolates the *personalisation-via-ICL*
effect versus PO-LEU's learned personalisation head.

## 2. Relationship to `ZeroShotClaudeRanker`

Shared infrastructure is extracted to `src/baselines/_llm_ranker_common.py`:

- `render_alternatives(alt_texts, letters)` — the alternatives block reused
  by both baselines.
- `letter_permutations(n_alts, K, seed)` — the K=4 rotation schedule.
- `call_llm_for_ranking(client, system, user, letters)` — single LLM call
  returning per-letter logprobs.
- `extract_letter_logprobs(response, letters)` — parses the response; see §10
  for the stub path.
- `LLMRankerBase` — holds `(llm_client, n_permutations, seed)`, defaults the
  client to `StubLLMClient()`, and exposes a shared `_score_one_event(...)`
  that runs the K-permutation average.

Zero-shot has no extras beyond this base. Few-shot adds: timeline
construction (§3), ICL-example rendering into the prompt (§5), and cold-start
fallback (§8). Both baselines MUST use identical letters, permutation seeds,
and extraction path so that ΔNLL between them is attributable to the ICL
prefix alone.

## 3. User-timeline construction at fit time

`fit(train, val)` never calls the LLM — it materialises a dict
`customer_id -> list[ICLExample]`, sorted ascending by `order_date`:

```python
@dataclass(frozen=True)
class ICLExample:
    order_date: pd.Timestamp
    c_d: str
    alt_texts: list[dict]   # length J, 7-key adapter schema
    chosen_idx: int         # in [0, J)

def build_customer_timeline(
    train: BaselineEventBatch,
) -> dict[str, list[ICLExample]]:
    if train.raw_events is None:
        return {}  # every test event will cold-start to zero-shot
    timeline: dict[str, list[ICLExample]] = {}
    for rec in train.raw_events:
        cid = str(rec["customer_id"])
        timeline.setdefault(cid, []).append(ICLExample(
            order_date=pd.Timestamp(rec["order_date"]),
            c_d=str(rec["c_d"]),
            alt_texts=list(rec["alt_texts"]),
            chosen_idx=int(rec["chosen_idx"]),
        ))
    for cid in timeline:
        timeline[cid].sort(key=lambda e: e.order_date)
    return timeline
```

`order_date` is guaranteed non-NaT and sort-stable: `src/data/invariants.py`
enforces `order_date_no_nat` + `order_date_dtype`, and `src/data/clean.py:80`
plus `src/data/state_features.py:152` sort events by
`(customer_id, order_date)`. `raw_events is None` degrades cleanly.

## 4. Test-time ICL example selection

Strategy: **recency** (Hou et al. 2023 §5's best simple heuristic).

```python
def select_icl_examples(timeline, customer_id, test_order_date, n_shots):
    history = timeline.get(customer_id, [])
    # Strict `<` to defend against tied timestamps from the same checkout.
    prior = [e for e in history if e.order_date < test_order_date]
    if not prior:
        return []                # cold-start → caller falls back to zero-shot
    return prior[-n_shots:]      # N most recent; list is sorted ascending
```

Random discards preference drift; similarity crosses into ReLLa territory and
conflates retrieval with personalisation.

## 5. Prompt template

System prompt (verbatim from `ZeroShotClaudeRanker`):

```
You are an expert recommender predicting a customer's next purchase. You are
shown the customer's current shopping context, a small number of their
previous choices (if any), and four candidate alternatives labelled A, B, C, D.
Choose the single alternative the customer is most likely to pick. Reply with
exactly one letter: A, B, C, or D. Do not explain.
```

User prompt — the `<<<END OF PRIOR CHOICES>>>` sentinel is the split marker
for Anthropic prompt caching; everything up to and including it goes in a
cacheable content block (`cache_control={"type":"ephemeral"}`), everything
after it is the per-event suffix. This mirrors the existing
`_USER_SPLIT_MARKER` pattern in `AnthropicLLMClient`
(`src/outcomes/generate.py:313`):

```
PRIOR CHOICES BY THIS CUSTOMER (chronological):

=== Example 1 ===
CONTEXT:
{icl[0].c_d}

ALTERNATIVES:
A. {render(icl[0].alt_texts[0])}
B. {render(icl[0].alt_texts[1])}
C. {render(icl[0].alt_texts[2])}
D. {render(icl[0].alt_texts[3])}

CHOSEN: {letter(icl[0].chosen_idx)}

=== Example 2 === ...
=== Example N === ...

<<<END OF PRIOR CHOICES>>>

NOW PREDICT THE CUSTOMER'S NEXT CHOICE.

CONTEXT:
{test.c_d}

ALTERNATIVES:
A. {render(test.alt_texts[0])}
B. {render(test.alt_texts[1])}
C. {render(test.alt_texts[2])}
D. {render(test.alt_texts[3])}

ANSWER (one letter, A/B/C/D):
```

The test-event portion (from `NOW PREDICT …` onward) is **byte-identical** to
the zero-shot ranker's user prompt so letter bias and extraction behaviour
are directly comparable.

## 6. Permutation debiasing — inherited

Yes, inherit `K=4` letter-permutation rotations from zero-shot. The letter
bias is independent of whether a demonstration prefix is present. Subtle
point: the ICL examples' `CHOSEN:` letter MUST rotate in lockstep with the
query's permutation — otherwise the model sees a mismatched label scheme and
develops a spurious preference for whichever letter was overrepresented in
history. A golden-prompt test (§11) pins this down.

## 7. Class signatures

```python
# src/baselines/_llm_ranker_common.py
class LLMRankerBase:
    name: str
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        n_permutations: int = 4,
        seed: int = 0,
    ) -> None: ...

# src/baselines/few_shot_icl_ranker.py
class FewShotICLRanker(LLMRankerBase):
    name = "FewShot-ICL-Claude"

    def __init__(
        self,
        n_shots: int = 3,
        llm_client: LLMClient | None = None,
        n_permutations: int = 4,
        seed: int = 0,
        max_prefix_tokens: int = 12_000,   # §13 truncation guard
    ) -> None: ...

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> "FewShotICLRankerFitted": ...

@dataclass
class FewShotICLRankerFitted:
    name: str
    llm_client: LLMClient
    timeline: dict[str, list[ICLExample]]
    n_shots: int
    n_permutations: int
    seed: int
    max_prefix_tokens: int
    _cold_start_count: int = 0       # populated during score_events
    _total_events: int = 0

    def score_events(self, batch: BaselineEventBatch) -> list[np.ndarray]: ...

    @property
    def n_params(self) -> int:
        # Frozen-LLM ranker exposes zero *learned* parameters; `timeline` is
        # a lookup table, not fit coefficients. Keeps AIC/BIC comparable
        # across frozen-LLM baselines.
        return 0

    @property
    def description(self) -> str:
        return (
            f"FewShot-ICL n_shots={self.n_shots} K={self.n_permutations} "
            f"cold_start={self._cold_start_count}/{self._total_events}"
        )
```

## 8. Cold-start semantics

A test event cold-starts when `select_icl_examples(...)` returns `[]`, which
covers: (a) `customer_id` not in train; (b) customer in train but every
training event is dated at-or-after the test (rare; happens only with
non-temporal splits); (c) `raw_events is None` globally. In every case the
baseline falls back to the zero-shot prompt for that single call.
`_cold_start_count` is surfaced in `description` (e.g. `"cold_start=42/500"`)
for audit. Cross-user pooling is deliberately out of scope (ReLLa territory).

## 9. Cost analysis

Amazon-pilot shape: `c_d` ≈300 tok, one alt ≈80 tok, J=4, letter output 2 tok.
Zero-shot per call ≈800 input tok; few-shot N=3 adds an ICL prefix of
3×(c_d + 4 alts + CHOSEN) ≈1020 tok → raw ≈1820 tok (naïve 2.3×).

With Anthropic caching (5-min TTL, write 1.25×, read 0.10×) over one
customer's ~50 events × K=4 permutations, the prefix is written once per
permutation and read for the remaining ~49 calls: amortised per-call billed
tokens ≈`(800 + 1020×1.25 + 49·(800 + 1020×0.10)) / 50` ≈ **~925 tok**, i.e.
**~1.15× zero-shot** rather than 2.3×.

20 customers × 50 events × K=4 × ~925 tok ≈ 3.7M input tokens → ~$11 at
Sonnet-4.6 input pricing, vs ~$9.60 for zero-shot. Output is negligible
(~$0.24). Cache efficiency grows with per-customer event density.

## 10. Stub behaviour

`StubLLMClient` does not expose per-token logprobs. `extract_letter_logprobs`
in `_llm_ranker_common.py` implements one shared fallback used by both
baselines: if the response's first non-whitespace character is one of the
four letters, assign that letter logprob `0.0` and the others `-∞`. This
gives a well-defined top-1 score (top-5 and MRR degenerate, as noted in the
zero-shot design). Unit tests monkey-patch this helper to return a fixed
4-vector so the permutation loop and ICL plumbing are verified
deterministically; few-shot-specific tests additionally stub the helper to
return a letter derived from `hash(customer_id, icl_letters,
permutation_idx)`, so tests can confirm that the ICL prefix actually reaches
the call and that permutations change the output.

## 11. Test strategy

`tests/baselines/test_few_shot_icl_ranker.py`:

- `test_builds_customer_timeline_from_raw_events` — fit on a hand-built batch
  with 2 customers × 5 events; assert `timeline` cardinality and ascending
  `order_date` order per customer.
- `test_chronological_order_enforced` — training events at `t=1,2,3`, test
  event at `t=0`; assert the selected ICL list is empty (cold-start), not
  the 3 most-recent.
- `test_cold_start_falls_back_to_zero_shot` — fit on customer "A", score a
  test event for customer "B"; assert the emitted user prompt omits
  `"PRIOR CHOICES BY THIS CUSTOMER"` and is byte-identical to the
  zero-shot prompt.
- `test_n_shots_truncates_to_most_recent` — timeline has 10 events,
  `n_shots=3`; assert the ICL block references exactly the 3 most recent.
- `test_score_events_respects_permutation_count` — count-call monkey-patch;
  assert `n_permutations=4` produces exactly `4 × n_events` LLM
  invocations and that score vectors are averaged across permutations.
- `test_raw_events_none_graceful_degradation` — batch with
  `raw_events=None`, fit + score, every event cold-starts.
- `test_icl_letters_rotate_with_permutation` — golden-prompt test: on each
  of the K=4 rotations, the ICL `CHOSEN:` letter matches the permutation's
  mapping for that example's true `chosen_idx`.

## 12. File layout

```
src/baselines/
  _llm_ranker_common.py         # NEW — LLMRankerBase, render_alternatives,
                                #       letter_permutations, call_llm_for_ranking,
                                #       extract_letter_logprobs
  zero_shot_claude_ranker.py    # (from sibling design doc)
  few_shot_icl_ranker.py        # NEW — FewShotICLRanker, ICLExample,
                                #       build_customer_timeline,
                                #       select_icl_examples
tests/baselines/
  test_few_shot_icl_ranker.py   # NEW — see §11
```

`src/baselines/run_all.py` registry append:
`("FewShot-ICL-Claude", "src.baselines.few_shot_icl_ranker", "FewShotICLRanker")`.
Same `ANTHROPIC_API_KEY` gating as zero-shot: if the env var is unset,
instantiate with `StubLLMClient()`.

## 13. Known failure modes

- **Context overflow** for long-history customers. Default `n_shots=3` keeps
  the prefix ≈1000 tok — well within the window — but the
  `max_prefix_tokens=12_000` guard truncates the *oldest* ICL examples when
  exceeded (logged). With `n_shots=3` this rarely fires.
- **ICL label leakage** — prevented by the strict `<` check in
  `select_icl_examples` and by building the timeline from `train.raw_events`
  only. `test_chronological_order_enforced` pins this down.
- **Recency bias in the choice distribution** — if a customer's 3 most
  recent purchases were the same brand, the model can lock onto that brand
  even when the test alternatives don't include it. Documented limitation;
  Hou et al. 2023's recommended recency strategy inherits this.
- **Cold-start collapse** — on splits where most test customers are unseen
  in train, this baseline degenerates to zero-shot and ΔNLL vs zero-shot
  approaches zero. The `_cold_start_count / _total_events` ratio in
  `description` makes the collapse observable rather than silent.
- **Permutation × ICL letter consistency** — a bug here looks like a spurious
  letter preference. Caught by `test_icl_letters_rotate_with_permutation`.
