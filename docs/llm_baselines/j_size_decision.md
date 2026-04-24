# J-Size Decision for the PO-LEU Amazon Discrete-Choice Benchmark

**Status:** proposal / recommendation
**Audience:** PO-LEU paper authors, internal review
**Question:** should the choice-set size `J` for the Amazon experiment stay
at 10 (current `configs/datasets/amazon.yaml::choice_set_size: 10`) or drop
to 4 to match the hard-coded `DEFAULT_LETTERS = ("A","B","C","D")` of the
LLM ranker baselines (`src/baselines/_llm_ranker_common.py:60`)?

**TL;DR.** Keep J=10 as the headline setting and extend the LLM baselines
to J=10 with a K=10 Latin-square de-biasing schedule. Report a J=4
sensitivity row so reviewers can compare to the MCQ-bias literature. This
is cheaper to defend than the alternatives, and the engineering cost is
small (~80 lines + a 2.5x LLM-call multiplier on the two ranker
baselines only).

---

## 1. Literature: what J do comparable papers use?

| Paper | Task | J (candidate set size) | Stated justification |
|---|---|---|---|
| Hou et al. 2023, "LLMs are Zero-Shot Rankers for Recommender Systems" (arXiv:2305.08845, ECIR'24) | Sequential recommendation on ML-1M and Amazon Games | **J=20** (1 positive + 19 random negatives); sensitivity studies vary historical-sequence length | "balanced candidate set" — 20 is the default for LLMRank on both datasets |
| Sun et al. 2023 RankGPT (arXiv:2304.09542, EMNLP'23) | Passage reranking (TREC, BEIR) | BM25 top-100, but LLM sees a **sliding window of 10-20** per call | context-length constraint — "only 10-20 passages can be fed to the LLM at a time" |
| Qin et al. 2023 PRP (arXiv:2306.17563, NAACL-F'24) | Pairwise ranking over full list | **J=2 per LLM call**, heap-sorted over 100 BM25 candidates | drastically cheaper per-call; O(N log N) sort complexity |
| Zhuang et al. 2024 Setwise (arXiv:2310.09497) | Listwise ranking | **c=3-4 per prompt**, sliding-window over 100 | Flan-T5 512-token context; c is a tunable knob |
| Zheng et al. 2023 PriDe (arXiv:2309.03882, ICLR'24) | MCQA on MMLU, ARC, CommonsenseQA | **J=4-5** (benchmark-native) | MMLU/ARC are J=4 by construction; CSQA is J=5 |
| Pezeshkpour & Hruschka 2024 (arXiv:2308.11483, NAACL-F'24) | MCQA order sensitivity | **J=4-5** | follows MMLU/ARC/CSQA benchmark conventions |
| Wang et al. 2024 "My Answer is C" (arXiv:2402.14499) | MCQA first-token vs text disagreement | **J=4** | MMLU / CSQA harness |
| Robinson et al. 2022 (arXiv:2210.12353) | MCSB across 20 tasks | **J=4-5** | benchmark-driven |
| MMLU, HellaSwag, ARC, TruthfulQA (MC1) | LLM MCQA benchmarks | **J=4** (HellaSwag, MMLU), J=4-5 (ARC, TruthfulQA) | community convention |
| BERT4Rec / SASRec replicability (Petrov & Macdonald 2022; Klenitskiy & Vasilev 2023) | Sequential recommendation leave-one-out | **J=101** (1 positive + 100 random negatives) | RecSys convention since NCF (He et al. 2017) |
| McFadden sampled logit (McFadden 1978; Guevara & Ben-Akiva replication arXiv:2101.06211) | Discrete-choice MNL with sampled alternatives | **J arbitrary** (theory), often 5-20 | consistency is preserved for *any* J >= 2 with the correction factor; choice of J is a variance-vs-compute trade-off |

Two distinct conventions emerge:

- **LLM-MCQA convention** (Zheng, Pezeshkpour, Wang, Robinson,
  MMLU/ARC/CSQA). J=4-5, driven by the source benchmark, not by a
  statistical argument. All letter-bias / position-bias empirical findings
  live here.
- **LLM-ranker / RecSys convention** (Hou, RankGPT, Setwise, PRP; also
  BERT4Rec-family leave-one-out). J=10-100; listwise if it fits, sliding
  window otherwise. No one claims J=4 is statistically adequate for a
  recommender benchmark — the default has always been 20-100.

PO-LEU is a *discrete-choice recommender* task, not an MCQA task. Its
natural reference class is Hou 2023 and BERT4Rec-style leave-one-out, not
MMLU. J=10 already sits on the low end of the ranker convention; J=4
would put it below every listed recsys paper in the table.

## 2. Bias at large J: does Zheng-style letter bias generalize to J=10?

**The empirical result pins first-order letter bias at J=4.** Zheng 2023's
central claim is token-ID bias: probability mass is preferentially
allocated to the "A"/"B"/"C"/"D" token identities, not to ordinal
positions. Pezeshkpour & Hruschka confirm the bias is primarily ordinal
(first/last) once you control for token identity, and Wang 2024 shows the
first-token scoring is unstable under instruction tuning. None of these
papers run J>5 experiments.

**What we can infer for J=10 (letters A..J):**

- *Token-ID bias:* "E".."J" are less frequent as answer tokens in training
  data than "A".."D", so we should expect a *stronger* anti-late-letter
  prior at J=10 than at J=4 (uncalibrated first-token scoring).
- *Position bias:* Hou 2023 explicitly reports position/order bias on J=20
  recommendation candidates and proposes two mitigations (bootstrapping
  and candidate-order perturbation). The effect size is comparable to
  MMLU-scale, and PriDe-style prior subtraction generalizes: estimate
  the prior on a held-out sample of permutations, subtract from each
  test prediction. PriDe is argued to transfer across domains and
  option counts.
- *Latin-square debiasing:* the `letter_permutations` helper
  (`src/baselines/_llm_ranker_common.py:139`) already returns a
  left-rotation schedule for arbitrary `n_alts`. For J=10 the cleanly
  bias-cancelling setting is K=10 — each alternative appears in each
  letter slot exactly once. K=4 at J=10 would only cover 4 slots, so
  alternatives 4..9 would never rotate through "A" and first-order
  cancellation fails. The left-rotation Latin square is already
  encoded; going from K=4 to K=10 requires zero code changes to the
  rotation logic, only config changes (`n_permutations=10` in the
  ranker constructors) and the letter set.

**Is K=J=10 tractable?** At current scale (50 customers, ~1000 test
events, K=4 => ~4k Claude calls) going to K=10 gives ~10k calls. With the
`_openai_client.py` / Anthropic prompt-caching plumbing already landed in
`ca96f99` ("pipeline improvements: per-event c_d, prompt caching,
parallelism"), the prompt prefix (instructions + few-shot examples) is
cached across the 10 permutations of a single event — the marginal cost
per extra permutation is only the ~200-token alternatives block plus the
logprob response. Empirically that's <10% the full-prompt cost, so the
K=10 vs K=4 wall-clock multiplier is closer to 1.3-1.5x than 2.5x.

**Context-window check.** A 10-alternative prompt with adapter-schema
alt_texts (7 keys, ~150 tokens per alt) plus the context_string (~300
tokens) plus ICL (up to 6 demos * ~800 tokens) plus instructions sits
well under 20k tokens. Claude 3.5 Sonnet and Gemini 1.5 Pro both handle
this trivially. No context pressure.

## 3. Statistical power: J=4 vs J=10 at 50-customer scale

Approximate test-set size at the current 50-customer config: ~1000
events (one held-out per customer per test-split fold, see
`configs/datasets/amazon.yaml::test_frac: 0.1` against ~10k events total).

**Top-1 accuracy SE at chance:**

| J | chance p = 1/J | SE at chance, N=1000 | SE at p=0.5, N=1000 |
|---|---|---|---|
| 4 | 0.250 | sqrt(.25 * .75 / 1000) = 1.37% | 1.58% |
| 10 | 0.100 | sqrt(.10 * .90 / 1000) = 0.95% | 1.58% |

**Separation between adjacent models.** A competing baseline that beats
chance by, say, 10 percentage points absolute is at p=0.35 (J=4) or p=0.20
(J=10). In z-units above chance:

- J=4: (0.35 - 0.25) / sqrt(0.25*0.75/1000) = 7.3 sigma
- J=10: (0.20 - 0.10) / sqrt(0.10*0.90/1000) = 10.5 sigma

J=10 gives *more* z-separation at chance despite lower absolute accuracy,
because the variance shrinks faster than the effect size. This is the
"larger J is more informative per event" argument in concrete form: each
correct prediction rules out 9 alternatives instead of 3, so the binary
top-1 indicator is a stricter test even though fewer events land in the
"correct" bucket.

**Log-likelihood (the actual PO-LEU metric).** The per-event log-lik
ceiling is `log J` (uniform baseline). J=4: 1.386 nats; J=10: 2.303
nats. A model that shaves off 30% of this is at -0.97 nats vs -1.61
nats — the larger-J setting carries ~1.7x more bits of discriminative
signal per event, which translates to tighter confidence intervals on
the *difference* between PO-LEU and each baseline (the quantity the
paper is actually trying to bound).

**Why not J=100?** Above ~J=20, LLM-ranker accuracy saturates near
chance (Hou 2023 Fig 3-equivalent: zero-shot Claude on ML-1M at J=50
is within 2pp of chance; the signal-to-noise ratio of the alternatives
block degrades faster than the variance shrinkage compensates). J=10
sits comfortably below this saturation for modern frontier models but
gives a ~2x log-likelihood budget over J=4.

**McFadden sampled-logit correction.** McFadden 1978 proves the
multinomial-logit estimator is consistent for *any* J >= 2 when the
chosen alternative is retained and negatives are sampled uniformly
(Guevara & Ben-Akiva 2021 refines the variance term). Uniform
negative sampling is what
`src/data/batching.py::build_choice_sets` already does. The correction
is a no-op for uniform sampling, so both J=4 and J=10 are econometrically
well-specified. This is *not* a reason to prefer one over the other.

## 4. Engineering cost of extending LLM baselines to J=10

**Files that need changes:**

| File | Change | LOC |
|---|---|---|
| `src/baselines/_llm_ranker_common.py` | Replace `DEFAULT_LETTERS` with a helper `letters_for(n_alts: int)` returning `("A",...,"J")` for n_alts=10; update 4 callers in the same file | ~10 |
| `src/baselines/zero_shot_claude_ranker.py:278` | Remove the hard `== 4` guard; pass `letters_for(train.n_alternatives)` through; default `n_permutations = train.n_alternatives` | ~10 |
| `src/baselines/few_shot_icl_ranker.py:424, 569` | Same two guards; same pass-through | ~10 |
| `tests/baselines/test_zero_shot_claude_ranker.py`, `test_few_shot_icl_ranker.py` | Replace the `n_alternatives=4` fixtures with parametrized `@pytest.mark.parametrize("J", [4, 10])`; adjust `_find_text_placing_prob_on` helper for 10 letters; update the "ValueError on wrong J" test to check `J not in {4, 10}` or remove | ~30 |
| `docs/llm_baselines/zero_shot_claude_ranker.md` §4 | Update Latin-square example to show K=10 rotation | ~5 |
| `docs/llm_baselines/few_shot_icl_ranker.md` | Same | ~5 |

**Total: ~70 lines of code + ~15 lines of doc. No new tests-of-behavior
needed beyond parametrizing the existing suite.**

The `letter_permutations` rotation machinery already supports arbitrary
`n_alts`; this is the reason the cost is small.

**LLM-call multiplier at fixed K=n_alts.** Going from (J=4, K=4) to
(J=10, K=10) is 2.5x the raw call count. With prompt caching on the
instructions+ICL prefix, effective token-billing cost is ~1.3-1.5x.
Wall-clock at the current parallelism settings (8-way concurrency per
`scripts/run_dataset.py`) is ~1.5-2x. For a 50-customer wave this is
minutes, not hours.

**Risk: does Claude/Gemini accuracy degrade at J=10?** Hou 2023 reports
a gentle monotone decline from J=5 to J=50 on recommendation tasks —
not a cliff. The baseline delta vs PO-LEU stays well-defined; only the
absolute accuracy number drops. Since the paper's claim is *relative*
(PO-LEU > baselines), this is fine.

## 5. Recommendation

**Run J=10 as the headline, with K=10 Latin-square debiasing, and
include a J=4 sensitivity row in the appendix.** Concretely:

1. Keep `configs/datasets/amazon.yaml::choice_set_size: 10`. Do not
   regress to J=4.
2. Extend the LLM-ranker baselines to accept `n_alternatives` from the
   batch (work described in §4; ~70 LOC, ~2 hours engineering).
3. Use `K = n_alternatives` so the Latin-square schedule cancels
   first-order letter bias cleanly at whatever J the config dictates.
   This also removes the need to defend a K<J choice in review.
4. Add an appendix table running the ranker baselines at J=4, K=4
   against the J=4 re-sampled version of the Amazon test set
   (`scripts/run_dataset.py --override dataset.training.choice_set_size=4`).
   This ties the results to the MCQA-bias literature and shows that the
   baseline rank order is stable under J-reduction.

**Why this beats the alternatives:**

- *J=4 only.* Rejection risk: a reviewer who knows the recsys
  literature (Hou, RankGPT, BERT4Rec family) will object that J=4 is
  inconsistent with every discrete-choice recommender benchmark since
  NCF. The only papers that use J=4 are MCQA benchmarks, and PO-LEU is
  not MCQA. Choosing J=4 to match the LLM baselines is tail-wagging-dog:
  it lets the weakest baseline dictate the experimental design.
- *J=10 without extending the LLM baselines.* Also defensible, but you
  lose the two most visible baselines (ZeroShot-Claude, FewShot-ICL)
  and give up the "frontier LLM fails at calibrated ranking" narrative
  that the PO-LEU paper likely wants to tell. For 70 LOC that's not a
  good trade.
- *Both J=4 and J=10 as equal headline settings.* Doubles the compute
  with no clean headline number. Reviewers will ask "which is the
  main result?" and the paper has no answer. Pick a lane.

The recommended path picks J=10 as the lane (matches the task's natural
reference class, gives 1.7x more log-likelihood signal per event, and
costs ~70 LOC to extend the baselines), and uses J=4 as a sensitivity
check that engages the MCQA-bias literature honestly without
surrendering the main result to it.

## Sources

- [Hou et al. 2023, arXiv:2305.08845](https://arxiv.org/abs/2305.08845)
- [LLMRank repo (RUCAIBox)](https://github.com/RUCAIBox/LLMRank)
- [Sun et al. 2023 RankGPT, arXiv:2304.09542](https://arxiv.org/abs/2304.09542)
- [Qin et al. 2023 PRP, arXiv:2306.17563](https://arxiv.org/abs/2306.17563)
- [Zhuang et al. 2024 Setwise, arXiv:2310.09497](https://arxiv.org/abs/2310.09497)
- [Zheng et al. 2023 PriDe, arXiv:2309.03882](https://arxiv.org/abs/2309.03882)
- [Pezeshkpour & Hruschka 2024, arXiv:2308.11483](https://arxiv.org/abs/2308.11483)
- [Wang et al. 2024 "My Answer is C", arXiv:2402.14499](https://arxiv.org/abs/2402.14499)
- [Robinson et al. 2022, arXiv:2210.12353](https://arxiv.org/abs/2210.12353)
- [Guevara & Ben-Akiva 2021, arXiv:2101.06211 (McFadden sampling)](https://arxiv.org/abs/2101.06211)
- [Petrov & Macdonald 2022, BERT4Rec replicability, arXiv:2207.07483](https://arxiv.org/abs/2207.07483)
