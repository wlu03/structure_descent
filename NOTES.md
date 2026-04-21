# PO-LEU Build Log

Source of truth: [`docs/redesign.md`](docs/redesign.md). Every deviation below is
also a note to self about what the paper/codebase must disclose.

---

## Scaffold

**What was created**
- §14 directory tree (`src/{data,outcomes,model,train,eval,baselines}`, `configs/`, `tests/`, `scripts/`, `outcomes_cache/`, `embeddings_cache/`, `checkpoints/`, `reports/`).
- `pyproject.toml` (Python 3.11+, torch, numpy, pyyaml, sentence-transformers, tqdm, pandas, scikit-learn; `[dev]` extras for pytest; `[llm]` extras for anthropic).
- `configs/default.yaml` encoding every row of redesign.md Appendix B plus a few structural keys (paths, subsample block, eval strata list).
- `tests/conftest.py` with `synthetic_batch` fixture (B=4, J=10, K=3, d_e=768, p=26) and a `default_config` loader.
- `README.md` with the wave plan.
- `src/train/subsample.py` copied verbatim from `old_pipeline/src/subsample.py` (Appendix C).

**Spec decisions recorded here (not deviations, just places redesign.md did not fully specify)**
- Repo root is `/Users/wesleylu/Desktop/structure_descent/` rather than a nested `po-leu/` directory — §14 showed a project root by that name, but we are already inside the checkout. The relative layout under §14 is preserved.
- Default `outcomes.generator.model_id = "stub"` so hermetic tests never hit an API. A real-API client swap is documented in the final integration summary.
- Default `subsample.enabled = false` because the smoke test runs on synthetic data (Appendix C importance weights degenerate to 1.0).

---

### Orchestrator-level spec resolution: per-head / salience parameter-count reconciliation (raised in Wave 4)

§5.1 claims per-head count `768·128 + 128 + 128·1 + 1 = 98,433`; the actual
sum is **98,561** (the spec formula drops the `fc1` bias term). §7.1 has
the same bug: `794·64 + 64 + 64·1 + 1 = 50,881` but the actual sum is
**50,945**. §6.1's weight-net count (`1,029`) is arithmetically correct.
§0 states "Xavier-uniform for weights, zero init for biases" (biases are
present on every linear by convention), and the architecture tables list
two-Linear MLPs. The architecture is authoritative; the spec constants
are the outliers.

**Decision.** Two-Linear-with-biases architecture stands. Corrected
constants (used by unit tests and by the §13 AIC/BIC reporting):

| Module                      | Spec count (wrong) | Corrected count |
|-----------------------------|--------------------|-----------------|
| Attribute head (per-head)   | 98,433             | **98,561**       |
| Attribute stack (M=5)       | 492,165            | **492,805**      |
| Weight net                  | 1,029              | 1,029 (unchanged) |
| Salience net                | 50,881             | **50,945**       |
| **Total k (§9.4 / §13)**    | 544,075            | **544,779**      |

The updated `k = 544,779` must propagate into the Wave-6 AIC/BIC reporting.
Audit trail preserved; `docs/redesign.md` is not edited.

---

## Wave 1 — data prep + cache + prompts

### Orchestrator-level spec resolution: p = 26 reconciliation

§2.1 claims `p = 26` with the breakdown
`6 + 5 + 1 + 1 + 4 + 1 + 1 + 1 + 1 + 1`, but that sum is 22. The 26 is
load-bearing across §6.1 (weight-net input), §7.1 (salience-net input
`768 + 26 = 794`), §9.4 (weight-net parameter count `26·32 + 32 = 864`),
Appendix B (`p (effective) = 26`), and `tests/conftest.py`'s
`synthetic_batch` fixture. The breakdown arithmetic is the outlier.

**Decision.** Treat `p = 26` as authoritative; fix the §2.1 arithmetic by
encoding `household_size` as a **5-bin one-hot** over `{1, 2, 3, 4, 5+}`
rather than an "int, standardized" scalar. Final breakdown:

```
age_bucket (6) + income_bucket (5) + household_size (5) + has_kids (1)
+ city_size (4) + education (1) + health_rating (1) + risk_tolerance (1)
+ purchase_frequency (1) + novelty_rate (1) = 26   ✓
```

`household_size` is a small positive integer in real survey data, and
bucketing at {1,2,3,4,5+} is the standard interpretation. Other candidate
expansions (one-hot education or health_rating, widen age to 10 bins)
would contradict more of the spec's explicit "(K levels)" or "(K bins)"
claims. Recording here rather than editing §2.1 so the audit trail is
preserved.

---

**`src/outcomes/prompts.py`** (§3.2).
- `SYSTEM_PROMPT`, `USER_BLOCK_TEMPLATE`, `PROMPT_VERSION = "v1"` exposed as
  module constants; `build_system_prompt(K)`, `build_user_block(...)`, and
  `build_messages(...)` are the pure helpers.
- **K-substitution decision.** The spec called out a choice between a literal
  K baked into the system prompt vs. a `{K}` format placeholder. I kept the
  `{K}` placeholder in the *stored* template (so the frozen string in
  `prompts.py` remains a single source of truth and a regression is easy to
  eyeball against §3.2) and substitute via `str.format` inside
  `build_system_prompt`. This matches "substitute K into the fixed system
  prompt" in the spec and keeps the prompt cheap to version — when K later
  becomes an ablation knob (§11), no wrapper or second template is needed.
- `_render_optional_fields` emits `- key: value` lines and collapses to `""`
  when the mapping is empty/None, so the template's blank line before
  `Generate K=...` is preserved and no double blank line is introduced.
- Missing required alt fields (`title`, `category`, `price`,
  `popularity_rank`) and non-positive K both raise `ValueError`.
- Tests: `tests/test_prompts.py`, 14 cases, all green.

**`src/outcomes/cache.py`** (§3.4, §4.3).
- Shared `KVStore` over stdlib `sqlite3` — no new deps (diskcache/joblib
  explicitly avoided). Single `kv(key TEXT PRIMARY KEY, value BLOB NOT NULL,
  created_at REAL NOT NULL)` table, opened in WAL mode
  (`PRAGMA journal_mode=WAL`) so warm-cache reads stay concurrent while a
  writer holds the log; writes use `INSERT OR REPLACE` for idempotency.
- `OutcomesCache.outcomes_key` hashes fields in the §3.4 order —
  `customer_id || asin || seed || prompt_version` — with NUL-byte separators
  so concatenations cannot collide across field boundaries. Values are
  JSON-encoded `{"outcomes": [...], "metadata": {...}}` as UTF-8 bytes.
- `EmbeddingsCache.embedding_key` hashes `outcome_string || encoder_id` (§4.3);
  serialization uses `numpy.save` into an `io.BytesIO` buffer (the `.npy`
  format, which carries dtype and shape). `put_embedding` raises `ValueError`
  unless the array is 1-D `float32`.
- Thread-safety is documented, not enforced: SQLite connections are not
  thread-safe, so the contract is "one store per thread."
- Tests: `tests/test_cache.py`, 11 cases, all green.

**`src/data/context_string.py`** (§2.2).
- `DEFAULT_PHRASINGS` commits to phrasings the spec left open: age buckets
  → "early 20s / late 20s-early 30s / mid-30s / mid-40s / mid-50s / mid-60s or
  older"; income midpoints $15k / $35k / $60k / $110k / "above $150k"; city
  → "rural area / small town / mid-size city / large U.S. city"; education
  1-5 → "some high school … graduate degree"; health 1-5 → "poor … excellent
  health"; risk-tolerance thresholds ±0.5 → "cautious / risk-neutral /
  comfortable with risk"; purchase-frequency (count/week) bins <1 / 1-3 / >3
  → "a few times per month / roughly N times per week / almost daily";
  novelty-rate bins <0.2 / <0.5 / ≥0.5 → "rarely / occasionally / frequently
  tries new products".
- The §2.2 "<share> of orders are first-time products" clause uses a finer
  share-phrase family ("very few / about a fifth / about a third / about half
  / most / nearly all") so the rendered text never dumps the raw decimal
  (§2.2 rule 2, no column dumps).
- Kids clause: `has_kids=True` with household_size∈{≤3,4,≥5} →
  "one child / two children / several children"; `has_kids=False` → "no
  children". Three-person household with kids = one child (one parent + one
  partner + one child) was the cleanest interpretation.
- Signature is `(row, *, recent_purchases, current_time)` — deliberately no
  future-purchase parameter (§2.2 rule 1); `paraphrase_rules_check` enforces
  rule 2 at render time by rejecting raw column names and literal bucket
  codes, and is exported for test reuse.
- Tests: `tests/test_context_string.py`, 36 cases, all green.

**`src/data/person_features.py`** (§2.1, with orchestrator override).
- Implements the binding 26-dim breakdown: `age(6) + income(5) +
  household_size(5, one-hot) + has_kids(1) + city(4) + education(1) +
  health_rating(1) + risk_tolerance(1) + purchase_frequency(1) +
  novelty_rate(1)`. `household_size` is one-hot over
  `{"1","2","3","4","5+"}` (fixed vocabulary, not learned).
- **novelty_rate pass-through decision.** §2.1 row 10 lists novelty_rate as
  a rate in [0, 1] without "standardized"; kept it un-scaled so the feature
  stays on its natural [0, 1] scale and is directly human-readable. Only 4
  scalars are standardized (education, health_rating, risk_tolerance,
  purchase_frequency). The task brief mentions "5 standardized dims" but
  the binding table has only 4 standardized rows (10 is pass-through) —
  went with the table as the source of truth. `PersonFeatureStats.means`
  and `.stds` therefore have shape (4,).
- `_bucket_household_size(hs)` is a public helper: accepts ints and
  integer-valued floats; rejects bool, None, negative, non-finite, and
  non-integer floats. Integers `>= 5` fold into `"5+"`.
- Stats exposes the four categorical vocabularies plus means/stds and the
  26-element `feature_columns`; `to_dict`/`from_dict` support JSON
  round-trip (`standardized_columns` is serialized for documentation
  purposes and ignored by `from_dict`).
- Tests: `tests/test_person_features.py`, 11 cases, all green.

---

## Wave 2 — LLM generation + diversity filter

**`src/outcomes/diversity_filter.py`** (§3.5).
- `HashEmbedder(dim=64, seed=0)` is the deterministic stub encoder: SHA-256
  of character 3-grams (with `\x02`/`\x03` sentinel padding so ≤2-char inputs
  still emit a gram) mods into `dim` buckets, a second seeded SHA-256 picks
  signs in `{-1, +1}`, and rows are L2-normalized (all-zero fallback snaps to
  a deterministic unit basis vector). Numpy + stdlib only; no
  sentence-transformers, torch, or sklearn.
- Threshold default is `0.9` (§3.5). `find_paraphrase_pair` scans the strict
  upper triangle in row-major order and returns the first `(i, j)` with
  `cos_sim > threshold`, or `None`.
- `diversity_filter(outcomes)` returns `(outcomes, ok)` — identity-preserving,
  `ok == (no paraphrase found)`. The module never regenerates; `generate.py`
  owns the "at most 2 retries" loop from §3.5.
- `EmbeddingFn` is a `typing.Protocol` (runtime-checkable). Wave 3 will ship a
  real `all-mpnet-base-v2`-backed encoder that plugs into the same protocol,
  at which point the `embed_fn=None` default can be swapped without touching
  this module.
- Tests: `tests/test_diversity_filter.py`, 16 cases, all green.

**`src/outcomes/generate.py`** (§3.3, §3.4).
- `StubLLMClient` determinism: the completion is a `"\n".join` of one phrasing
  per attribute family (`financial`, `health`, `convenience`, `emotional`,
  `social`, in that order). A SHA-256 over `repr(messages) || "\x00" || seed`
  drives the selection — byte `i` of the digest, mod the family size, picks
  phrasing `i`. Same `(messages, seed)` → same text; distinct seeds flip at
  least one byte of the digest with overwhelming probability, so the
  per-family selections diverge.
- `max_retries` is threaded into the generate loop by calling
  `client.generate(..., seed=seed + attempt_idx)` for `attempt_idx in
  range(max_retries + 1)`. A fresh seed per attempt means the stub actually
  produces different text on retry; the `seed` recorded in metadata is the
  final bumped seed, while the cache is keyed on the caller's *base* `seed`
  so a re-call with the same base seed still hits the cache (§3.4).
- Anthropic client: imported lazily inside `__init__` (and again defensively
  at the top of `generate`). `import src.outcomes.generate` therefore never
  touches the `anthropic` package — `ImportError` surfaces only on
  instantiation. API key resolution reads `ANTHROPIC_API_KEY` inside the
  constructor when `api_key is None`, never at module load; system prompts
  are split off and forwarded via the SDK's top-level `system=` kwarg (the
  SDK does not accept `role="system"` in `messages`). `seed` is not
  forwarded because the Anthropic API has no seed parameter today.
- Sentinel padding: `parse_completion` splits on `"\n"`, strips, drops
  empties, keeps the first `K`; shortfalls are padded with
  `SENTINEL_OUTCOME = "no additional consequence."` (exact §3.3 wording).
  Every pad emits a stdlib `logger.warning` that includes the
  `(customer_id, asin)` context the caller threads through.
- Metadata recorded per cache write: `temperature`, `top_p`, `max_tokens`,
  `model_id`, `finish_reason`, `seed`, `prompt_version`, `timestamp`
  (`time.time()`). Covers row 1 of Reproducibility Checklist §15.
- Tests: `tests/test_generate.py`, 11 cases, all green.

---

## Wave 3 — encoder

**`src/outcomes/encode.py`** (§4).
- `EncoderClient` is a `typing.Protocol` (runtime-checkable) exposing
  `encoder_id`, `d_e`, and `encode(texts) -> (N, d_e) float32` with rows
  L2-normalized. Two concrete clients satisfy it.
- `StubEncoder` (default, hermetic): per-string seed via
  `blake2b(encoder_id || "\x00" || text, digest_size=8)` big-endian
  → fresh `np.random.default_rng(seed)` draws `d_e` Gaussian samples,
  then L2-normalize. Numpy + stdlib only; same string → same vector;
  distinct inputs diverge far below the 0.999 cosine bar because every
  byte of the blake2b digest drives a different RNG state.
- `SentenceTransformersEncoder` (real, optional): `sentence_transformers`
  is imported **lazily** inside `__init__` so importing this module
  never loads torch. `encoder_id` is `f"{model_id}|pooling={pooling}|
  max_length={max_length}"` — changing any of the three invalidates
  the §4.3 cache. `encode()` delegates to the model with
  `normalize_embeddings=True` (§4.2 step 4) and 64-token truncation.
- `encode_batch` layers the §4.3 `EmbeddingsCache` on top of any
  client: cache-misses are encoded in **one** batch, results are
  written back, and the full tensor is reassembled in original input
  order. `cache=None` is a pure pass-through with no side effects;
  otherwise a second call with the same texts issues zero new
  encodings (verified by `test_encode_batch_cache_miss_then_hit`).
- `encode_outcomes_tensor` flattens `[B][J][K]` → `B*J*K` strings,
  reuses `encode_batch`, reshapes to `(B, J, K, d_e)`. Ragged inputs
  on either the J or K axis raise `ValueError`.
- Tests: `tests/test_encode.py`, 11 cases, all green. All tests use
  `StubEncoder` — `SentenceTransformersEncoder` is never instantiated
  in the suite.

---

## Wave 4 — model modules

**`src/model/weight_net.py`** (§6).
- `WeightNet(p=26, M=5, hidden=32, normalization="softmax")`: Linear(p→32) →
  ReLU → Linear(32→5) → normalization, matching §6.1 exactly.
- `normalization` is a simple switch. `"softmax"` (default) uses
  `nn.Softmax(dim=-1)`; `"softplus"` applies `softplus(raw) /
  softplus(raw).sum(-1, keepdim=True)` — the A4 ablation (§6.2 / §11) without
  a second module. Any other string raises `ValueError`.
- Xavier-uniform linear weights, zero biases (§0 convention). Final-layer
  bias is retained (spec is silent).
- `EXPECTED_PARAM_COUNT_DEFAULT = 1_029` exposed as a module-level constant
  and asserted in `test_param_count_default`: `26*32 + 32 + 32*5 + 5`.
- Forward contract: `(B, 26) → (B, 5)`, rows sum to 1.0 (atol 1e-6) under
  both normalization modes.
- Tests: `tests/test_weight_net.py`, 12 cases, all green.

**`src/model/salience_net.py`** (§7, with orchestrator override).
- `SalienceNet(d_e=768, p=26, hidden=64)`: `Linear(794 → 64) → ReLU →
  Linear(64 → 1)`, both Linears carry biases per §0's "zero init for
  biases" convention. Xavier-uniform weights, zero biases.
- `forward(E, z_d)`: broadcasts `z_d` to `(B, J, K, p)` via
  `z_d[:, None, None, :].expand(...)`, concatenates with `E` to
  `(B, J, K, 794)`, runs the MLP to `(B, J, K, 1)`, squeezes, then
  softmaxes over `K` — softmax is applied AFTER the MLP scalar score,
  not before, so rows sum to 1 along `K` within each `(b, j)`.
- `EXPECTED_PARAM_COUNT_DEFAULT = 50_945` (corrected from the spec's
  arithmetic-wrong 50_881, per the §7.1 reconciliation block above);
  asserted in `test_param_count_default`.
- `UniformSalience` ablation A6 (§7.3): same signature, returns `1/K`
  per entry, zero trainable params.
- Tests: `tests/test_salience_net.py`, 11 cases, all green.

**`src/model/attribute_heads.py`** (§5, with orchestrator override).
- Confirmed: both Linears carry biases (`nn.Linear` default) per §0
  "Xavier-uniform for weights, zero init for biases" — applies to both
  `fc1` (d_e → hidden) and `fc2` (hidden → 1). §5.1's headline count of
  `98,433` drops the `fc1` bias term; the actual sum
  `768·128 + 128 + 128·1 + 1 = 98,561` is authoritative per the
  orchestrator resolution block above.
- `AttributeHead`: Linear(768→128) → ReLU → Linear(128→1), Xavier-uniform
  weights, zero biases, no dropout / layernorm / residuals.
- `AttributeHeadStack(M=5)` holds M independent heads in `nn.ModuleList`
  (per §5.1 "M independent small MLPs"; no fused linear). Person-
  independent (§5.3): `forward(E)` takes only the embedding tensor;
  passing `z_d` as an extra positional raises `TypeError`.
- Shape contract: `(B, J, K, d_e) → (B, J, K, M)` via last-axis concat of
  each head's `(..., 1)` output.
- `EXPECTED_PARAM_COUNT_PER_HEAD = 98_561` and
  `EXPECTED_PARAM_COUNT_STACK_M5 = 492_805` are asserted at the corrected
  values in `test_single_head_param_count` / `test_stack_param_count_default`.
- Tests: `tests/test_attribute_heads.py`, 9 cases, all green.

---

## Wave 5 — assembly + ablations

(pending)

---

## Wave 6 — training + eval

(pending)

---

## Wave 7 — interpretability + ablation configs

(pending)

---

## Final integration

(pending)
