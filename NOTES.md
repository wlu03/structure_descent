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

(pending)

---

## Wave 3 — encoder

(pending)

---

## Wave 4 — model modules

(pending)

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
