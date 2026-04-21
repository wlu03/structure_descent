# Structure Descent
### LLM-Guided Utility Model Search for Interpretable Discrete Choice

Replication of the Structure Descent framework on the Amazon e-commerce dataset. The framework discovers interpretable utility functions for predicting consumer purchase behavior by combining LLM-guided structural search with gradient-based weight fitting.

---

## How it works

Structure Descent separates two optimization problems:

| Problem | Space | Tool |
|---|---|---|
| Which factors appear in the utility function? | Discrete, combinatorial | LLM-guided search |
| How much should each factor weigh? | Continuous | L-BFGS gradient optimization |

The **outer loop** prompts an LLM with a diagnostic report of the current model's failures and asks it to propose structural changes to the DSL. The **inner loop** fits hierarchical weights for any fixed structure. A proposal is accepted only if it improves the Bayesian posterior score — the LLM proposes, the math decides.

---

## Project structure

```
structure_descent/
├── amazon_ecom/                    # Raw data (not committed)
│   ├── amazon-purchases.csv        # 1.85M purchase transactions
│   ├── survey.csv                  # 5,027 respondent survey responses
│   └── fields.csv                  # Survey field descriptions
├── notebooks/
│   ├── 00_data_exploration.ipynb   # EDA on both datasets
│   ├── 01_data_preparation.ipynb   # Build choice sets, temporal split
│   ├── 02_dsl_features.ipynb       # Extract feature matrices
│   ├── 03_inner_loop.ipynb         # Fit hierarchical weights
│   ├── 04_outer_loop_llm.ipynb     # Run Structure Descent
│   └── 05_evaluation.ipynb         # Full evaluation + ablations
├── src/
│   ├── data_prep.py                # Data loading and preprocessing
│   ├── dsl.py                      # DSL feature functions + DSLStructure
│   ├── inner_loop.py               # Hierarchical weight fitting (L-BFGS)
│   ├── outer_loop.py               # LLM proposal + accept/reject
│   └── evaluation.py               # Metrics, baselines, ablations
├── data/                           # Processed outputs (created by notebooks)
├── .env                            # API keys and provider config (not committed)
├── .env.example                    # Template for .env
└── requirements.txt
```

---

## Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/wlu03/structure_descent
cd structure_descent
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name structure_descent --display-name "Structure Descent"
```

### 5. Configure your LLM provider

Copy the example env file and fill in your settings:

```bash
cp .env.example .env
```

#### Option A — Anthropic Claude (default)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-6
```

#### Option B — Ollama (local LLM)

First install and start Ollama:

```bash
# Install from https://ollama.com  (or: brew install ollama)

# Start the server
ollama serve

# Pull a model (in a separate terminal)
ollama pull llama3        # 4.7 GB — recommended
ollama pull mistral       # 4.1 GB — strong JSON output
ollama pull qwen2.5:7b    # 4.4 GB — strong instruction following
```

Then set your `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Switching providers at any time only requires changing `LLM_PROVIDER` in `.env`. All other code is unchanged.

---

## Running the notebooks

Start Jupyter and select the **"Structure Descent"** kernel when prompted:

```bash
jupyter notebook
```

Run notebooks in order:

| Notebook | What it does |
|---|---|
| `00_data_exploration.ipynb` | EDA — distributions, categories, purchase volume |
| `01_data_preparation.ipynb` | Clean data, compute state features, build choice sets, temporal split → saves to `data/` |
| `02_dsl_features.ipynb` | Extract `[n_alts × n_terms]` feature matrices from choice events → saves `train_features.pkl` |
| `03_inner_loop.ipynb` | Fit hierarchical weights θ_g + θ_c + Δ_i for the initial structure → saves `current_state.pkl` |
| `04_outer_loop_llm.ipynb` | Run Structure Descent (LLM proposals + accept/reject) → saves `final_structure.pkl` |
| `05_evaluation.ipynb` | Full evaluation: metrics, breakdowns, posterior predictive checks, baselines, ablations |

> **Tip:** `MAX_EVENTS = 10_000` in notebook 02 lets you run quickly on a subset. Set to `None` for the full 1.85M row dataset.

---

## Evaluation

The full evaluation in `05_evaluation.ipynb` covers every method from the paper:

**Predictive metrics** — top-1, top-5, MRR, NLL broken down by:
- Product category
- Repeat vs. novel purchases
- User activity level (low / medium / high)
- Time of day

**Posterior predictive checks** — simulate sequences from the fitted model and compare to real data:
- Repeat purchase rate
- Category switching matrix P(cat_{t+1} | cat_t)
- Inter-purchase gap distribution
- Price trajectory
- Brand loyalty index (HHI)

**Baselines:**
- Frequency (always predict most-purchased item)
- Markov chain (category transition probabilities)
- Standard conditional logit (hand-specified features, no search)
- Structure Descent (final discovered structure)

**Ablations:**
- Structure search without LLM (random DSL proposals)
- No hierarchy (single global weights, no θ_c or Δ_i)
- TextGrad without priors (unconstrained LLM, no grammar)

---

## DSL reference

The utility function is a weighted sum of DSL terms: `u(a | s_t) = Σ θ_k · f_k(a, s_t)`

**Layer 1 — Universal behavioral primitives**

| Term | Meaning |
|---|---|
| `routine` | How many times the customer previously bought this item |
| `recency` | Inverse days since last purchase of this item |
| `novelty` | 1 if item never purchased by this customer |
| `popularity` | Log-scaled global purchase frequency |
| `affinity` | Log-scaled prior purchases in this category |
| `time_match` | Whether this category is typical at this time of day |

**Layer 2 — Amazon-specific features**

| Term | Meaning |
|---|---|
| `price_sensitivity` | Negative price ratio relative to category average |
| `rating_signal` | Star rating × log(review count) |
| `brand_affinity` | Log-scaled brand purchase history |
| `price_rank` | Cheapness relative to other items in session |
| `delivery_speed` | Prime / fast shipping indicator |
| `co_purchase` | Log-scaled co-purchase frequency with recent items |

**Layer 3 — Combinators**

| Term | Meaning |
|---|---|
| `interaction(a, b)` | Joint effect of two terms |
| `split_by(term, cond)` | Different weights under different conditions |
| `threshold(term, c)` | Binary indicator when term exceeds cutoff |
| `log_transform(term)` | Signed log scaling |
| `ratio(a, b)` | Ratio of two terms |
| `power(term, n)` | Signed power transform |
| `difference(a, b)` | Difference of two terms |

> **Note.** The `decay(term, halflife)` combinator has been removed. Its previous implementation looked up the *transformed* `recency` score in `[0, 1]` rather than raw days, which made the resulting exponential meaningless. Any writeup or experiment log that still references `decay` is stale — the current Layer 3 set is the seven combinators above.

---

## Paper

> *Structure Descent, LLM-Guided Utility Model Search for Interpretable Discrete Choice*
