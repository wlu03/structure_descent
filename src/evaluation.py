"""
Evaluation metrics for Structure Descent — Amazon e-commerce domain.

Predictive metrics (paper Section "Evaluation"):
  top-1, top-5, MRR, NLL on held-out sequences
  Broken down by: category, time of day, new vs. repeat, user activity level

Posterior predictive checks (most important for claiming behavioral model):
  repeat_purchase_rate, category_switching_matrix, inter_purchase_gap,
  price_trajectory, brand_loyalty_index

Baselines + ablations:
  frequency, markov_chain, no_hierarchy, random_structure_search
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from typing import Callable, Dict, List, Optional, Tuple


def compute_metrics(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    weight_fn: Callable[[str, str], np.ndarray],
    customer_ids: List[str],
    categories: List[str],
) -> Dict[str, float]:
    """
    Compute top-1, top-5, MRR, and per-event NLL on a held-out set.
    """
    top1 = top5 = 0
    rr_list = []
    total_nll = 0.0
    n = len(features_list)

    for feats, chosen, cid, cat in zip(features_list, chosen_indices, customer_ids, categories):
        w = weight_fn(cid, cat)
        utils = feats @ w
        probs = softmax(utils)
        rank = int(np.sum(utils > utils[chosen]))
        top1 += rank == 0
        top5 += rank < 5
        rr_list.append(1.0 / (rank + 1))
        total_nll -= np.log(probs[chosen] + 1e-12)

    return {
        "top1": top1 / n,
        "top5": top5 / n,
        "mrr": float(np.mean(rr_list)),
        "val_nll": total_nll / n,
        "n_events": n,
    }


def breakdown_by_category(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    weight_fn: Callable[[str, str], np.ndarray],
    customer_ids: List[str],
    categories: List[str],
) -> pd.DataFrame:
    """
    Top-1 accuracy and NLL broken down by product category.
    Paper: "Broken down by: context (category)..."
    """
    cat_data: Dict[str, Dict] = {}

    for feats, chosen, cid, cat in zip(features_list, chosen_indices, customer_ids, categories):
        w = weight_fn(cid, cat)
        utils = feats @ w
        probs = softmax(utils)
        rank = int(np.sum(utils > utils[chosen]))

        if cat not in cat_data:
            cat_data[cat] = {"top1": 0, "nll": 0.0, "n": 0}
        cat_data[cat]["top1"] += rank == 0
        cat_data[cat]["nll"] -= np.log(probs[chosen] + 1e-12)
        cat_data[cat]["n"] += 1

    rows = []
    for cat, d in sorted(cat_data.items(), key=lambda x: -x[1]["n"]):
        rows.append({
            "category": cat,
            "top1": d["top1"] / d["n"],
            "nll": d["nll"] / d["n"],
            "n_events": d["n"],
        })
    return pd.DataFrame(rows)


def breakdown_by_repeat_vs_novel(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    weight_fn: Callable[[str, str], np.ndarray],
    customer_ids: List[str],
    categories: List[str],
    metadata: List[dict],
) -> pd.DataFrame:
    """
    Top-1 accuracy split by repeat (previously purchased) vs. novel items.
    Paper: "Broken down by: new vs. repeat choices"
    """
    buckets: Dict[str, Dict] = {"repeat": {"top1": 0, "n": 0}, "novel": {"top1": 0, "n": 0}}

    for feats, chosen, cid, cat, meta in zip(
        features_list, chosen_indices, customer_ids, categories, metadata
    ):
        w = weight_fn(cid, cat)
        utils = feats @ w
        rank = int(np.sum(utils > utils[chosen]))
        key = "repeat" if meta.get("is_repeat", False) else "novel"
        buckets[key]["top1"] += rank == 0
        buckets[key]["n"] += 1

    rows = []
    for key, d in buckets.items():
        rows.append({"type": key, "top1": d["top1"] / max(d["n"], 1), "n_events": d["n"]})
    return pd.DataFrame(rows)


def breakdown_by_activity_level(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    weight_fn: Callable[[str, str], np.ndarray],
    customer_ids: List[str],
    categories: List[str],
    purchase_counts: Dict[str, int],
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Top-1 accuracy by user activity level (low / medium / high purchase frequency).
    Paper: "Broken down by: user activity level"

    purchase_counts: dict mapping customer_id -> total number of purchases
    """
    all_counts = np.array(list(purchase_counts.values()))
    quantiles = np.quantile(all_counts, np.linspace(0, 1, n_bins + 1))
    labels = ["low", "medium", "high"][:n_bins]

    def get_bucket(cid: str) -> str:
        c = purchase_counts.get(cid, 0)
        for i in range(n_bins - 1, -1, -1):
            if c >= quantiles[i]:
                return labels[min(i, n_bins - 1)]
        return labels[0]

    buckets: Dict[str, Dict] = {lbl: {"top1": 0, "n": 0} for lbl in labels}

    for feats, chosen, cid, cat in zip(features_list, chosen_indices, customer_ids, categories):
        w = weight_fn(cid, cat)
        utils = feats @ w
        rank = int(np.sum(utils > utils[chosen]))
        b = get_bucket(cid)
        buckets[b]["top1"] += rank == 0
        buckets[b]["n"] += 1

    rows = []
    for lbl in labels:
        d = buckets[lbl]
        rows.append({"activity_level": lbl, "top1": d["top1"] / max(d["n"], 1), "n_events": d["n"]})
    return pd.DataFrame(rows)


def breakdown_by_time_of_day(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    weight_fn: Callable[[str, str], np.ndarray],
    customer_ids: List[str],
    categories: List[str],
    order_hours: List[int],
) -> pd.DataFrame:
    """
    Top-1 accuracy broken down by hour-of-day bucket (morning/afternoon/evening/night).
    Paper: "Broken down by: time of day"

    order_hours: list of int (0–23), one per event
    """
    def hour_bucket(h: int) -> str:
        if 6 <= h < 12:   return "morning (6-12)"
        if 12 <= h < 18:  return "afternoon (12-18)"
        if 18 <= h < 23:  return "evening (18-23)"
        return "night (23-6)"

    buckets: Dict[str, Dict] = {}
    for feats, chosen, cid, cat, h in zip(
        features_list, chosen_indices, customer_ids, categories, order_hours
    ):
        w = weight_fn(cid, cat)
        utils = feats @ w
        rank = int(np.sum(utils > utils[chosen]))
        b = hour_bucket(h)
        if b not in buckets:
            buckets[b] = {"top1": 0, "n": 0}
        buckets[b]["top1"] += rank == 0
        buckets[b]["n"] += 1

    rows = []
    for b, d in buckets.items():
        rows.append({"time_of_day": b, "top1": d["top1"] / max(d["n"], 1), "n_events": d["n"]})
    return pd.DataFrame(rows).sort_values("time_of_day")


def compute_residuals(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    weight_fn: Callable[[str, str], np.ndarray],
    customer_ids: List[str],
    categories: List[str],
    metadata: List[dict],
    top_k_worst: int = 3,
) -> Dict[str, str]:
    """
    Compact residual summary for the LLM diagnostic report.
    Returns human-readable strings describing where the model fails.
    """
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}
    repeat_correct = repeat_total = 0
    novel_correct = novel_total = 0

    for feats, chosen, cid, cat, meta in zip(
        features_list, chosen_indices, customer_ids, categories, metadata
    ):
        w = weight_fn(cid, cat)
        utils = feats @ w
        is_correct = int(np.sum(utils > utils[chosen]) == 0)

        cat_correct[cat] = cat_correct.get(cat, 0) + is_correct
        cat_total[cat] = cat_total.get(cat, 0) + 1

        if meta.get("is_repeat", False):
            repeat_correct += is_correct
            repeat_total += 1
        else:
            novel_correct += is_correct
            novel_total += 1

    results = {}
    cat_acc = {c: cat_correct[c] / cat_total[c] for c in cat_correct}
    for cat, acc in sorted(cat_acc.items(), key=lambda x: x[1])[:top_k_worst]:
        results[f"Category '{cat}' top-1"] = f"{acc:.1%}  (n={cat_total[cat]})"

    if repeat_total > 0:
        results["Repeat purchase top-1"] = f"{repeat_correct / repeat_total:.1%}  (n={repeat_total})"
    if novel_total > 0:
        results["Novel purchase top-1"] = f"{novel_correct / novel_total:.1%}  (n={novel_total})"

    return results


def simulate_sequences(
    df: pd.DataFrame,
    weight_fn: Callable[[str, str], np.ndarray],
    structure,
    extract_features_fn: Callable,
    n_steps: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate synthetic purchase sequences from the fitted model.
    Required for posterior predictive checks.

    For each customer, starting from their last observed state, sample
    n_steps purchases from the conditional logit distribution.

    Returns a DataFrame with the same schema as the real purchase data.
    """
    rng = np.random.default_rng(seed)
    records = []

    cat_to_asins = {cat: grp["asin"].unique().tolist() for cat, grp in df.groupby("category")}
    catalog = df["asin"].unique()

    for cid, cust_df in df.groupby("customer_id"):
        cust_df = cust_df.sort_values("order_date")
        last_date = cust_df["order_date"].max()
        category = cust_df["category"].iloc[-1]

        for step in range(n_steps):
            # Build a small candidate set: same-cat + random
            same_cat = cat_to_asins.get(category, list(catalog[:20]))
            candidates = list(rng.choice(same_cat, size=min(5, len(same_cat)), replace=False))
            candidates += list(rng.choice(catalog, size=5, replace=False))
            candidates = list(set(candidates))[:10]

            # Build dummy events for each candidate
            feats = np.zeros((len(candidates), len(structure.terms)))
            for k, asin in enumerate(candidates):
                prior = cust_df[cust_df["asin"] == asin]
                feats[k, 0] = len(prior)  # routine proxy

            w = weight_fn(cid, category)
            utils = feats @ w
            probs = softmax(utils)
            chosen_k = int(rng.choice(len(candidates), p=probs))
            chosen_asin = candidates[chosen_k]

            sim_date = last_date + pd.Timedelta(days=int(rng.integers(1, 30)))
            last_date = sim_date

            chosen_row = cust_df[cust_df["asin"] == chosen_asin]
            price = float(chosen_row["price"].iloc[0]) if len(chosen_row) > 0 else 10.0
            brand = chosen_row["brand"].iloc[0] if "brand" in chosen_row.columns and len(chosen_row) > 0 else "unknown"

            records.append({
                "customer_id": cid,
                "order_date": sim_date,
                "asin": chosen_asin,
                "category": category,
                "price": price,
                "brand": brand,
            })

    return pd.DataFrame(records)


def category_switching_matrix(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute P(cat_{t+1} | cat_t) from a purchase sequence DataFrame.
    Paper: "Category switching matrix — Does P(cat_{t+1} | cat_t) match real purchase chains?"

    Returns a normalized transition matrix (rows sum to 1).
    """
    df = df.sort_values(["customer_id", "order_date"])
    df["next_cat"] = df.groupby("customer_id")["category"].shift(-1)
    df = df.dropna(subset=["next_cat"])

    # Limit to top_n categories by frequency
    top_cats = df["category"].value_counts().head(top_n).index.tolist()
    df = df[df["category"].isin(top_cats) & df["next_cat"].isin(top_cats)]

    counts = pd.crosstab(df["category"], df["next_cat"])
    # Normalize rows
    transition = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return transition


def posterior_predictive_checks(
    real_df: pd.DataFrame,
    simulated_df: pd.DataFrame,
    top_n_cats: int = 10,
) -> pd.DataFrame:
    """
    Compare distributional statistics between real and simulated sequences.

    Both DataFrames must have columns: customer_id, order_date, asin, category, price.

    Statistics (from paper Table — Amazon domain):
      - repeat_purchase_rate
      - category_switching_matrix (summarized as mean diagonal — self-transition rate)
      - inter_purchase_gap_days
      - avg_price_per_session  (price trajectory)
      - brand_loyalty_index (HHI over brands per customer)
    """
    stats = {}

    # 1. Repeat purchase rate
    def repeat_rate(df: pd.DataFrame) -> float:
        return df.groupby("customer_id")["asin"].apply(
            lambda x: x.duplicated().mean()
        ).mean()

    stats["repeat_purchase_rate"] = {
        "real": repeat_rate(real_df),
        "simulated": repeat_rate(simulated_df),
        "what_it_tests": "Fraction of purchases that are re-buys of the same product",
    }

    # 2. Category switching: summarized as mean self-transition probability
    def self_transition_rate(df: pd.DataFrame) -> float:
        mat = category_switching_matrix(df, top_n=top_n_cats)
        diag_vals = [mat.loc[c, c] for c in mat.index if c in mat.columns]
        return float(np.mean(diag_vals)) if diag_vals else float("nan")

    stats["category_self_transition_rate"] = {
        "real": self_transition_rate(real_df),
        "simulated": self_transition_rate(simulated_df),
        "what_it_tests": "Does P(cat_{t+1}|cat_t) match real purchase chains?",
    }

    # 3. Inter-purchase gap (days)
    def mean_gap(df: pd.DataFrame) -> float:
        return (
            df.sort_values(["customer_id", "order_date"])
            .groupby("customer_id")["order_date"]
            .diff()
            .dt.days
            .dropna()
            .mean()
        )

    stats["inter_purchase_gap_days"] = {
        "real": mean_gap(real_df),
        "simulated": mean_gap(simulated_df),
        "what_it_tests": "Time between consecutive purchases",
    }

    # 4. Price trajectory (avg spend per customer per session)
    def avg_price(df: pd.DataFrame) -> float:
        return df.groupby("customer_id")["price"].mean().mean()

    stats["avg_price_per_session"] = {
        "real": avg_price(real_df),
        "simulated": avg_price(simulated_df),
        "what_it_tests": "Does average spend per session match observed patterns?",
    }

    # 5. Brand loyalty index (HHI — higher = more concentrated within brands)
    def brand_hhi(df: pd.DataFrame) -> float:
        if "brand" not in df.columns:
            return float("nan")

        def hhi(x: pd.Series) -> float:
            shares = x.value_counts(normalize=True)
            return float((shares ** 2).sum())

        return df.groupby("customer_id")["brand"].apply(hhi).mean()

    stats["brand_loyalty_index"] = {
        "real": brand_hhi(real_df),
        "simulated": brand_hhi(simulated_df),
        "what_it_tests": "How concentrated are purchases within brands?",
    }

    result = pd.DataFrame(stats).T[["real", "simulated", "what_it_tests"]]
    result["real"] = result["real"].astype(float).round(4)
    result["simulated"] = result["simulated"].astype(float).round(4)
    result["pct_diff"] = (
        (result["simulated"].astype(float) - result["real"].astype(float))
        / (result["real"].astype(float).abs() + 1e-9) * 100
    ).round(1)
    return result


def frequency_baseline(
    choice_events: List[dict],
    customer_purchase_history: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """
    Baseline: always predict the customer's most-purchased item.
    Paper: "Frequency — Predict the person's most-chosen item every time"

    customer_purchase_history: {customer_id: {asin: count}}
    """
    top1 = top5 = 0
    rr_list = []
    n = len(choice_events)

    for ev in choice_events:
        cid = ev["customer_id"]
        chosen_asin = ev["chosen_asin"]
        choice_asins = ev["choice_asins"]
        chosen_idx = ev["chosen_idx"]

        history = customer_purchase_history.get(cid, {})
        # Rank each candidate by frequency
        ranked = sorted(choice_asins, key=lambda a: history.get(a, 0), reverse=True)
        rank = ranked.index(chosen_asin) if chosen_asin in ranked else len(ranked)

        top1 += rank == 0
        top5 += rank < 5
        rr_list.append(1.0 / (rank + 1))

    return {
        "top1": top1 / n,
        "top5": top5 / n,
        "mrr": float(np.mean(rr_list)),
        "val_nll": float("nan"),
        "n_events": n,
    }


def markov_chain_baseline(
    choice_events: List[dict],
    transition_matrix: pd.DataFrame,
    category_popularity: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """
    Baseline: transition probabilities over categories.
    Paper: "Markov chain — Transition probabilities over categories"

    Predicts: for each event, rank candidates by
      P(candidate_category | previous_category) × popularity(candidate)

    transition_matrix: category → category transition probabilities (from category_switching_matrix)
    category_popularity: {category: {asin: count}}
    """
    top1 = top5 = 0
    rr_list = []
    n = len(choice_events)

    for ev in choice_events:
        chosen_asin = ev["chosen_asin"]
        choice_asins = ev["choice_asins"]
        prev_cat = ev.get("prev_category", ev["category"])

        # Score each candidate by transition prob × popularity
        def score_asin(asin: str) -> float:
            cat = ev["category"]
            trans_prob = float(
                transition_matrix.loc[prev_cat, cat]
                if prev_cat in transition_matrix.index and cat in transition_matrix.columns
                else 1e-3
            )
            pop = category_popularity.get(cat, {}).get(asin, 0)
            return trans_prob * (1 + pop)

        ranked = sorted(choice_asins, key=score_asin, reverse=True)
        rank = ranked.index(chosen_asin) if chosen_asin in ranked else len(ranked)

        top1 += rank == 0
        top5 += rank < 5
        rr_list.append(1.0 / (rank + 1))

    return {
        "top1": top1 / n,
        "top5": top5 / n,
        "mrr": float(np.mean(rr_list)),
        "val_nll": float("nan"),
        "n_events": n,
    }


def no_hierarchy_weight_fn(
    theta_g: np.ndarray,
) -> Callable[[str, str], np.ndarray]:
    """
    Ablation: single global weight vector — no θ_c or Δ_i.
    Paper: "No hierarchy — Single-level weights ablation"

    Wraps theta_g as a weight_fn callable.
    """
    def weight_fn(customer_id: str, category: str) -> np.ndarray:
        return theta_g
    return weight_fn


def print_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    nll_str = f"{metrics['val_nll']:.4f}" if not np.isnan(metrics.get("val_nll", float("nan"))) else "n/a"
    print(
        f"{prefix}top-1: {metrics['top1']:.1%}  "
        f"top-5: {metrics['top5']:.1%}  "
        f"MRR: {metrics['mrr']:.4f}  "
        f"NLL: {nll_str}  "
        f"(n={metrics.get('n_events', '?')})"
    )
