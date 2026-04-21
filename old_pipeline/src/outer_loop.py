"""
Outer loop: LLM-guided structure search (Structure Descent).

Algorithm per iteration:
  1. Generate diagnostics report from current fitted model
  2. Prompt the LLM with structure + diagnostics + available DSL operations
  3. Parse LLM JSON response into a new structure S'
  4. Validate S' against the DSL grammar (reject invalid proposals at parse time)
  5. Run inner loop on S', compute posterior Score(S', θ')
  6. Accept S' if Score(S', θ') > Score(S, θ)   [Metropolis-Hastings criterion]

Annealing schedule:
  - Iterations 0-2:  single ADD/REMOVE only
  - Iterations 3-5:  interactions and split conditions allowed
  - Iterations 6+:   all combinators allowed
"""

import os
import json
import re
from collections import Counter
from typing import Callable, Optional, Tuple

import requests
from dotenv import load_dotenv

from .dsl import (
    DSLStructure, DSLTerm, ALL_TERMS, LAYER2_AMAZON, LAYER3_COMBINATORS,
    BINARY_COMBINATORS, UNARY_COMBINATORS,
)
from .inner_loop import HierarchicalWeights
from .accept_strategies import AcceptStrategy, GreedyAccept


# ── Importance thresholds for fitted global weights (item 1) ─────────────────
_IMPORTANCE_HIGH = 1.0
_IMPORTANCE_MEDIUM = 0.3
_REMOVAL_CANDIDATE_THRESHOLD = 0.1
_CATEGORY_WANT_THRESHOLD = 0.5

load_dotenv()


def _call_anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    """Call Anthropic Claude API. Requires ANTHROPIC_API_KEY in env."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text.strip()


def _call_ollama(system: str, user: str, model: str, max_tokens: int) -> str:
    """
    Call a local Ollama model via its OpenAI-compatible REST endpoint.
    Requires Ollama to be running: https://ollama.com
    Set OLLAMA_BASE_URL (default http://localhost:11434) and OLLAMA_MODEL in .env.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_json_response(content: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
    if match:
        content = match.group(1)
    return json.loads(content)


_COMBINATOR_SET = BINARY_COMBINATORS | UNARY_COMBINATORS


def _extract_combinators_and_bases(changes: list[dict]) -> Tuple[list[str], list[str]]:
    """Given a changes list, return (combinators, base_terms) mentioned."""
    combinators: list[str] = []
    bases: list[str] = []
    for c in changes or []:
        raw = c.get("term", "") or ""
        try:
            parsed = DSLTerm.parse(raw)
        except Exception:
            continue
        if parsed.name in _COMBINATOR_SET:
            combinators.append(parsed.name)
            for a in parsed.args:
                bases.append(a)
        else:
            bases.append(parsed.name)
    return combinators, bases


def _detect_rejection_pattern(rejected: list[dict]) -> Optional[str]:
    """Item 6 — outcome-aware rejection pattern detection.

    Returns a short diagnostic string if a pattern is found across ≥3 rejections
    where ≥50% share a combinator or base term. Otherwise None.
    """
    if len(rejected) < 3:
        return None

    comb_counter: Counter = Counter()
    base_counter: Counter = Counter()
    for p in rejected:
        combs, bases = _extract_combinators_and_bases(p.get("changes", []))
        for c in set(combs):
            comb_counter[c] += 1
        for b in set(bases):
            base_counter[b] += 1

    n = len(rejected)
    best_kind = None
    best_name = None
    best_count = 0

    for name, cnt in comb_counter.most_common():
        if cnt / n >= 0.5 and cnt >= best_count:
            best_kind, best_name, best_count = "combinator", name, cnt

    for name, cnt in base_counter.most_common():
        if cnt / n >= 0.5 and cnt > best_count:
            best_kind, best_name, best_count = "base", name, cnt

    if best_kind is None:
        return None

    if best_kind == "combinator":
        return (
            f"Pattern detected: {best_count} of {n} rejected proposals involve "
            f"`{best_name}(...)` combinators.\n"
            f"Hypothesis: {best_name} transformations aren't fitting this data well. "
            f"Consider other combinators (ratio, log_transform, threshold) or plain "
            f"base terms instead."
        )
    else:
        return (
            f"Pattern detected: {best_count} of {n} rejected proposals involve "
            f"the base term `{best_name}`.\n"
            f"Hypothesis: `{best_name}` may not carry useful marginal signal given "
            f"the current structure. Consider a different base term, or a compound "
            f"involving `{best_name}` only through an interaction or threshold."
        )


def _format_proposal_history(proposal_log: list[dict]) -> str:
    """Format proposal history as context for the LLM.

    Summarizes past proposals so the LLM avoids repeating rejected changes
    and builds on successful patterns. Also runs rejection-pattern detection
    (item 6) and surfaces stored hypotheses for rejected attempts (item 4/8).
    """
    if not proposal_log:
        return ""

    rejected = [p for p in proposal_log if not p.get("accepted", False)]
    accepted = [p for p in proposal_log if p.get("accepted", False)]

    lines = ["", "Proposal history (do NOT repeat rejected proposals):"]

    if rejected:
        lines.append("  REJECTED proposals:")
        for p in rejected:
            changes_str = ", ".join(
                f"{c['op']} {c['term']}" for c in p.get("changes", [])
            )
            delta = p.get("score_delta")
            delta_str = f" (delta: {delta:+.2f})" if delta is not None else ""
            lines.append(f"    - iter {p['iteration']}: [{changes_str}]{delta_str}")
            hyp = p.get("hypothesis")
            if hyp:
                lines.append(f"      hypothesis was: {hyp[:150]}")
            if p.get("reasoning"):
                lines.append(f"      reason was:     {p['reasoning'][:150]}")

    if accepted:
        lines.append("  ACCEPTED proposals (successful patterns):")
        for p in accepted:
            changes_str = ", ".join(
                f"{c['op']} {c['term']}" for c in p.get("changes", [])
            )
            delta = p.get("score_delta")
            delta_str = f" (delta: {delta:+.2f})" if delta is not None else ""
            lines.append(f"    - iter {p['iteration']}: [{changes_str}]{delta_str}")

    pattern_msg = _detect_rejection_pattern(rejected)
    if pattern_msg:
        lines.append("")
        for pl in pattern_msg.split("\n"):
            lines.append(f"  {pl}")

    lines.append("")
    lines.append("Learn from this history: avoid changes that were already rejected,")
    lines.append("and consider why accepted changes worked.")

    return "\n".join(lines)


def _classify_importance(w: float) -> str:
    aw = abs(w)
    if aw > _IMPORTANCE_HIGH:
        return "HIGH"
    if aw >= _IMPORTANCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _format_structure_with_weights(
    structure: DSLStructure, weights: "HierarchicalWeights"
) -> list[str]:
    """Item 1 — render structure with fitted global weights + category deviations."""
    lines = ["Current structure (with fitted global weights theta_g):"]

    term_names = [t.display_name for t in structure.terms]
    theta_g = weights.theta_g

    # Find top contributor by |w|
    if len(theta_g) > 0:
        top_idx = int(max(range(len(theta_g)), key=lambda i: abs(theta_g[i])))
    else:
        top_idx = -1

    name_width = max((len(n) for n in term_names), default=10)
    for i, name in enumerate(term_names):
        if i >= len(theta_g):
            break
        w = float(theta_g[i])
        importance = _classify_importance(w)
        annotation = ""
        if i == top_idx:
            annotation = "  (top contributor)"
        elif importance == "LOW" and abs(w) < _REMOVAL_CANDIDATE_THRESHOLD:
            annotation = "  (near-zero - candidate for removal?)"
        lines.append(
            f"  {name:<{name_width}} : weight={w:+.2f}  "
            f"importance={importance:<6}{annotation}"
        )

    # Category-level deviations
    theta_c = weights.theta_c or {}
    if theta_c:
        import numpy as _np
        cat_mags = []
        for cat, vec in theta_c.items():
            cat_mags.append((cat, float(_np.sum(_np.abs(vec)))))
        cat_mags.sort(key=lambda x: x[1], reverse=True)
        top_cats = cat_mags[:3]

        if top_cats:
            lines.append("")
            lines.append("Category-level deviations from global (top 3 by magnitude):")
            for cat, _mag in top_cats:
                vec = theta_c[cat]
                if len(vec) == 0:
                    continue
                order = sorted(
                    range(len(vec)), key=lambda k: abs(float(vec[k])), reverse=True
                )[:2]
                parts = []
                for k in order:
                    tname = term_names[k] if k < len(term_names) else f"term{k}"
                    parts.append(f"{tname} {float(vec[k]):+.2f}")
                lines.append(f"  {cat:<15}: " + ", ".join(parts))

    return lines


def _format_score_trajectory(
    posterior_score: float,
    history: list[dict],
    iteration: int,
    n_iterations: int,
) -> list[str]:
    """Item 5 — trajectory block."""
    iter_of_total = f"(iteration {iteration} of {n_iterations})" if n_iterations else ""
    lines = [f"Posterior score: {posterior_score:.2f}    {iter_of_total}".rstrip()]
    if not history:
        return lines

    lines.append("")
    lines.append("Trajectory so far:")

    for i, h in enumerate(history):
        it = h.get("iteration", i)
        score = h.get("score", 0.0)
        if it == 0:
            tag = "baseline"
            delta_str = ""
            status = ""
        else:
            changes = h.get("changes") or h.get("changes_summary") or []
            summary = ""
            if isinstance(changes, list) and changes:
                parts = []
                for c in changes:
                    if isinstance(c, dict):
                        op = (c.get("op", "") or "").upper()
                        term = c.get("term", "")
                        sign = "+" if op == "ADD" else ("-" if op == "REMOVE" else "?")
                        parts.append(f"{sign}{term}")
                summary = ",".join(parts)
            if not summary:
                summary = "change"
            if len(summary) > 30:
                summary = summary[:27] + "..."
            tag = summary
            prev = history[i - 1].get("score", 0.0) if i > 0 else 0.0
            delta = score - prev
            delta_str = f"   ({delta:+.1f})"
            status = "   ACCEPTED" if h.get("accepted", False) else "   REJECTED"
        cur_marker = "  <- current" if it == iteration else ""
        lines.append(
            f"  iter {it} ({tag}): {score:.2f}{delta_str}{status}{cur_marker}"
        )

    if iteration and (not history or history[-1].get("iteration", -1) < iteration):
        lines.append(f"  iter {iteration} so far: (pending)")

    # Best so far
    best = max(h.get("score", float("-inf")) for h in history)
    if posterior_score >= best:
        best_note = "you are at the best"
    else:
        gap = best - posterior_score
        best_note = f"you are {gap:.2f} below best"
    lines.append(f"Best score so far: {best:.2f} ({best_note})")

    # Recent deltas and trajectory classification
    deltas = []
    for i in range(1, len(history)):
        d = history[i].get("score", 0.0) - history[i - 1].get("score", 0.0)
        if d != 0:
            deltas.append(d)
    recent = deltas[-3:]
    if recent:
        recent_str = ", ".join(f"{d:+.1f}" for d in recent)
        net = sum(recent)
        all_pos = all(d > 0 for d in recent)
        if net < 0:
            classification = "regressing"
        elif all_pos and len(recent) >= 2:
            mags = [abs(d) for d in recent]
            if all(mags[i] >= mags[i + 1] for i in range(len(mags) - 1)):
                classification = "diminishing returns"
            elif all(mags[i] <= mags[i + 1] for i in range(len(mags) - 1)):
                classification = "still accelerating"
            else:
                classification = "oscillating"
        elif all_pos:
            classification = "still improving"
        else:
            classification = "oscillating"
        lines.append(f"Recent deltas:     {recent_str}   (search is {classification})")
    return lines


def _format_structured_residuals(residuals: dict) -> list[str]:
    """Item 2 — render structured residuals as a table."""
    slices = residuals.get("slices", [])
    patterns = residuals.get("overall_patterns", [])
    lines = ["Error analysis (worst-performing slices, top 5):"]
    header = (
        f"  {'slice':<20}{'top-1':>8}{'#events':>12}"
        f"{'first-buy-err':>18}{'popular-err':>15}"
    )
    lines.append(header)
    for s in slices[:5]:
        name = str(s.get("name", ""))[:20]
        top1 = s.get("top1", 0.0)
        n_events = int(s.get("n_events", 0))
        fb = s.get("pct_first_buy_errors", 0.0)
        pop = s.get("pct_popular_errors", 0.0)
        lines.append(
            f"  {name:<20}{top1:>7.1%}{n_events:>12,}"
            f"{fb:>17.0%}{pop:>14.0%}"
        )
    if patterns:
        lines.append("")
        lines.append("Key patterns:")
        for p in patterns:
            lines.append(f"  - {p}")
    return lines


def generate_diagnostics(
    structure: DSLStructure,
    posterior_score: float,
    metrics: dict,
    residuals: dict,
    proposal_log: Optional[list[dict]] = None,
    weights: Optional["HierarchicalWeights"] = None,
    history: Optional[list[dict]] = None,
    iteration: int = 0,
    n_iterations: int = 0,
) -> str:
    """Format a structured diagnostic report for the LLM proposal step."""
    current_names = {t.name for t in structure.terms}
    available_add = [t for t in ALL_TERMS if t not in current_names]

    lines: list[str] = ["Domain: amazon"]

    # Item 1 — structure with fitted weights (or fallback)
    if weights is not None:
        lines.extend(_format_structure_with_weights(structure, weights))
    else:
        lines.append(f"Current structure: {structure}")

    lines.append("")
    lines.append(
        f"Validation top-1: {metrics.get('top1', 0):.1%} | "
        f"top-5: {metrics.get('top5', 0):.1%} | "
        f"MRR: {metrics.get('mrr', 0):.4f}"
    )

    # Item 5 — trajectory (or single line fallback)
    if history:
        lines.extend(
            _format_score_trajectory(
                posterior_score, history, iteration, n_iterations
            )
        )
    else:
        lines.append(f"Posterior score: {posterior_score:.2f}")

    lines.append(f"Validation NLL: {metrics.get('val_nll', 0):.4f}")
    lines.append("")

    # Item 2 — structured residuals vs. old dict-of-strings fallback
    if isinstance(residuals, dict) and "slices" in residuals:
        lines.extend(_format_structured_residuals(residuals))
    else:
        lines.append("Error analysis:")
        for k, v in (residuals or {}).items():
            lines.append(f"- {k}: {v}")

    # Inject proposal history so LLM doesn't repeat failed proposals
    history_section = _format_proposal_history(proposal_log or [])
    if history_section:
        lines.append(history_section)

    # Current terms for display
    current_term_names = [t.display_name for t in structure.terms]

    lines += [
        "",
        "Available DSL operations (amazon domain):",
        f"  ADD base terms (pick only WIRED ones from the system message):  {available_add}",
        f"  REMOVE:                                                         {current_term_names}",
        "  COMPOUND: any of the 7 Layer 3 combinators defined in the system message",
        "            (interaction, split_by, ratio, difference, threshold, power, log_transform),",
        "            referencing any base term in the args — even base terms not currently",
        "            present as top-level terms in the structure.",
        "",
        "See the system message for full formulas, ranges, and worked examples of every",
        "base term and every combinator.",
        "",
        "How to propose terms:",
        "  Simple term:    {\"op\": \"ADD\", \"term\": \"popularity\", \"reason\": \"...\"}",
        "  Compound term:  {\"op\": \"ADD\", \"term\": \"interaction(routine, recency)\", \"reason\": \"...\"}",
        "  Or with args:   {\"op\": \"ADD\", \"term\": \"interaction\", \"args\": [\"routine\", \"recency\"], \"reason\": \"...\"}",
        "  Threshold:      {\"op\": \"ADD\", \"term\": \"threshold(routine, cutoff=3)\", \"reason\": \"...\"}",
        "  Remove:         {\"op\": \"REMOVE\", \"term\": \"time_match\", \"reason\": \"...\"}",
        "",
        "Before proposing changes, follow this reasoning template:",
        "",
        "1. HYPOTHESIS: What is the single most important failure mode in the current model?",
        "   Reference the error analysis above; be specific about WHICH slice and WHAT KIND of error.",
        "",
        "2. MECHANISM: Why would your proposed change fix the hypothesis?",
        "   What term or interaction would capture the missing signal? Why that one and not others?",
        "",
        "3. CANDIDATES: Generate 3 candidate changes, each addressing the hypothesis from a",
        "   different angle. Rank them by expected score improvement.",
        "",
        "4. SELECT: Choose the top-ranked candidate.",
        "",
        "Constraint: propose at most 2 structural changes per candidate.",
        "",
        'Respond in JSON only:\n'
        '{\n'
        '  "hypothesis":  "<1-2 sentences on the main failure mode>",\n'
        '  "mechanism":   "<1-2 sentences on why your fix addresses it>",\n'
        '  "candidates": [\n'
        '    {"changes": [{"op": "ADD", "term": "...", "reason": "..."}], "expected_impact": "HIGH",   "rationale": "..."},\n'
        '    {"changes": [{"op": "ADD", "term": "...", "reason": "..."}], "expected_impact": "MEDIUM", "rationale": "..."},\n'
        '    {"changes": [{"op": "ADD", "term": "...", "reason": "..."}], "expected_impact": "LOW",    "rationale": "..."}\n'
        '  ],\n'
        '  "selected":    0,\n'
        '  "reasoning":   "<why the selected candidate is best>"\n'
        '}',
    ]
    return "\n".join(lines)


def prompt_llm(
    report: str,
    iteration: int = 0,
    model: Optional[str] = None,
) -> dict:
    """
    Send diagnostic report to the LLM; return parsed JSON proposal.

    Provider is selected via LLM_PROVIDER in .env:
      LLM_PROVIDER=anthropic  →  uses ANTHROPIC_API_KEY + ANTHROPIC_MODEL
      LLM_PROVIDER=ollama     →  uses OLLAMA_BASE_URL + OLLAMA_MODEL

    Raises on API error or JSON parse failure.
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if model is None:
        if provider == "ollama":
            model = os.environ.get("OLLAMA_MODEL", "llama3")
        else:
            model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")

    if iteration < 2:
        anneal_note = "\nAnnealing constraint: propose up to 2 ADD or REMOVE operations on Layer 1/2 terms."
    elif iteration < 4:
        anneal_note = "\nAnnealing: up to 2 changes. Interactions and split_by combinators are now allowed in addition to Layer 1/2 terms."
    else:
        anneal_note = "\nAll DSL operations including all combinators (interaction, split_by, threshold, log_transform, ratio, power, difference) are allowed. Propose up to 2 changes."

    system = (
        "You are an expert in discrete choice modeling and consumer behavior.\n"
        "\n"
        "MODEL\n"
        "-----\n"
        "The utility model is a hierarchical multinomial logit (MNL) over choice sets.\n"
        "For each event i with alternatives k in {0, ..., K-1}, the utility of alternative k is:\n"
        "\n"
        "    U_i(k) = sum_j  theta_j(customer_i, category_i) * phi_j(k)\n"
        "\n"
        "where phi_j(k) is the j-th DSL term evaluated at alternative k. The coefficient\n"
        "theta_j decomposes hierarchically:\n"
        "\n"
        "    theta_j(c, cat) = theta_g_j + theta_c_j[cat] + delta_i_j[c]\n"
        "                      \\_ global _/  \\_ category _/  \\_ customer _/\n"
        "\n"
        "Choice probability is P(k | i) = softmax(U_i)[k]. The inner loop maximizes the\n"
        "log posterior via L-BFGS-B:\n"
        "\n"
        "    Score(S, theta) = sum_i  log softmax(U_i)[chosen_i]            # log-likelihood\n"
        "                    - ||theta_g||^2 / (2 * sigma^2)                # global prior   (sigma=10)\n"
        "                    - sum_cat ||theta_c[cat]||^2 / (2 * tau^2)     # category prior (tau=1)\n"
        "                    - sum_c   ||delta_i[c]||^2 / (2 * nu^2)        # customer prior (nu=0.5)\n"
        "                    - L(S) * log(2)                                # structure prior\n"
        "\n"
        "L(S) is DSL complexity: 1 per simple term, (1 + n_args) per compound term.\n"
        "Higher Score is better. Your job: propose structural changes to S that increase Score.\n"
        "\n"
        "DSL BASE TERMS  (Layer 1 + Layer 2)\n"
        "------------------------------------\n"
        "Each base term is a function phi(k) evaluated at alternative k for the current\n"
        "event i. 'events j < i' means 'events of this customer strictly before event i'\n"
        "(no lookahead — features are causal at event time).\n"
        "\n"
        "Layer 1 — Universal behavioral primitives:\n"
        "\n"
        "  routine           phi(k) = |{ j < i : j.customer == i.customer AND j.asin == k.asin }|\n"
        "                    count of prior purchases of this (customer, item) pair\n"
        "                    range [0, inf);  0 for a never-seen alternative\n"
        "\n"
        "  recency           phi(k) = 1 / (1 + days_since_last_purchase_of_k)\n"
        "                    transformed recency;  range (0, 1];  1 = bought today;  ~0.001 = never\n"
        "\n"
        "  novelty           phi(k) = 1 if routine(k) == 0 else 0\n"
        "                    first-purchase indicator;  range {0, 1}\n"
        "\n"
        "  popularity        phi(k) = log(1 + train_count(k.asin))\n"
        "                    train-only log frequency;  does NOT leak val/test\n"
        "\n"
        "  affinity          phi(k) = log(1 + |{ j < i : j.customer == i.customer AND j.category == k.category }|)\n"
        "                    customer's prior purchase count in this category, log-compressed\n"
        "\n"
        "  time_match        [NOT WIRED — always 0 for every alternative. Proposing adds a zero column.]\n"
        "\n"
        "Layer 2 — Amazon-specific features:\n"
        "\n"
        "  price_sensitivity phi(k) = -(price(k) / cat_avg_price - 1)\n"
        "                    positive when item is cheaper than category mean;  range (-inf, inf)\n"
        "\n"
        "  price_rank        phi(k) = 1 - price(k) / cat_avg_price\n"
        "                    within-category price ranking;  higher = relatively cheaper;  range (-inf, 1]\n"
        "\n"
        "  rating_signal     [NOT WIRED — always 0]\n"
        "  brand_affinity    [NOT WIRED post-leakage-fix — always 0 on both positives and negatives]\n"
        "  delivery_speed    [NOT WIRED — always 0]\n"
        "  co_purchase       [NOT WIRED — always 0]\n"
        "\n"
        "IMPORTANT: only propose terms WITHOUT a [NOT WIRED] tag. Unwired terms add a\n"
        "zero column to the utility — they waste an iteration and block no progress.\n"
        "Price features on the NEGATIVES are currently zero (no leakage-safe per-asin price\n"
        "available), so price_sensitivity and price_rank only carry signal on the positive row.\n"
        "\n"
        "LAYER 3 COMBINATORS\n"
        "-------------------\n"
        "Compose base terms into a single compound feature. Each compound is computed\n"
        "once at feature-extraction time from the raw 12-column base-feature matrix, then\n"
        "fit with a single theta_g_j coefficient (plus category and customer deviations).\n"
        "You MAY reference any base term in compound args even if it is not currently a\n"
        "top-level term in the structure — the raw base column is always available.\n"
        "\n"
        "Binary combinators (two base-term args):\n"
        "\n"
        "  interaction(a, b)        phi(k) = a(k) * b(k)\n"
        "                           element-wise product; use when the effect of a depends on b\n"
        "                           ex: interaction(routine, recency) — recent repeats count more than old ones\n"
        "\n"
        "  split_by(a, condition)   phi(k) = a(k) * condition(k)\n"
        "                           same math as interaction; semantically, `condition` is a {0,1} gate\n"
        "                           ex: split_by(popularity, novelty) — popularity only activates on new items\n"
        "\n"
        "  ratio(a, b)              phi(k) = a(k) / (b(k) + 1e-8)\n"
        "                           normalized ratio with epsilon for stability\n"
        "                           ex: ratio(routine, popularity) — personalized frequency relative to global\n"
        "\n"
        "  difference(a, b)         phi(k) = a(k) - b(k)\n"
        "                           explicit contrast between two features\n"
        "                           ex: difference(routine, affinity) — item-level vs category-level pattern\n"
        "\n"
        "Unary combinators (one base-term arg + scalar hyperparameter):\n"
        "\n"
        "  threshold(a, cutoff=N)   phi(k) = 1 if a(k) > N else 0\n"
        "                           hard indicator; N is the tunable cutoff\n"
        "                           ex: threshold(routine, cutoff=3) — 'heavy repeat' binary flag\n"
        "\n"
        "  power(a, exponent=N)     phi(k) = |a(k)|^N * sign(a(k))\n"
        "                           signed power;  N>1 amplifies large values;  0<N<1 compresses them\n"
        "                           ex: power(popularity, exponent=0.5) — square-root popularity\n"
        "\n"
        "  log_transform(a)         phi(k) = sign(a(k)) * log(1 + |a(k)|)\n"
        "                           signed log;  preserves sign, compresses magnitude\n"
        "                           ex: log_transform(price_sensitivity)\n"
        "\n"
        "NOTES ON COMPOUND COMPOSITION\n"
        "-----------------------------\n"
        "- Arguments to combinators must be NAMES of base terms (e.g. 'routine'), not\n"
        "  nested compounds. Nested compound-of-compound is not currently supported.\n"
        "- threshold and power take their scalar hyperparameter as a kwarg\n"
        "  (cutoff=N / exponent=N), written inline: `threshold(routine, cutoff=3)`.\n"
        "- Each compound adds complexity L += 1 + n_args, so a compound costs more than\n"
        "  a simple term under the structure prior — propose only when the mechanism is\n"
        "  clearly justified by the error analysis.\n"
        "\n"
        "RESPONSE FORMAT\n"
        "---------------\n"
        "Respond with valid JSON only — no markdown fences, no prose outside the JSON.\n"
        "The diagnostic report in the user message specifies the exact schema for this\n"
        "iteration (candidates, selected, reasoning)."
    )
    user = report + anneal_note

    print(f"  [LLM] provider={provider}  model={model}")

    if provider == "ollama":
        content = _call_ollama(system, user, model, max_tokens=1024)
    else:
        content = _call_anthropic(system, user, model, max_tokens=1024)

    return _parse_json_response(content)


def _parse_proposed_term(raw_term: str, args: Optional[list] = None) -> Optional[DSLTerm]:
    """
    Parse a proposed term from the LLM into a DSLTerm.

    Handles:
      'popularity'                          -> DSLTerm('popularity')
      'interaction(routine, recency)'       -> DSLTerm('interaction', args=['routine','recency'])
      'interaction' + args=['routine','recency'] -> DSLTerm('interaction', args=['routine','recency'])
      'log_transform(popularity)'           -> DSLTerm('log_transform', args=['popularity'])
      'threshold(routine, cutoff=3)'        -> DSLTerm('threshold', args=['routine'], kwargs={'cutoff':3.0})

    Returns None if the term is not in the valid DSL.
    """
    valid = ALL_TERMS + LAYER3_COMBINATORS

    # Try parsing as compound expression first: 'interaction(routine, recency)'
    parsed = DSLTerm.parse(raw_term)

    if parsed.name not in valid:
        return None

    # If it's a combinator name without args, check if args were passed separately
    if parsed.name in BINARY_COMBINATORS | UNARY_COMBINATORS and not parsed.args:
        if args:
            # Validate that args are known base terms
            for a in args:
                if a not in ALL_TERMS:
                    print(f"  [DSL] Unknown arg '{a}' in compound term '{raw_term}'")
                    return None
            parsed = DSLTerm(name=parsed.name, args=args, kwargs=parsed.kwargs)
        # If still no args for a combinator, it's incomplete but we allow it
        # (the feature builder will use zeros for missing args)

    # Validate args reference valid base terms
    if parsed.args:
        for a in parsed.args:
            if a not in ALL_TERMS:
                print(f"  [DSL] Unknown arg '{a}' in compound term '{parsed.display_name}'")
                return None

    print(f"  [DSL] Parsed term: {parsed.display_name}")
    return parsed


def apply_proposal(structure: DSLStructure, proposal: dict) -> Optional[DSLStructure]:
    """
    Apply LLM-proposed changes to a structure.
    Handles simple terms, compound expressions, and separate args field.
    Returns None if the proposal is invalid.

    Supports two JSON schemas for backwards compatibility:
      - NEW (item 8): {"candidates": [...], "selected": idx, ...}
      - OLD:          {"changes": [...], "reasoning": "..."}
    """
    # Item 8 — NEW schema: pick selected candidate and unwrap its changes
    if "candidates" in proposal and "selected" in proposal:
        candidates = proposal.get("candidates", []) or []
        try:
            idx = int(proposal.get("selected", 0))
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(candidates):
            selected_changes = candidates[idx].get("changes", [])
            proposal = {"changes": selected_changes}
        else:
            return None

    new_structure = structure

    for change in proposal.get("changes", []):
        op = change.get("op", "").upper()
        raw_term = change.get("term", "")
        args = change.get("args")  # Optional: ['routine', 'recency']

        if op == "ADD":
            term = _parse_proposed_term(raw_term, args=args)
            if term is None:
                print(f"  [DSL] Rejected unknown term: '{raw_term}'")
                return None
            new_structure = new_structure.add_term(term)

        elif op == "REMOVE":
            before_len = len(new_structure)
            new_structure = new_structure.remove_term(raw_term)
            if len(new_structure) == before_len:
                print(f"  [DSL] Cannot remove absent term: '{raw_term}'")
                return None

        else:
            print(f"  [DSL] Unknown operation: '{op}'")
            return None

    if len(new_structure.terms) == 0:
        print("  [DSL] Proposal would produce empty structure.")
        return None

    return new_structure


def structure_descent_step(
    current_structure: DSLStructure,
    current_score: float,
    metrics: dict,
    residuals: dict,
    fit_fn: Callable[[DSLStructure], Tuple[object, float]],
    iteration: int = 0,
    proposal_log: Optional[list[dict]] = None,
    accept_strategy: Optional[AcceptStrategy] = None,
    n_iterations_total: int = 0,
    weights: Optional[HierarchicalWeights] = None,
    history: Optional[list[dict]] = None,
) -> Tuple[DSLStructure, float, bool, dict]:
    """
    One full outer-loop iteration.

    Args:
        current_structure: current DSLStructure S
        current_score:     Score(S, θ) from inner loop
        metrics:           dict with top1, top5, mrr, val_nll
        residuals:         dict of human-readable failure descriptions
        fit_fn:            callable(structure) -> (weights, score)
        iteration:         outer loop iteration count (for annealing)
        proposal_log:      list of past proposal dicts for history-aware prompting
        accept_strategy:   pluggable accept criterion; defaults to GreedyAccept
        n_iterations_total: total planned outer-loop iterations (needed by
                            annealing / threshold schedules)

    Returns:
        (accepted_structure, accepted_score, was_accepted, proposal_detail)
    """
    strategy = accept_strategy if accept_strategy is not None else GreedyAccept()
    report = generate_diagnostics(
        current_structure, current_score, metrics, residuals,
        proposal_log=proposal_log,
        weights=weights,
        history=history,
        iteration=iteration,
        n_iterations=n_iterations_total,
    )

    # Show what history the LLM is seeing
    if proposal_log:
        n_rej = sum(1 for p in proposal_log if not p.get("accepted"))
        n_acc = sum(1 for p in proposal_log if p.get("accepted"))
        print(f"  [History] {len(proposal_log)} prior proposals ({n_acc} accepted, {n_rej} rejected)")
        for p in proposal_log:
            changes_str = ", ".join(f"{c['op']} {c['term']}" for c in p.get("changes", []))
            status = "✓" if p.get("accepted") else "✗"
            delta = p.get("score_delta")
            delta_str = f" Δ={delta:+.2f}" if delta is not None else ""
            print(f"    {status} iter {p['iteration']}: [{changes_str}]{delta_str}")

    print(f"\n[Outer Loop iter={iteration}] Querying LLM for structural proposal...")
    try:
        proposal = prompt_llm(report, iteration=iteration)
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return current_structure, current_score, False, {}

    reasoning = proposal.get("reasoning", "")[:300]
    hypothesis = proposal.get("hypothesis", "")
    mechanism = proposal.get("mechanism", "")

    # Extract the effective changes list for logging. Prefer selected
    # candidate's changes (new schema) over the top-level changes (old schema).
    if "candidates" in proposal and "selected" in proposal:
        try:
            _sel = int(proposal.get("selected", 0))
        except (TypeError, ValueError):
            _sel = -1
        _cands = proposal.get("candidates", []) or []
        if 0 <= _sel < len(_cands):
            changes = _cands[_sel].get("changes", [])
        else:
            changes = []
    else:
        changes = proposal.get("changes", [])
    print(f"  [LLM] Reasoning: {reasoning}")

    new_structure = apply_proposal(current_structure, proposal)
    if new_structure is None:
        print("  [DSL] Proposal failed grammar validation — keeping current structure.")
        detail = {
            "iteration": iteration,
            "changes": changes,
            "reasoning": proposal.get("reasoning", ""),
            "hypothesis": hypothesis,
            "mechanism": mechanism,
            "accepted": False,
            "score_delta": None,
            "accept_strategy": strategy.config(),
        }
        return current_structure, current_score, False, detail

    if new_structure.terms == current_structure.terms:
        print("  [DSL] No structural change proposed.")
        detail = {
            "iteration": iteration,
            "changes": changes,
            "reasoning": proposal.get("reasoning", ""),
            "hypothesis": hypothesis,
            "mechanism": mechanism,
            "accepted": False,
            "score_delta": 0.0,
            "accept_strategy": strategy.config(),
        }
        return current_structure, current_score, False, detail

    print(f"  [Proposal] {current_structure}  →  {new_structure}")
    print("  [Inner Loop] Fitting weights for proposed structure...")

    _, new_score = fit_fn(new_structure)
    score_delta = new_score - current_score

    detail = {
        "iteration": iteration,
        "changes": changes,
        "reasoning": proposal.get("reasoning", ""),
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "score_before": current_score,
        "score_after": new_score,
        "score_delta": score_delta,
        "structure_proposed": str(new_structure),
        "accept_strategy": strategy.config(),
    }

    accepted = strategy.decide(
        current_score, new_score, iteration, n_iterations_total
    )

    if accepted:
        strategy.on_accept(new_score, iteration)
        tag = "+" if score_delta >= 0 else ""
        print(
            f"  [Accept/{strategy.name}] Score: {current_score:.2f} → {new_score:.2f}  "
            f"({tag}{score_delta:.2f})"
        )
        detail["accepted"] = True
        return new_structure, new_score, True, detail
    else:
        strategy.on_reject(new_score, iteration)
        print(
            f"  [Reject/{strategy.name}] Proposed score {new_score:.2f} "
            f"vs current {current_score:.2f}"
        )
        detail["accepted"] = False
        return current_structure, current_score, False, detail


def random_structure_search(
    initial_structure: DSLStructure,
    fit_fn: Callable,
    n_iterations: int = 10,
    seed: int = 42,
) -> list:
    """
    Ablation: random DSL proposals instead of LLM guidance.
    Paper: "Structure search without LLM — random DSL proposals,
    ablation showing LLM is better than random."

    At each step, randomly ADD or REMOVE one term and accept if score improves.
    """
    import random
    random.seed(seed)

    structure = initial_structure
    _, score = fit_fn(structure)
    history = [{"iteration": 0, "structure": str(structure), "score": score, "accepted": True}]

    print(f"[Random Search] Initial: {structure}  |  Score: {score:.2f}")

    for i in range(1, n_iterations + 1):
        op = random.choice(["add", "remove"])

        if op == "add":
            current_names = {t.name for t in structure.terms}
            available = [t for t in ALL_TERMS if t not in current_names]
            if not available:
                op = "remove"
            else:
                term = random.choice(available)
                candidate = structure.add_term(term)

        if op == "remove" and len(structure.terms) > 1:
            term = random.choice(structure.terms)
            candidate = structure.remove_term(term.display_name)
        elif op == "remove":
            history.append({"iteration": i, "structure": str(structure), "score": score, "accepted": False})
            continue

        print(f"  [Random iter={i}] Trying: {candidate}")
        _, new_score = fit_fn(candidate)

        if new_score > score:
            print(f"  [Accept] {score:.2f} → {new_score:.2f}")
            structure, score = candidate, new_score
            accepted = True
        else:
            print(f"  [Reject] {new_score:.2f} ≤ {score:.2f}")
            accepted = False

        history.append({"iteration": i, "structure": str(structure), "score": score, "accepted": accepted})

    return history


def prompt_llm_unconstrained(
    report: str,
    model: Optional[str] = None,
) -> dict:
    """
    Ablation: TextGrad-style unconstrained LLM edits — no DSL grammar enforcement,
    no complexity prior. LLM can propose any expression string.
    Paper: "TextGrad without priors — unconstrained LLM edits,
    ablation showing priors prevent overfit."

    Uses the same LLM_PROVIDER / model settings as prompt_llm.
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if model is None:
        model = os.environ.get("OLLAMA_MODEL" if provider == "ollama" else "ANTHROPIC_MODEL",
                               "llama3" if provider == "ollama" else "claude-opus-4-6")

    system = (
        "You are a discrete choice modeler. Given a model's diagnostic report, "
        "propose any structural modification you think will improve it. "
        "You are NOT constrained to any grammar or term list. "
        'Respond in JSON: {"reasoning": "...", "new_expression": "..."}'
    )

    if provider == "ollama":
        content = _call_ollama(system, report, model, max_tokens=512)
    else:
        content = _call_anthropic(system, report, model, max_tokens=512)

    try:
        return _parse_json_response(content)
    except json.JSONDecodeError:
        return {"reasoning": content, "new_expression": content}


def run_structure_descent(
    initial_structure: DSLStructure,
    fit_fn: Callable,
    get_metrics_fn: Callable,
    get_residuals_fn: Callable,
    n_iterations: int = 10,
    checkpoint=None,
    accept_strategy: Optional[AcceptStrategy] = None,
) -> list[dict]:
    """
    Full outer loop: run n_iterations of Structure Descent.

    Supports resuming from a checkpoint. If a checkpoint is provided and
    contains a prior outer loop state, the search resumes from the last
    completed iteration.

    Args:
        checkpoint:       optional Checkpoint instance for save/resume
        accept_strategy:  pluggable accept criterion; defaults to GreedyAccept
                          so existing call sites remain unchanged.

    Returns history list of dicts:
      {iteration, structure, score, accepted}
    """
    if accept_strategy is None:
        accept_strategy = GreedyAccept()
    print(f"[Strategy] {accept_strategy.name}: {accept_strategy.config()}")

    # ── Resume from checkpoint if available ──
    start_iter = 1
    proposal_log: list[dict] = []

    if checkpoint is not None:
        state = checkpoint.get_outer_loop_state()
        if state is not None:
            start_iter = state['iteration'] + 1
            structure = DSLStructure.from_dict(state['structure'])
            score = state['score']
            history = state['history']
            proposal_log = state.get('proposal_log', [])
            print(f"Resuming from checkpoint: iter {state['iteration']}  |  {structure}  |  Score: {score:.2f}")
            print(
                f"  [Warning] accept_strategy internal state is NOT restored from "
                f"checkpoint. Strategies with history (e.g. LateAcceptanceHillClimbing) "
                f"will start with an empty buffer after resume."
            )
            if proposal_log:
                n_acc = sum(1 for p in proposal_log if p.get("accepted"))
                print(f"  [History] {len(proposal_log)} prior proposals ({n_acc} accepted, {len(proposal_log) - n_acc} rejected)")
            if start_iter > n_iterations:
                print(f"All {n_iterations} iterations already complete.")
                return history
            weights, score = fit_fn(structure)
        else:
            structure = initial_structure
            weights, score = fit_fn(structure)
            history = [{"iteration": 0, "structure": str(structure), "score": score, "accepted": True}]
            print(f"Initial: {structure}  |  Score: {score:.2f}")
            if checkpoint is not None:
                checkpoint.save_outer_loop_iteration(0, structure, score, history,
                                                     proposal_log=proposal_log)
    else:
        structure = initial_structure
        weights, score = fit_fn(structure)
        history = [{"iteration": 0, "structure": str(structure), "score": score, "accepted": True}]
        print(f"Initial: {structure}  |  Score: {score:.2f}")

    for i in range(start_iter, n_iterations + 1):
        metrics = get_metrics_fn(weights, structure)
        residuals = get_residuals_fn(weights, structure)

        new_structure, new_score, accepted, proposal_detail = structure_descent_step(
            structure, score, metrics, residuals, fit_fn,
            iteration=i, proposal_log=proposal_log,
            accept_strategy=accept_strategy,
            n_iterations_total=n_iterations,
            weights=weights,
            history=history,
        )

        # Track proposal for future iterations
        if proposal_detail:
            proposal_log.append(proposal_detail)

        if accepted:
            structure, score = new_structure, new_score
            weights, score = fit_fn(structure)  # re-fit accepted structure

        history.append({
            "iteration": i,
            "structure": str(structure),
            "score": score,
            "accepted": accepted,
        })

        # Save checkpoint only after iteration fully completes
        if checkpoint is not None:
            checkpoint.save_outer_loop_iteration(i, structure, score, history,
                                                 proposal_log=proposal_log)
            # Export JSON for visualization
            _export_search_history(checkpoint.data_dir, history, proposal_log)

    return history


def _export_search_history(data_dir, history, proposal_log):
    """Export search history as JSON for the visualization frontend.

    Only emits an ``accepted_N`` node for history entries that actually
    accepted a change. Stuck iterations (proposal rejected, structure
    unchanged) would otherwise produce a chain of identical accepted nodes.
    """
    nodes = []
    edges = []

    accepted_iters: list[int] = []
    prev_accepted_id: Optional[str] = None
    for h in history:
        if not h.get("accepted", False):
            continue
        it = h["iteration"]
        node_id = f"accepted_{it}"
        accepted_iters.append(it)
        terms = h["structure"].replace("S = ", "").split(" + ")
        nodes.append({
            "id": node_id,
            "iteration": it,
            "score": h["score"],
            "structure": h["structure"],
            "terms": terms,
            "n_terms": len(terms),
            "accepted": True,
            "type": "accepted",
        })
        if prev_accepted_id is not None:
            edges.append({"source": prev_accepted_id, "target": node_id, "type": "accepted"})
        prev_accepted_id = node_id

    def _parent_accepted(iter_: int) -> Optional[int]:
        parent = None
        for it in accepted_iters:
            if it < iter_:
                parent = it
            else:
                break
        return parent

    for p in proposal_log:
        if p.get("accepted"):
            continue
        proposed_str = p.get("structure_proposed", "")
        if not proposed_str:
            changes_str = ", ".join(f"{c['op']} {c['term']}" for c in p.get("changes", []))
            proposed_str = f"(proposed: {changes_str})"

        node_id = f"rejected_{p['iteration']}"
        proposed_terms = proposed_str.replace("S = ", "").split(" + ") if "S = " in proposed_str else []
        nodes.append({
            "id": node_id,
            "iteration": p["iteration"],
            "score": p.get("score_after", p.get("score_before", 0)),
            "structure": proposed_str,
            "terms": proposed_terms,
            "n_terms": len(proposed_terms) if proposed_terms else 0,
            "accepted": False,
            "type": "rejected",
            "reasoning": p.get("reasoning", "")[:200],
            "changes": p.get("changes", []),
            "score_delta": p.get("score_delta"),
        })
        parent_iter = _parent_accepted(p["iteration"])
        if parent_iter is not None:
            edges.append({
                "source": f"accepted_{parent_iter}",
                "target": node_id,
                "type": "rejected",
            })

    search_data = {
        "nodes": nodes,
        "edges": edges,
        "n_iterations": max(h["iteration"] for h in history) if history else 0,
        "best_score": max((n["score"] for n in nodes), default=0),
        "worst_score": min((n["score"] for n in nodes if n["score"] != 0), default=0),
    }

    import json
    from pathlib import Path
    out_path = Path(data_dir) / "search_history.json"
    with open(out_path, "w") as f:
        json.dump(search_data, f, indent=2)
    print(f"  [Viz] Exported search history to {out_path}")
