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
from typing import Callable, Optional, Tuple

import requests
from dotenv import load_dotenv

from .dsl import DSLStructure, ALL_TERMS, LAYER2_AMAZON, LAYER3_COMBINATORS

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


def generate_diagnostics(
    structure: DSLStructure,
    posterior_score: float,
    metrics: dict,
    residuals: dict,
) -> str:
    """Format a structured diagnostic report for the LLM proposal step."""
    available_add = [t for t in ALL_TERMS if t not in structure.terms]

    lines = [
        "Domain: amazon",
        f"Current structure: {structure}",
        f"Validation top-1: {metrics.get('top1', 0):.1%} | "
        f"top-5: {metrics.get('top5', 0):.1%} | "
        f"MRR: {metrics.get('mrr', 0):.4f}",
        f"Posterior score: {posterior_score:.2f}",
        f"Validation NLL: {metrics.get('val_nll', 0):.4f}",
        "",
        "Error analysis:",
    ]
    for k, v in residuals.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "Available DSL operations (amazon domain):",
        f"  ADD terms:       {available_add}",
        f"  ADD combinators: {LAYER3_COMBINATORS}",
        f"  REMOVE:          {structure.terms}",
        "",
        "Constraint: propose at most 2 structural changes.",
        "For each change, state the operation (ADD/REMOVE), the term name,",
        "and which specific failure mode it addresses.",
        "",
        'Respond in JSON only:\n'
        '{\n'
        '  "reasoning": "...",\n'
        '  "changes": [\n'
        '    {"op": "ADD|REMOVE", "term": "<term_name>", "reason": "..."},\n'
        '    ...\n'
        '  ]\n'
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

    if iteration < 3:
        anneal_note = "\nAnnealing constraint: propose only a single ADD or REMOVE operation."
    elif iteration < 6:
        anneal_note = "\nAnnealing: interactions and split_by are now allowed."
    else:
        anneal_note = "\nAll DSL operations including combinators are allowed."

    system = (
        "You are an expert in discrete choice modeling and consumer behavior. "
        "Your task is to propose structural modifications to a utility model that "
        "improve predictive accuracy while maintaining interpretability. "
        "Respond with valid JSON only — no markdown fences, no prose outside JSON."
    )
    user = report + anneal_note

    print(f"  [LLM] provider={provider}  model={model}")

    if provider == "ollama":
        content = _call_ollama(system, user, model, max_tokens=1024)
    else:
        content = _call_anthropic(system, user, model, max_tokens=1024)

    return _parse_json_response(content)


def apply_proposal(structure: DSLStructure, proposal: dict) -> Optional[DSLStructure]:
    """
    Apply LLM-proposed changes to a structure.
    Rejects any change that references a term outside the valid DSL grammar.
    Returns None if the proposal is invalid.
    """
    new_structure = structure

    for change in proposal.get("changes", []):
        op = change.get("op", "").upper()
        term = change.get("term", "")

        if op == "ADD":
            if term not in ALL_TERMS + LAYER3_COMBINATORS:
                print(f"  [DSL] Rejected unknown term: '{term}'")
                return None
            new_structure = new_structure.add_term(term)

        elif op == "REMOVE":
            if term not in new_structure.terms:
                print(f"  [DSL] Cannot remove absent term: '{term}'")
                return None
            new_structure = new_structure.remove_term(term)

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
) -> Tuple[DSLStructure, float, bool]:
    """
    One full outer-loop iteration.

    Args:
        current_structure: current DSLStructure S
        current_score:     Score(S, θ) from inner loop
        metrics:           dict with top1, top5, mrr, val_nll
        residuals:         dict of human-readable failure descriptions
        fit_fn:            callable(structure) -> (weights, score)
        iteration:         outer loop iteration count (for annealing)

    Returns:
        (accepted_structure, accepted_score, was_accepted)
    """
    report = generate_diagnostics(current_structure, current_score, metrics, residuals)

    print(f"\n[Outer Loop iter={iteration}] Querying LLM for structural proposal...")
    try:
        proposal = prompt_llm(report, iteration=iteration)
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return current_structure, current_score, False

    reasoning = proposal.get("reasoning", "")[:300]
    print(f"  [LLM] Reasoning: {reasoning}")

    new_structure = apply_proposal(current_structure, proposal)
    if new_structure is None:
        print("  [DSL] Proposal failed grammar validation — keeping current structure.")
        return current_structure, current_score, False

    if new_structure.terms == current_structure.terms:
        print("  [DSL] No structural change proposed.")
        return current_structure, current_score, False

    print(f"  [Proposal] {current_structure}  →  {new_structure}")
    print("  [Inner Loop] Fitting weights for proposed structure...")

    _, new_score = fit_fn(new_structure)

    if new_score > current_score:
        print(f"  [Accept] Score: {current_score:.2f} → {new_score:.2f}  (+{new_score - current_score:.2f})")
        return new_structure, new_score, True
    else:
        print(f"  [Reject] Proposed score {new_score:.2f} ≤ current {current_score:.2f}")
        return current_structure, current_score, False


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
            available = [t for t in ALL_TERMS if t not in structure.terms]
            if not available:
                op = "remove"
            else:
                term = random.choice(available)
                candidate = structure.add_term(term)

        if op == "remove" and len(structure.terms) > 1:
            term = random.choice(structure.terms)
            candidate = structure.remove_term(term)
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
) -> list[dict]:
    """
    Full outer loop: run n_iterations of Structure Descent.

    Returns history list of dicts:
      {iteration, structure, score, accepted}
    """
    structure = initial_structure
    weights, score = fit_fn(structure)
    history = [{"iteration": 0, "structure": str(structure), "score": score, "accepted": True}]

    print(f"Initial: {structure}  |  Score: {score:.2f}")

    for i in range(1, n_iterations + 1):
        metrics = get_metrics_fn(weights)
        residuals = get_residuals_fn(weights)

        new_structure, new_score, accepted = structure_descent_step(
            structure, score, metrics, residuals, fit_fn, iteration=i
        )

        if accepted:
            structure, score = new_structure, new_score
            weights, score = fit_fn(structure)  # re-fit accepted structure

        history.append({
            "iteration": i,
            "structure": str(structure),
            "score": score,
            "accepted": accepted,
        })

    return history
