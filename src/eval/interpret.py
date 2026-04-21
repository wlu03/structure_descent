"""Interpretability reports for PO-LEU (redesign.md §12).

Four self-contained reporting functions, plus a ``run_all_reports`` aggregator,
that implement the §12 interpretability protocol:

* §12.1 :func:`head_naming_report` — per-attribute top-N outcome strings.
* §12.2 :func:`per_decision_report` — three-panel decomposition for one event.
* §12.3 :func:`dominant_attribute_report` — top-1/top-5/MRR/NLL bucketed by
  the §12.3 dominant attribute (``m* = argmax_m w_m · |u_m(e_{c*})|``).
* §12.4 :func:`counterfactual_report` — re-score after perturbing ``z_d``
  for one event, without regenerating outcomes.

Conventions
-----------
* Pure except for optional JSON file writes when ``out_dir`` is passed to
  :func:`run_all_reports`.
* All returned dicts are JSON-serializable — tensors are converted with
  ``.tolist()`` before returning.
* Deterministic; no RNG, no global state.
* Dependencies: ``torch`` + ``numpy`` + stdlib (``json``, ``pathlib``).

Defaults
--------
* ``head_names`` defaults to the §5.2 ordering
  ``["financial", "health", "convenience", "emotional", "social"]``.
* The default :func:`run_all_reports` counterfactual adds ``+1.0`` to
  ``z_d[event_idx, 0]``. This perturbation is deliberately simple and
  documented in ``NOTES.md``; callers needing a semantically meaningful
  perturbation (e.g. "+1 child") should build the corresponding
  ``perturbation_fn`` themselves and call :func:`counterfactual_report`
  directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from src.eval.strata import dominant_attribute_breakdown
from src.model.po_leu import POLEU, POLEUIntermediates


# Default attribute head naming (§5.2).
DEFAULT_HEAD_NAMES: list[str] = [
    "financial",
    "health",
    "convenience",
    "emotional",
    "social",
]


# ---------------------------------------------------------------------------
# §12.1 — attribute-head naming via top-N outcome strings per head.
# ---------------------------------------------------------------------------


def head_naming_report(
    outcomes: list[list[list[str]]],
    intermediates: POLEUIntermediates,
    *,
    top_n: int = 100,
    head_names: list[str] | None = None,
) -> dict:
    """§12.1: top-``N`` outcome strings per attribute head.

    For each attribute ``m``, collect the top-``n`` outcome strings globally
    across ``B * J * K`` by ``u_m(e_k)``; i.e., the raw attribute score
    ``intermediates.A[..., m]``.

    Parameters
    ----------
    outcomes:
        Nested list of shape ``[B][J][K]`` holding the generated outcome
        strings for each event / alternative / outcome slot.
    intermediates:
        :class:`POLEUIntermediates` from a forward pass on the batch
        ``(z_d, E)`` whose ``E`` was encoded from ``outcomes``. Must have
        ``A`` shape ``(B, J, K, M)`` consistent with ``outcomes``.
    top_n:
        Number of top outcomes to retain per head (default 100). Clamped
        to the number of available ``B*J*K`` outcomes.
    head_names:
        Optional explicit names; defaults to :data:`DEFAULT_HEAD_NAMES`
        (§5.2 naming). If the number of names exceeds ``M`` the extras
        are kept at the tail; if it is shorter, unnamed heads fall back
        to ``"m{idx}"``. Names are used only for the returned
        ``"head_names"`` list; per-head dict keys are always ``"m{idx}"``
        for a stable, spec-aligned schema.

    Returns
    -------
    dict
        ``{
            "head_names": [str, ...],
            "top_outcomes_per_head": {
                "m0": [{"score": float, "outcome": str}, ...],
                "m1": [...],
                ...
            }
        }``

    Shape contract:
        ``intermediates.A`` must have shape ``(B, J, K, M)`` and
        ``outcomes`` must have the matching ``[B][J][K]`` nesting.
    """
    A = intermediates.A
    if A.dim() != 4:
        raise ValueError(f"intermediates.A must be 4-D; got {tuple(A.shape)}")
    B, J, K, M = A.shape

    # Validate outcomes shape compatibility.
    if len(outcomes) != B:
        raise ValueError(
            f"outcomes outer length {len(outcomes)} != batch B={B}"
        )
    for b, per_event in enumerate(outcomes):
        if len(per_event) != J:
            raise ValueError(
                f"outcomes[{b}] length {len(per_event)} != J={J}"
            )
        for j, per_alt in enumerate(per_event):
            if len(per_alt) != K:
                raise ValueError(
                    f"outcomes[{b}][{j}] length {len(per_alt)} != K={K}"
                )

    # Flatten outcomes and A in the same order: (b, j, k) row-major.
    flat_strings: list[str] = []
    for b in range(B):
        for j in range(J):
            for k in range(K):
                flat_strings.append(str(outcomes[b][j][k]))

    flat_A = A.reshape(B * J * K, M).detach()

    resolved_names: list[str] = list(head_names) if head_names is not None else list(DEFAULT_HEAD_NAMES)
    # Pad with "m{idx}" if the caller supplied too few names; truncate to M.
    if len(resolved_names) < M:
        resolved_names = resolved_names + [f"m{i}" for i in range(len(resolved_names), M)]
    else:
        resolved_names = resolved_names[:M]

    n_total = flat_A.shape[0]
    effective_top_n = max(0, min(int(top_n), n_total))

    top_outcomes_per_head: dict[str, list[dict[str, Any]]] = {}
    for m in range(M):
        scores_m = flat_A[:, m]  # (B*J*K,)
        if effective_top_n == 0:
            top_outcomes_per_head[f"m{m}"] = []
            continue
        # torch.topk returns values and indices sorted descending by value.
        top = torch.topk(scores_m, k=effective_top_n, largest=True, sorted=True)
        vals = top.values.tolist()
        idxs = top.indices.tolist()
        top_outcomes_per_head[f"m{m}"] = [
            {"score": float(vals[i]), "outcome": flat_strings[idxs[i]]}
            for i in range(effective_top_n)
        ]

    return {
        "head_names": resolved_names,
        "top_outcomes_per_head": top_outcomes_per_head,
    }


# ---------------------------------------------------------------------------
# §12.2 — per-decision decomposition.
# ---------------------------------------------------------------------------


def per_decision_report(
    event_idx: int,
    outcomes: list[list[list[str]]],
    intermediates: POLEUIntermediates,
    logits: torch.Tensor,
    c_star: torch.Tensor,
) -> dict:
    """§12.2: three-panel decomposition for a single held-out event.

    Parameters
    ----------
    event_idx:
        Index of the event ``b`` within the batch.
    outcomes:
        ``[B][J][K]`` outcome strings.
    intermediates:
        :class:`POLEUIntermediates` from the forward pass.
    logits:
        ``(B, J)`` choice logits from the same forward pass.
    c_star:
        ``(B,)`` int64 chosen indices.

    Returns
    -------
    dict
        ``{
            "event_idx": int,
            "outcomes": list[list[str]],                 # (J, K) strings
            "attribute_scores": list[list[list[float]]], # (J, K, M)
            "weights": list[float],                       # (M,)
            "salience": list[list[float]],                # (J, K)
            "values": list[float],                        # (J,)
            "probabilities": list[float],                 # (J,) softmax
            "chosen": int,
        }``
    """
    b = int(event_idx)
    B, J, K, M = intermediates.A.shape

    if not (0 <= b < B):
        raise ValueError(f"event_idx {b} outside [0, B={B})")
    if logits.shape != (B, J):
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != (B, J)=({B}, {J})"
        )
    if c_star.shape[0] != B:
        raise ValueError(f"c_star length {c_star.shape[0]} != B={B}")

    # Panel 1: outcomes strings for event b.
    strings_jk: list[list[str]] = [
        [str(outcomes[b][j][k]) for k in range(K)] for j in range(J)
    ]

    # Panel 2: attribute-score matrix per alternative — (J, K, M).
    A_event = intermediates.A[b].detach().tolist()

    # Panel 3 pieces: weights (M,) and salience (J, K).
    w_event = intermediates.w[b].detach().tolist()
    S_event = intermediates.S[b].detach().tolist()

    # Panel 4: V (J,), softmax probs (J,), chosen index.
    V_event = intermediates.V[b].detach().tolist()
    probs = torch.softmax(logits[b].detach(), dim=-1).tolist()
    chosen = int(c_star[b].item())

    return {
        "event_idx": b,
        "outcomes": strings_jk,
        "attribute_scores": A_event,
        "weights": w_event,
        "salience": S_event,
        "values": V_event,
        "probabilities": probs,
        "chosen": chosen,
    }


# ---------------------------------------------------------------------------
# §12.3 — dominant-attribute evaluation (thin wrapper over strata).
# ---------------------------------------------------------------------------


def dominant_attribute_report(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    intermediates: POLEUIntermediates,
) -> dict:
    """§12.3: stratified metrics bucketed by dominant attribute.

    Thin wrapper over :func:`src.eval.strata.dominant_attribute_breakdown`.
    Repackages the result into a spec-aligned schema with per-head string
    keys (``"m{idx}"``) and a companion per-attribute count dict.

    Returns
    -------
    dict
        ``{
            "by_dominant_attribute": {"m0": {...metrics...}, ...},
            "n_by_attribute":        {"m0": int, ...}
        }``
    """
    raw = dominant_attribute_breakdown(logits, c_star, intermediates)

    by_dom: dict[str, dict[str, float]] = {}
    n_by: dict[str, int] = {}
    # Keys from strata are plain Python ints (attribute indices).
    for k, metrics in raw.items():
        # Defensive: the strata module casts numpy keys via .item(); we ensure
        # a consistent "m{idx}" string key for JSON stability.
        idx = int(k)
        head_key = f"m{idx}"
        # Split off the "n" count so the metrics dict is pure scores.
        metrics_copy = {mk: mv for mk, mv in metrics.items() if mk != "n"}
        # Cast numeric values to plain floats for JSON serialization.
        by_dom[head_key] = {mk: float(mv) for mk, mv in metrics_copy.items()}
        n_by[head_key] = int(metrics.get("n", 0))

    return {
        "by_dominant_attribute": by_dom,
        "n_by_attribute": n_by,
    }


# ---------------------------------------------------------------------------
# §12.4 — counterfactual sensitivity.
# ---------------------------------------------------------------------------


def counterfactual_report(
    model: POLEU,
    z_d: torch.Tensor,
    E: torch.Tensor,
    c_star: torch.Tensor,
    perturbation_fn: Callable[[torch.Tensor], torch.Tensor],
    event_idx: int,
    *,
    label: str = "custom",
) -> dict:
    """§12.4: re-score one event after perturbing its ``z_d`` row.

    Outcomes are **not** regenerated — ``E`` is held fixed across the two
    forward passes so the attribute scores ``u_m(e_k)`` are invariant and
    the weights / salience carry the full counterfactual signal.

    Parameters
    ----------
    model:
        Trained :class:`POLEU`.
    z_d:
        ``(B, p)`` person features.
    E:
        ``(B, J, K, d_e)`` outcome embeddings.
    c_star:
        ``(B,)`` chosen indices.
    perturbation_fn:
        Callable mapping the event's ``z_d`` row (shape ``(p,)``) to a
        perturbed row of the same shape. Called with a detached clone,
        so the caller may mutate in place safely.
    event_idx:
        Event index ``b`` in the batch.
    label:
        Free-form label describing the perturbation (e.g. ``"+1 child"``).

    Returns
    -------
    dict
        ``{
            "label": str,
            "event_idx": int,
            "delta_weights": [float, ...],        # length M
            "delta_salience": [[float, ...], ...], # shape (J, K)
            "delta_P_chosen": float,
            "P_chosen_baseline": float,
            "P_chosen_counterfactual": float
        }``
    """
    b = int(event_idx)
    B = z_d.shape[0]
    if not (0 <= b < B):
        raise ValueError(f"event_idx {b} outside [0, B={B})")

    # Baseline forward pass.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits_0, interm_0 = model(z_d, E)

            # Counterfactual: clone and perturb the target event's row only.
            z_d_cf = z_d.clone()
            perturbed_row = perturbation_fn(z_d_cf[b].clone())
            if not isinstance(perturbed_row, torch.Tensor):
                raise TypeError(
                    "perturbation_fn must return a torch.Tensor; "
                    f"got {type(perturbed_row).__name__}"
                )
            if perturbed_row.shape != z_d_cf[b].shape:
                raise ValueError(
                    f"perturbation_fn returned shape {tuple(perturbed_row.shape)}; "
                    f"expected {tuple(z_d_cf[b].shape)}"
                )
            z_d_cf[b] = perturbed_row.to(dtype=z_d_cf.dtype, device=z_d_cf.device)

            logits_cf, interm_cf = model(z_d_cf, E)
    finally:
        if was_training:
            model.train()

    # Δw — weight vector delta on the perturbed event.
    delta_w = (interm_cf.w[b] - interm_0.w[b]).detach().tolist()  # (M,)

    # ΔS — salience map delta on the perturbed event, shape (J, K).
    delta_S = (interm_cf.S[b] - interm_0.S[b]).detach().tolist()  # (J, K)

    # P(a_{c*}) baseline vs counterfactual, and the delta.
    c_idx = int(c_star[b].item())
    probs_0 = torch.softmax(logits_0[b].detach(), dim=-1)
    probs_cf = torch.softmax(logits_cf[b].detach(), dim=-1)
    P_chosen_base = float(probs_0[c_idx].item())
    P_chosen_cf = float(probs_cf[c_idx].item())

    return {
        "label": str(label),
        "event_idx": b,
        "delta_weights": delta_w,
        "delta_salience": delta_S,
        "delta_P_chosen": P_chosen_cf - P_chosen_base,
        "P_chosen_baseline": P_chosen_base,
        "P_chosen_counterfactual": P_chosen_cf,
    }


# ---------------------------------------------------------------------------
# Bundled runner — all four reports, optional JSON write-out.
# ---------------------------------------------------------------------------


def _default_perturbation_fn(row: torch.Tensor) -> torch.Tensor:
    """Default counterfactual perturbation: add +1.0 to ``z_d[:, 0]``.

    Documented in ``NOTES.md``. Deliberately trivial — a meaningful
    perturbation like ``+1 child`` depends on the :mod:`src.data.person_features`
    layout and is left to callers.
    """
    out = row.clone()
    out[0] = out[0] + 1.0
    return out


def run_all_reports(
    model: POLEU,
    z_d: torch.Tensor,
    E: torch.Tensor,
    c_star: torch.Tensor,
    outcomes: list[list[list[str]]],
    *,
    out_dir: Path | str | None = None,
    event_idx: int = 0,
    head_names: list[str] | None = None,
    counterfactual_label: str = "perturb-first-dim-+1",
) -> dict[str, Any]:
    """Run all four §12 reports on a single batch.

    Parameters
    ----------
    model, z_d, E, c_star, outcomes:
        Inputs for a forward pass; ``outcomes`` matches the ``E`` batch.
    out_dir:
        Optional directory path. When not ``None``, the four sub-reports
        are additionally written as JSON files
        (``head_naming.json``, ``per_decision.json``,
        ``dominant_attribute.json``, ``counterfactual.json``); the
        directory is created if missing. When ``None`` (default), no
        files are written.
    event_idx:
        Event used for the per-decision and counterfactual reports.
    head_names:
        Overrides :data:`DEFAULT_HEAD_NAMES` for the head-naming report.
    counterfactual_label:
        Human-readable label recorded in the counterfactual report. The
        default matches the default perturbation (``+1`` on
        ``z_d[event_idx, 0]``) described in ``NOTES.md``.

    Returns
    -------
    dict
        ``{"head_naming": ..., "per_decision": ...,
          "dominant_attribute": ..., "counterfactual": ...}``.
    """
    # Single forward pass used by reports 1-3 (counterfactual does its own
    # two-pass forward inside). This keeps the report numbers consistent
    # with one another.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, intermediates = model(z_d, E)
    finally:
        if was_training:
            model.train()

    report_head = head_naming_report(
        outcomes, intermediates, head_names=head_names
    )
    report_dec = per_decision_report(
        event_idx=event_idx,
        outcomes=outcomes,
        intermediates=intermediates,
        logits=logits,
        c_star=c_star,
    )
    report_dom = dominant_attribute_report(
        logits=logits,
        c_star=c_star,
        intermediates=intermediates,
    )
    report_cf = counterfactual_report(
        model=model,
        z_d=z_d,
        E=E,
        c_star=c_star,
        perturbation_fn=_default_perturbation_fn,
        event_idx=event_idx,
        label=counterfactual_label,
    )

    bundle: dict[str, Any] = {
        "head_naming": report_head,
        "per_decision": report_dec,
        "dominant_attribute": report_dom,
        "counterfactual": report_cf,
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename_by_key = {
            "head_naming": "head_naming.json",
            "per_decision": "per_decision.json",
            "dominant_attribute": "dominant_attribute.json",
            "counterfactual": "counterfactual.json",
        }
        for key, fname in filename_by_key.items():
            (out_path / fname).write_text(json.dumps(bundle[key], indent=2))

    return bundle
