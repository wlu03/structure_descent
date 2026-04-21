"""
Regression test for notebook callback signatures.

The outer loop calls `get_metrics_fn(weights, structure)` and
`get_residuals_fn(weights, structure)` with TWO args. Earlier versions of
`notebooks/04_outer_loop_llm.ipynb` defined them with only ONE arg, which
blew up at runtime after the outer loop had already spent an iteration
fitting the structure.

This test parses the notebook, finds the cell that defines those two
callbacks, and verifies their signatures accept the two-arg calling
convention — without executing any fitting or LLM calls.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "04_outer_loop_llm.ipynb"


def _extract_function_signatures(source: str, names: list[str]) -> dict[str, str]:
    """Find `def NAME(...)` lines in raw notebook source and return the arg lists."""
    found: dict[str, str] = {}
    for name in names:
        m = re.search(rf"def\s+{re.escape(name)}\s*\(([^)]*)\)", source)
        if m:
            found[name] = m.group(1).strip()
    return found


def _assert_two_arg_callable(sig_str: str, fn_name: str) -> None:
    """Parse 'w, structure=None' → assert ≥ 2 accepted positional args."""
    # Build a dummy function with that signature and ask inspect if it takes 2 args.
    dummy_src = f"def _dummy({sig_str}): return None"
    ns: dict = {}
    exec(dummy_src, ns)
    sig = inspect.signature(ns["_dummy"])
    params = list(sig.parameters.values())
    assert len(params) >= 2, (
        f"{fn_name} has signature ({sig_str}) — outer_loop calls it with 2 args "
        f"(weights, structure), but it only accepts {len(params)}."
    )
    # Both positions must be bindable (either positional-or-keyword or positional-only,
    # not keyword-only or **kwargs).
    bindable = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    assert params[0].kind in bindable, (
        f"{fn_name}: first parameter must be positional (got {params[0].kind})"
    )
    assert params[1].kind in bindable, (
        f"{fn_name}: second parameter must be positional (got {params[1].kind})"
    )


def main() -> int:
    assert NOTEBOOK.exists(), f"notebook not found: {NOTEBOOK}"
    nb = json.loads(NOTEBOOK.read_text())

    target_source = None
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def get_metrics_fn" in src and "def get_residuals_fn" in src:
            target_source = src
            break

    assert target_source is not None, (
        "could not find a cell in 04_outer_loop_llm.ipynb defining both "
        "get_metrics_fn and get_residuals_fn"
    )

    sigs = _extract_function_signatures(
        target_source, ["get_metrics_fn", "get_residuals_fn"]
    )
    assert "get_metrics_fn" in sigs, "get_metrics_fn definition not matched by regex"
    assert "get_residuals_fn" in sigs, "get_residuals_fn definition not matched by regex"

    _assert_two_arg_callable(sigs["get_metrics_fn"], "get_metrics_fn")
    _assert_two_arg_callable(sigs["get_residuals_fn"], "get_residuals_fn")

    print("PASS  get_metrics_fn   signature:", sigs["get_metrics_fn"])
    print("PASS  get_residuals_fn signature:", sigs["get_residuals_fn"])
    print()
    print("Both callbacks accept (weights, structure) — outer loop is safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
