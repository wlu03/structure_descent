"""YAML-driven schema translator for the PO-LEU data layer (Wave 8, design doc §2, §4).

This module takes a *dataset YAML* (see ``configs/datasets/amazon.yaml`` for the
reference shape) and returns a frozen :class:`DatasetSchema` describing every
column rename, dtype coercion, categorical mapping, constant, composite
formula, external CSV lookup, or event-derived aggregate needed to translate a
dataset's raw events/persons DataFrames into the canonical schema consumed by
``src/data/person_features.py`` and ``src/data/context_string.py``.

Nine kinds of ``z_d`` translator are supported (exact set locked in the Wave 8
design):

    categorical_map, categorical_map_with_collapse, categorical_to_int,
    ordinal_map, constant, composite, external_lookup, derived_from_events,
    passthrough

Composite formulas are evaluated via a *restricted AST walker* (stdlib
``ast``) — ``eval``/``exec`` are never used, and the whitelist is deliberately
narrow: arithmetic, unary minus, ``==`` comparisons, numeric/string literals,
``min``/``max``/``abs``, and column references. Backtick-quoted identifiers
(e.g. ``\\`Q-personal-diabetes\\```) are supported so survey columns with
hyphens can be referenced verbatim.

The module is pure — no I/O on import, no globals, no RNG. The one filesystem
touch is ``load_schema``'s YAML read (and the per-field external_lookup CSV
read during ``translate_persons``).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

__all__ = [
    "DatasetSchema",
    "ZDFieldSpec",
    "UnknownCategoryError",
    "CompositeFormulaError",
    "load_schema",
    "translate_events",
    "translate_persons",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class UnknownCategoryError(ValueError):
    """Raised when a raw categorical value has no entry in ``values`` and is
    not listed in ``drop_on_unknown``."""


class CompositeFormulaError(ValueError):
    """Raised when a composite formula references an unsupported AST node, an
    unknown column, or a non-whitelisted function."""


# --------------------------------------------------------------------------- #
# Canonical kinds and z_d column order
# --------------------------------------------------------------------------- #


VALID_KINDS: frozenset[str] = frozenset(
    {
        "categorical_map",
        "categorical_map_with_collapse",
        "categorical_to_int",
        "ordinal_map",
        "constant",
        "composite",
        "external_lookup",
        "derived_from_events",
        "passthrough",
    }
)

# Canonical dtype-coercion keys supported by translate_events.
_DTYPE_COERCIONS: frozenset[str] = frozenset({"float", "int", "str"})


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ZDFieldSpec:
    """Per-column translator for a single canonical ``z_d`` column.

    ``canonical_column`` is the output name (e.g. ``"income_bucket"``).
    ``kind`` is one of :data:`VALID_KINDS`. The remaining fields are only
    meaningful for particular kinds; see the dispatch table in
    :func:`translate_persons`.
    """

    canonical_column: str
    kind: str
    source: str | None = None
    values: dict[Any, Any] | None = None
    drop_on_unknown: tuple[str, ...] = ()
    value: Any | None = None
    formula: str | None = None
    clamp: tuple[float, float] | None = None
    fallback: Any | None = None
    lookup_path: Path | None = None
    aggregator: str | None = None
    aggregator_column: str | None = None
    group_by: str | None = None
    note: str = ""


@dataclass(frozen=True)
class DatasetSchema:
    """Frozen parse of a dataset YAML.

    Field names mirror the YAML structure one-to-one where possible.
    """

    name: str
    description: str

    # Events
    events_path: Path
    events_parse_dates: tuple[str, ...]
    events_column_map: dict[str, str]
    events_dropna_subset: tuple[str, ...]
    events_category_null_fill: str
    events_dtype_coerce: dict[str, str]

    # Persons
    persons_path: Path
    persons_id_column: str

    # Ordered z_d translators (one per canonical z_d column).
    z_d_mapping: tuple[ZDFieldSpec, ...]

    # Training block
    choice_set_size: int
    n_resamples: int
    val_frac: float
    test_frac: float
    subsample_enabled: bool
    subsample_n_customers: int
    subsample_seed: int


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{context}: missing required key {key!r}.")
    return mapping[key]


def _build_z_d_spec(canonical_column: str, raw: Mapping[str, Any]) -> ZDFieldSpec:
    """Turn one ``z_d_mapping`` YAML entry into a :class:`ZDFieldSpec`.

    Applies a couple of conveniences:

    * ``derived_from_events`` is encoded in YAML with ``source: derived_from_events``;
      the spec object stores ``kind="derived_from_events"`` and ``source=None``.
    * ``clamp`` is normalized to a ``(lo, hi)`` tuple.
    * ``drop_on_unknown`` is normalized to a tuple.
    """
    kind = raw.get("kind")
    source = raw.get("source")

    # The YAML uses `source: derived_from_events` as a shorthand; normalize.
    if source == "derived_from_events":
        kind = "derived_from_events"
        source = None

    if kind is None:
        raise ValueError(
            f"z_d column {canonical_column!r}: missing 'kind' (and no "
            f"'source: derived_from_events' shorthand)."
        )
    if kind not in VALID_KINDS:
        raise ValueError(
            f"z_d column {canonical_column!r}: unknown kind {kind!r}. "
            f"Must be one of {sorted(VALID_KINDS)}."
        )

    lookup_path = raw.get("lookup_path")
    if lookup_path is not None:
        lookup_path = Path(lookup_path)

    clamp = raw.get("clamp")
    if clamp is not None:
        if not (isinstance(clamp, (list, tuple)) and len(clamp) == 2):
            raise ValueError(
                f"z_d column {canonical_column!r}: 'clamp' must be a 2-element "
                f"[lo, hi] list; got {clamp!r}."
            )
        clamp = (float(clamp[0]), float(clamp[1]))

    drop_on_unknown = tuple(raw.get("drop_on_unknown") or ())

    values = raw.get("values")
    if values is not None and not isinstance(values, dict):
        raise ValueError(
            f"z_d column {canonical_column!r}: 'values' must be a mapping; "
            f"got {type(values).__name__}."
        )

    return ZDFieldSpec(
        canonical_column=canonical_column,
        kind=kind,
        source=source,
        values=values,
        drop_on_unknown=drop_on_unknown,
        value=raw.get("value"),
        formula=raw.get("formula"),
        clamp=clamp,
        fallback=raw.get("fallback"),
        lookup_path=lookup_path,
        aggregator=raw.get("aggregator"),
        aggregator_column=raw.get("aggregator_column"),
        group_by=raw.get("group_by"),
        note=str(raw.get("note") or "").strip(),
    )


def load_schema(path: Path | str) -> DatasetSchema:
    """Parse a dataset YAML into a frozen :class:`DatasetSchema`.

    Parameters
    ----------
    path:
        Path to a dataset YAML (e.g. ``configs/datasets/amazon.yaml``).

    Returns
    -------
    DatasetSchema
        Frozen dataclass. All keys required by :class:`DatasetSchema` must be
        present in the YAML or a ``KeyError`` / ``ValueError`` is raised.
    """
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "dataset" not in doc:
        raise ValueError(f"{yaml_path}: top-level 'dataset:' block missing.")

    ds = doc["dataset"]
    name = str(_require(ds, "name", "dataset"))
    description = str(ds.get("description", "")).strip()

    events = _require(ds, "events", f"dataset {name!r}")
    events_path = Path(_require(events, "path", "dataset.events"))
    events_parse_dates = tuple(events.get("parse_dates") or ())
    column_map_raw = _require(events, "column_map", "dataset.events")
    events_column_map: dict[str, str] = {str(k): str(v) for k, v in column_map_raw.items()}
    events_dropna_subset = tuple(events.get("dropna_subset") or ())
    events_category_null_fill = str(events.get("category_null_fill", "Unknown"))
    events_dtype_coerce_raw = events.get("dtype_coerce") or {}
    events_dtype_coerce: dict[str, str] = {}
    for col, dtype in events_dtype_coerce_raw.items():
        dtype_s = str(dtype)
        if dtype_s not in _DTYPE_COERCIONS:
            raise ValueError(
                f"dataset.events.dtype_coerce[{col!r}]: unsupported dtype "
                f"{dtype_s!r}. Must be one of {sorted(_DTYPE_COERCIONS)}."
            )
        events_dtype_coerce[str(col)] = dtype_s

    persons = _require(ds, "persons", f"dataset {name!r}")
    persons_path = Path(_require(persons, "path", "dataset.persons"))
    persons_id_column = str(_require(persons, "id_column", "dataset.persons"))

    z_d_mapping_raw = _require(persons, "z_d_mapping", "dataset.persons")
    if not isinstance(z_d_mapping_raw, dict):
        raise ValueError("dataset.persons.z_d_mapping must be a mapping.")
    # Preserve YAML iteration order (Python dicts are insertion-ordered and
    # PyYAML preserves key order on safe_load in 5.1+).
    z_d_mapping = tuple(
        _build_z_d_spec(str(canonical), raw) for canonical, raw in z_d_mapping_raw.items()
    )

    training = _require(ds, "training", f"dataset {name!r}")
    subsample = training.get("subsample") or {}

    schema = DatasetSchema(
        name=name,
        description=description,
        events_path=events_path,
        events_parse_dates=events_parse_dates,
        events_column_map=events_column_map,
        events_dropna_subset=events_dropna_subset,
        events_category_null_fill=events_category_null_fill,
        events_dtype_coerce=events_dtype_coerce,
        persons_path=persons_path,
        persons_id_column=persons_id_column,
        z_d_mapping=z_d_mapping,
        choice_set_size=int(_require(training, "choice_set_size", "dataset.training")),
        n_resamples=int(_require(training, "n_resamples", "dataset.training")),
        val_frac=float(_require(training, "val_frac", "dataset.training")),
        test_frac=float(_require(training, "test_frac", "dataset.training")),
        subsample_enabled=bool(subsample.get("enabled", False)),
        subsample_n_customers=int(subsample.get("n_customers", 0)),
        subsample_seed=int(subsample.get("seed", 0)),
    )
    logger.debug(
        "Loaded dataset schema %r with %d z_d columns from %s.",
        schema.name,
        len(schema.z_d_mapping),
        yaml_path,
    )
    return schema


# --------------------------------------------------------------------------- #
# Events translation
# --------------------------------------------------------------------------- #


def translate_events(
    events_raw: pd.DataFrame,
    schema: DatasetSchema,
) -> pd.DataFrame:
    """Apply events_column_map, dropna_subset, dtype_coerce, category_null_fill.

    Behaviour:

    * ``events_column_map`` is applied first. Every raw key in the map that
      is absent from ``events_raw`` raises ``KeyError`` (strict, no escape).
      If multiple raw keys alias to the same canonical name (e.g. Amazon's
      ``"ASIN/ISBN (Product Code)"`` and ``"ASIN/ISBN"``), only the raw keys
      actually present are renamed; if none of a group are present, ``KeyError``.
    * After rename, the ``dropna_subset`` canonical columns are used for a
      ``dropna``.
    * ``dtype_coerce`` is applied to the canonical columns (``float``, ``int``,
      ``str``). Unknown dtype tokens were rejected at load time.
    * ``category`` column NaNs (if present) are filled with
      ``events_category_null_fill``.

    Returns a new DataFrame; the input is never mutated.
    """
    # Group raw keys by canonical target so a set of aliases counts as satisfied
    # if ANY raw alias is present. This matches the Amazon YAML which lists both
    # "ASIN/ISBN (Product Code)" and "ASIN/ISBN" mapping to "asin".
    canonical_to_raw: dict[str, list[str]] = {}
    for raw_key, canonical in schema.events_column_map.items():
        canonical_to_raw.setdefault(canonical, []).append(raw_key)

    rename_map: dict[str, str] = {}
    missing_groups: list[str] = []
    for canonical, raw_keys in canonical_to_raw.items():
        present = [r for r in raw_keys if r in events_raw.columns]
        if not present:
            missing_groups.append(
                f"{canonical!r} (expected one of {raw_keys})"
            )
            continue
        # Rename the first present alias; drop the rest if they're also there
        # (to avoid a column-name collision after rename).
        chosen = present[0]
        rename_map[chosen] = canonical

    if missing_groups:
        raise KeyError(
            "translate_events: events DataFrame is missing required raw "
            f"columns for canonical targets: {', '.join(missing_groups)}."
        )

    df = events_raw.rename(columns=rename_map).copy()

    # Drop any other raw aliases for the same canonical target, keeping only
    # the one just renamed, to avoid duplicate canonical names.
    for canonical, raw_keys in canonical_to_raw.items():
        dupes = [r for r in raw_keys if r in df.columns and r != canonical]
        if dupes:
            df = df.drop(columns=dupes)

    # dropna on canonical columns
    if schema.events_dropna_subset:
        missing_canon = [
            c for c in schema.events_dropna_subset if c not in df.columns
        ]
        if missing_canon:
            raise KeyError(
                "translate_events: dropna_subset references canonical columns "
                f"not present after rename: {missing_canon}."
            )
        df = df.dropna(subset=list(schema.events_dropna_subset))

    # dtype coercion
    for col, dtype_tok in schema.events_dtype_coerce.items():
        if col not in df.columns:
            # dtype_coerce may name columns that weren't in the map; skip
            # silently (YAML author's choice which canonical columns to coerce).
            continue
        if dtype_tok == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif dtype_tok == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype_tok == "str":
            df[col] = df[col].astype(str)

    # category null fill (only if the canonical `category` column exists).
    if "category" in df.columns:
        df["category"] = df["category"].fillna(schema.events_category_null_fill)

    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Composite formula parser (restricted AST, no eval())
# --------------------------------------------------------------------------- #


# Whitelisted AST node types.
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY_OPS = (ast.USub,)
_ALLOWED_COMPARE_OPS = (ast.Eq,)
_ALLOWED_CALLS = frozenset({"min", "max", "abs"})


def _backtick_preprocess(formula: str) -> tuple[str, dict[str, str]]:
    """Replace backtick-quoted column names with a safe identifier.

    Returns the rewritten formula and a mapping ``safe_ident -> original_name``
    so ``ast.Name`` lookups can round-trip.

    Example: ``5 - (`Q-personal-diabetes` == 'Yes')`` becomes
    ``5 - (_bt_0 == 'Yes')`` with ``{"_bt_0": "Q-personal-diabetes"}``.
    """
    out_chars: list[str] = []
    alias_by_name: dict[str, str] = {}
    name_by_alias: dict[str, str] = {}
    i = 0
    n = len(formula)
    while i < n:
        ch = formula[i]
        if ch == "`":
            j = formula.find("`", i + 1)
            if j < 0:
                raise CompositeFormulaError(
                    f"Unterminated backtick quote in formula: {formula!r}."
                )
            original = formula[i + 1 : j]
            if not original:
                raise CompositeFormulaError(
                    f"Empty backtick-quoted identifier in formula: {formula!r}."
                )
            if original in alias_by_name:
                alias = alias_by_name[original]
            else:
                alias = f"_bt_{len(alias_by_name)}"
                alias_by_name[original] = alias
                name_by_alias[alias] = original
            out_chars.append(alias)
            i = j + 1
        else:
            out_chars.append(ch)
            i += 1
    return "".join(out_chars), name_by_alias


def _collect_names(tree: ast.AST, alias_to_original: dict[str, str]) -> set[str]:
    """Collect the original column names referenced by all ast.Name nodes."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(alias_to_original.get(node.id, node.id))
    return names


def _validate_ast(tree: ast.AST, alias_to_original: dict[str, str]) -> None:
    """Walk the tree and reject any node outside the whitelist."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise CompositeFormulaError(
                    f"Disallowed binary operator {type(node.op).__name__}."
                )
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise CompositeFormulaError(
                    f"Disallowed unary operator {type(node.op).__name__}."
                )
            continue
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(node.ops[0], _ALLOWED_COMPARE_OPS):
                raise CompositeFormulaError(
                    "Only single-operand '==' comparisons are allowed in composite formulas."
                )
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str, bool)) or node.value is None:
                raise CompositeFormulaError(
                    f"Disallowed constant literal {node.value!r}."
                )
            continue
        if isinstance(node, ast.Name):
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise CompositeFormulaError(
                    "Only top-level function calls (min/max/abs) are allowed."
                )
            if node.func.id not in _ALLOWED_CALLS:
                raise CompositeFormulaError(
                    f"Disallowed function call {node.func.id!r}. "
                    f"Allowed: {sorted(_ALLOWED_CALLS)}."
                )
            if node.keywords:
                raise CompositeFormulaError(
                    "Keyword arguments are not allowed in composite formulas."
                )
            continue
        if isinstance(node, (ast.Load, ast.Store)):
            continue
        # Bare operator nodes appear in ast.walk() as children of BinOp /
        # UnaryOp / Compare; they were already validated in-context above,
        # so accept the whitelisted concrete op types here.
        if isinstance(node, _ALLOWED_BIN_OPS) or isinstance(node, _ALLOWED_UNARY_OPS) \
                or isinstance(node, _ALLOWED_COMPARE_OPS):
            continue
        # Anything else (Attribute, Subscript, Lambda, Import, etc.) is rejected.
        raise CompositeFormulaError(
            f"Disallowed AST node {type(node).__name__} in composite formula."
        )


def _eval_node(node: ast.AST, row: Mapping[str, Any], alias_to_original: dict[str, str]) -> Any:
    """Evaluate a single whitelisted AST node against a row mapping."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, row, alias_to_original)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        original = alias_to_original.get(node.id, node.id)
        # Mapping-style lookup (pandas Series supports __getitem__).
        return row[original]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, row, alias_to_original)
        # USub is the only allowed unary op (validated above).
        return -_to_number(operand)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, row, alias_to_original)
        right = _eval_node(node.right, row, alias_to_original)
        lf = _to_number(left)
        rf = _to_number(right)
        op = node.op
        if isinstance(op, ast.Add):
            return lf + rf
        if isinstance(op, ast.Sub):
            return lf - rf
        if isinstance(op, ast.Mult):
            return lf * rf
        if isinstance(op, ast.Div):
            if rf == 0 or (isinstance(rf, float) and not np.isfinite(rf)):
                return float("nan")
            return lf / rf
        if isinstance(op, ast.Pow):
            return lf ** rf
        # Unreachable given validator.
        raise CompositeFormulaError(f"Unexpected BinOp {type(op).__name__}.")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, row, alias_to_original)
        right = _eval_node(node.comparators[0], row, alias_to_original)
        # Single op Eq (validated).
        return 1 if left == right else 0
    if isinstance(node, ast.Call):
        fn = node.func.id  # type: ignore[union-attr]
        args = [_eval_node(a, row, alias_to_original) for a in node.args]
        if fn == "min":
            return min(*[_to_number(a) for a in args])
        if fn == "max":
            return max(*[_to_number(a) for a in args])
        if fn == "abs":
            if len(args) != 1:
                raise CompositeFormulaError("abs() takes exactly one argument.")
            return abs(_to_number(args[0]))
    raise CompositeFormulaError(
        f"Cannot evaluate AST node {type(node).__name__}."
    )


def _to_number(value: Any) -> float:
    """Coerce bool/int/float (including bool from == comparisons) to float."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if value is None:
        return float("nan")
    # Strings in arithmetic context are always an error.
    raise CompositeFormulaError(
        f"Cannot coerce value {value!r} (type {type(value).__name__}) to a number."
    )


def _parse_composite(
    formula: str, available_columns: set[str]
) -> tuple[ast.AST, dict[str, str]]:
    """Parse-and-validate a composite formula; return the AST tree and the
    backtick-alias map. Raise CompositeFormulaError on any violation OR on
    any reference to a column not in ``available_columns``."""
    if not isinstance(formula, str) or not formula.strip():
        raise CompositeFormulaError("Composite formula must be a non-empty string.")
    rewritten, alias_map = _backtick_preprocess(formula)
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise CompositeFormulaError(
            f"Syntax error in composite formula {formula!r}: {exc.msg}"
        ) from exc
    _validate_ast(tree, alias_map)
    # Column-existence check.
    referenced = _collect_names(tree, alias_map)
    # Remove whitelisted function names (they are ast.Name nodes at Call.func).
    referenced = {n for n in referenced if n not in _ALLOWED_CALLS}
    missing = referenced - available_columns
    if missing:
        raise CompositeFormulaError(
            f"Composite formula {formula!r} references missing columns: "
            f"{sorted(missing)}."
        )
    return tree, alias_map


# --------------------------------------------------------------------------- #
# Kind handlers for translate_persons
# --------------------------------------------------------------------------- #


def _apply_categorical_map(
    series: pd.Series,
    spec: ZDFieldSpec,
    *,
    drop_mask: pd.Series,
    collapse: bool,
) -> pd.Series:
    """Shared backbone for categorical_map / categorical_map_with_collapse.

    The caller pre-marks ``drop_mask`` (rows whose source value is in
    ``drop_on_unknown``). Any remaining value not in ``values`` raises.
    """
    if spec.values is None:
        raise ValueError(
            f"z_d column {spec.canonical_column!r} kind={spec.kind}: 'values' is required."
        )
    values_map = spec.values
    out = pd.Series(index=series.index, dtype=object)
    for idx, val in series.items():
        if drop_mask.loc[idx]:
            continue
        if val in values_map:
            out.loc[idx] = values_map[val]
            continue
        # Compare as string too, to be forgiving of YAML quoting drift.
        sval = val if not isinstance(val, float) else val
        raise UnknownCategoryError(
            f"Unknown value {val!r} in column {spec.source!r} "
            f"(canonical {spec.canonical_column!r}). Known keys: "
            f"{sorted(map(str, values_map.keys()))}. "
            f"Add it to 'values' or to 'drop_on_unknown'."
        )
    _ = collapse  # silence linter; collapse is documentation-only.
    return out


def _apply_categorical_to_int(series: pd.Series, spec: ZDFieldSpec) -> pd.Series:
    """categorical_to_int / ordinal_map: map stripped string to int."""
    if spec.values is None:
        raise ValueError(
            f"z_d column {spec.canonical_column!r} kind={spec.kind}: 'values' is required."
        )
    values_map = spec.values
    # Build a stripped-key view for tolerant matching.
    stripped = {
        (k.strip() if isinstance(k, str) else k): v for k, v in values_map.items()
    }
    out = pd.Series(index=series.index, dtype="Int64")
    for idx, val in series.items():
        key = val.strip() if isinstance(val, str) else val
        if key in values_map:
            out.loc[idx] = int(values_map[key])
            continue
        if key in stripped:
            out.loc[idx] = int(stripped[key])
            continue
        raise UnknownCategoryError(
            f"Unknown value {val!r} in column {spec.source!r} "
            f"(canonical {spec.canonical_column!r}, kind={spec.kind}). "
            f"Known keys: {sorted(map(str, values_map.keys()))}."
        )
    return out


def _apply_constant(n_rows: int, spec: ZDFieldSpec, index: pd.Index) -> pd.Series:
    """Constant: broadcast ``spec.value`` to every row."""
    if spec.source is not None:
        raise ValueError(
            f"z_d column {spec.canonical_column!r} kind=constant: "
            f"source must be None, got {spec.source!r}."
        )
    return pd.Series([spec.value] * n_rows, index=index)


def _apply_composite(persons_raw: pd.DataFrame, spec: ZDFieldSpec) -> pd.Series:
    """Composite: evaluate the (validated) AST for every row."""
    if spec.formula is None:
        raise ValueError(
            f"z_d column {spec.canonical_column!r} kind=composite: 'formula' is required."
        )
    tree, alias_map = _parse_composite(spec.formula, set(persons_raw.columns))
    out = pd.Series(index=persons_raw.index, dtype=float)
    for idx, row in persons_raw.iterrows():
        try:
            v = _eval_node(tree, row, alias_map)
            fv = float(v)
            if not np.isfinite(fv):
                fv = float("nan")
        except CompositeFormulaError:
            raise
        except Exception:
            fv = float("nan")
        if np.isnan(fv):
            if spec.fallback is not None:
                fv = float(spec.fallback)
            # else leave as NaN
        if spec.clamp is not None and not np.isnan(fv):
            lo, hi = spec.clamp
            fv = min(max(fv, lo), hi)
        out.loc[idx] = fv
    return out


def _apply_external_lookup(persons_raw: pd.DataFrame, spec: ZDFieldSpec) -> pd.Series:
    """External lookup: load CSV (columns: key, value); map source column."""
    if spec.source is None:
        raise ValueError(
            f"z_d column {spec.canonical_column!r} kind=external_lookup: "
            f"'source' is required."
        )
    if spec.lookup_path is None:
        raise ValueError(
            f"z_d column {spec.canonical_column!r} kind=external_lookup: "
            f"'lookup_path' is required."
        )
    csv_path = spec.lookup_path
    table = pd.read_csv(csv_path)
    if list(table.columns)[:2] != ["key", "value"] and not {"key", "value"}.issubset(table.columns):
        raise ValueError(
            f"external_lookup CSV {csv_path} must have 'key' and 'value' columns; "
            f"found {list(table.columns)}."
        )
    if len(table) == 0:
        # Header-only: every row falls back.
        logger.info(
            "external_lookup CSV %s is empty; every row falls through to fallback=%r.",
            csv_path,
            spec.fallback,
        )
        return pd.Series(
            [spec.fallback] * len(persons_raw), index=persons_raw.index
        )
    lookup = dict(zip(table["key"].astype(str), table["value"]))
    source = persons_raw[spec.source]
    mapped = source.astype(str).map(lookup)
    return mapped.where(mapped.notna(), spec.fallback)


def _apply_derived_from_events(
    persons: pd.DataFrame,
    spec: ZDFieldSpec,
    training_events: pd.DataFrame,
    *,
    persons_id_canonical: str,
) -> pd.Series:
    """derived_from_events: aggregate training_events; join back by id.

    ``aggregator`` is ``"count"`` or ``"mean"``. For ``mean`` the
    ``aggregator_column`` must name a column present on ``training_events``.
    Customers in ``persons`` with no matching event rows get NaN.
    """
    if spec.aggregator not in {"count", "mean"}:
        raise ValueError(
            f"z_d column {spec.canonical_column!r}: derived_from_events "
            f"aggregator must be 'count' or 'mean'; got {spec.aggregator!r}."
        )
    group_by = spec.group_by or "customer_id"
    if group_by not in training_events.columns:
        raise ValueError(
            f"z_d column {spec.canonical_column!r}: derived_from_events "
            f"group_by column {group_by!r} is not on training_events "
            f"(columns={list(training_events.columns)})."
        )
    if spec.aggregator == "count":
        agg = training_events.groupby(group_by).size()
    else:  # mean
        if not spec.aggregator_column:
            raise ValueError(
                f"z_d column {spec.canonical_column!r}: aggregator=mean requires "
                f"'aggregator_column'."
            )
        if spec.aggregator_column not in training_events.columns:
            raise ValueError(
                f"z_d column {spec.canonical_column!r}: aggregator_column "
                f"{spec.aggregator_column!r} not on training_events. "
                f"(Did state_features run before translate_persons?)"
            )
        agg = training_events.groupby(group_by)[spec.aggregator_column].mean()
    # Join back via the persons id column (already renamed to canonical
    # 'customer_id' by the caller).
    ids = persons[persons_id_canonical]
    return ids.map(agg)


# --------------------------------------------------------------------------- #
# translate_persons
# --------------------------------------------------------------------------- #


def translate_persons(
    persons_raw: pd.DataFrame,
    schema: DatasetSchema,
    *,
    training_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Translate the raw persons DataFrame into canonical z_d columns.

    The returned DataFrame has exactly the columns listed in
    ``schema.z_d_mapping`` (in order), plus a canonical ``customer_id``
    column (renamed from ``schema.persons_id_column``).

    ``drop_on_unknown`` policy: any row whose *source* column value is in
    the drop list is removed from the output entirely before translation
    proceeds — no z_d field is emitted for those rows.

    For ``derived_from_events`` fields, ``training_events`` must be provided
    and is assumed to already carry the canonical column naming
    (``customer_id`` + any state-features output the aggregator references).
    """
    # Determine which fields need training_events.
    needs_events = any(
        spec.kind == "derived_from_events" for spec in schema.z_d_mapping
    )
    if needs_events and training_events is None:
        raise ValueError(
            "translate_persons: at least one z_d field uses "
            "kind='derived_from_events'; `training_events` must be provided."
        )

    # Canonicalize the id column.
    id_col_raw = schema.persons_id_column
    if id_col_raw not in persons_raw.columns:
        raise KeyError(
            f"translate_persons: persons DataFrame missing id column "
            f"{id_col_raw!r} (columns={list(persons_raw.columns)})."
        )
    working = persons_raw.copy()
    if id_col_raw != "customer_id":
        if "customer_id" in working.columns and id_col_raw != "customer_id":
            working = working.drop(columns=["customer_id"])
        working = working.rename(columns={id_col_raw: "customer_id"})

    # Apply drop_on_unknown first: union across all specs that carry the list.
    drop_mask = pd.Series(False, index=working.index)
    for spec in schema.z_d_mapping:
        if spec.drop_on_unknown and spec.source and spec.source in working.columns:
            mask = working[spec.source].isin(list(spec.drop_on_unknown))
            drop_mask = drop_mask | mask
    if drop_mask.any():
        dropped = int(drop_mask.sum())
        logger.info(
            "translate_persons: dropping %d rows via drop_on_unknown filter.", dropped
        )
        working = working.loc[~drop_mask].reset_index(drop=True)

    out = pd.DataFrame(index=working.index)
    out["customer_id"] = working["customer_id"].values

    for spec in schema.z_d_mapping:
        kind = spec.kind
        if kind == "categorical_map":
            source_series = working[spec.source] if spec.source else None
            if source_series is None:
                raise ValueError(
                    f"z_d column {spec.canonical_column!r}: categorical_map requires 'source'."
                )
            # drop_on_unknown already pruned; at this point every remaining
            # value must be in spec.values.
            # Pass an all-False mask — we already dropped at the top.
            col = _apply_categorical_map(
                source_series,
                spec,
                drop_mask=pd.Series(False, index=source_series.index),
                collapse=False,
            )
        elif kind == "categorical_map_with_collapse":
            source_series = working[spec.source]
            col = _apply_categorical_map(
                source_series,
                spec,
                drop_mask=pd.Series(False, index=source_series.index),
                collapse=True,
            )
        elif kind == "categorical_to_int":
            col = _apply_categorical_to_int(working[spec.source], spec)
        elif kind == "ordinal_map":
            # Alias: identical handler to categorical_to_int (documentation-only
            # distinction per the Wave 8 design doc).
            col = _apply_categorical_to_int(working[spec.source], spec)
        elif kind == "constant":
            col = _apply_constant(len(working), spec, working.index)
        elif kind == "composite":
            col = _apply_composite(working, spec)
        elif kind == "external_lookup":
            col = _apply_external_lookup(working, spec)
        elif kind == "derived_from_events":
            col = _apply_derived_from_events(
                working,
                spec,
                training_events,  # type: ignore[arg-type]  # None-case guarded above
                persons_id_canonical="customer_id",
            )
        elif kind == "passthrough":
            if not spec.source:
                raise ValueError(
                    f"z_d column {spec.canonical_column!r}: passthrough requires 'source'."
                )
            if spec.source not in working.columns:
                raise KeyError(
                    f"z_d column {spec.canonical_column!r}: passthrough source "
                    f"{spec.source!r} missing from persons DataFrame."
                )
            col = working[spec.source].copy()
        else:
            # Guarded by VALID_KINDS at load time; defensive.
            raise ValueError(f"Unhandled z_d kind {kind!r}.")

        out[spec.canonical_column] = col.values if hasattr(col, "values") else col

    # Column order: customer_id then the canonical z_d columns in YAML order.
    ordered = ["customer_id"] + [s.canonical_column for s in schema.z_d_mapping]
    return out[ordered].reset_index(drop=True)
