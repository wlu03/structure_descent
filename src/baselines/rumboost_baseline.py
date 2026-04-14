"""
RUMBoost baseline — DISABLED (Path B stub).

Reference
---------
Salvade, N. and Hillel, T. (2024). "RUMBoost: Gradient boosted random
utility models." (https://github.com/big-ucl/rumboost)

Status
------
BLOCKED on a NumPy 2.x ABI incompatibility in the rumboost dependency
chain. The package is installed under site-packages but ``import rumboost``
fails transitively via biogeme -> cythonbiogeme, a Cython extension
compiled against NumPy 1.x. Under this project's NumPy 2.x the C
extension raises::

    ImportError: numpy.core.multiarray failed to import

Downgrading to numpy<2 would break the rest of the Structure Descent
pipeline (pandas 2.x, the DSL, data prep, and the other baselines), so
it is not a safe in-place fix.

Design choice
-------------
This stub used to ``raise ImportError`` at module-import time, which
silently broke ``from src.baselines import ...`` whenever the package
was present. That was wrong: a disabled baseline should not block
importing the other baselines.

The stub now imports cleanly and raises ``NotImplementedError`` only
when ``RUMBoostBaseline`` is actually instantiated, with an actionable
install hint. ``run_all.py`` already omits RUMBoost from the registry,
so disabling this baseline does not affect the rest of the suite.

Unblock path
------------
Create a separate Python 3.11 venv pinned to NumPy 1.x::

    python3.11 -m venv .venv-rumboost
    .venv-rumboost/bin/pip install "numpy<2" rumboost biogeme \
        cythonbiogeme lightgbm

or wait for an upstream cythonbiogeme wheel rebuilt against NumPy 2.x
(track https://github.com/michelbierlaire/cythonbiogeme). Do NOT
substitute a vanilla gradient-booster as a RUMBoost proxy — the whole
point of RUMBoost is the RUM-consistent ensemble structure and
monotonicity constraints from the paper.
"""

from __future__ import annotations


_INSTALL_HINT = (
    "RUMBoost baseline is disabled in this environment due to a NumPy 2.x / "
    "cythonbiogeme ABI incompatibility. To enable it, create a separate "
    "Python 3.11 venv pinned to NumPy 1.x: "
    "`pip install 'numpy<2' rumboost biogeme cythonbiogeme lightgbm`. "
    "See the module docstring for details."
)


class RUMBoostBaseline:
    """RUMBoost stub — raises NotImplementedError at instantiation.

    Kept as a named class so ``run_all.py`` / notebooks can probe for
    availability via ``try: RUMBoostBaseline(); except NotImplementedError``.
    """

    name = "RUMBoost"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_INSTALL_HINT)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(_INSTALL_HINT)
