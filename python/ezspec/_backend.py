"""Selects the numerical backend: compiled Rust core or Python reference.

>>> from ezspec._backend import core
>>> core().gaussian(x, 1.0, 0.0, 1.0)

The Rust core is used when available. ``set_backend("python")`` switches to
the NumPy/SciPy reference implementation (used by the test-suite to compare
both, and as automatic fallback when the extension is not compiled).
"""

from __future__ import annotations

import contextlib
import numpy as np

from . import _reference

try:  # pragma: no cover - depends on build
    from . import _core as _rust
except ImportError:  # pragma: no cover
    _rust = None

HAVE_RUST = _rust is not None
_active = _rust if HAVE_RUST else _reference


def set_backend(name: str) -> None:
    """Select ``"rust"`` or ``"python"``."""
    global _active
    if name == "rust":
        if not HAVE_RUST:
            raise RuntimeError("the compiled extension ezspec._core is not available")
        _active = _rust
    elif name == "python":
        _active = _reference
    else:
        raise ValueError(f"unknown backend {name!r}")


def backend_name() -> str:
    return "rust" if _active is _rust and HAVE_RUST else "python"


@contextlib.contextmanager
def use_backend(name: str):
    previous = backend_name()
    set_backend(name)
    try:
        yield
    finally:
        set_backend(previous)


def core():
    """The active backend module."""
    return _active


def as_f64(a) -> np.ndarray:
    """Contiguous float64 copy-free view where possible (required by the Rust core)."""
    return np.ascontiguousarray(a, dtype=np.float64)
