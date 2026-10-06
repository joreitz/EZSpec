"""Registry of versioned, scriptable processing operations.

Every GUI action that changes data is a call of one of these operations with
plain-JSON parameters. That makes the processing history storable
(``pipeline.json``), undoable, and exportable as a Python script that
reproduces the analysis without the GUI::

    s = ops.crop(s, xmin=200.0, xmax=1800.0)
    s = ops.baseline_arpls(s, lam=1e6)

If the semantics of an operation change, its ``version`` is incremented and
old projects keep a record of the version they were created with.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ..spectrum import Spectrum


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str              # float | log_float | int | bool | choice | str | range | ranges | anchors
    default: Any = None
    label: str = ""
    min: float | None = None
    max: float | None = None
    choices: tuple = ()
    help: str = ""
    optional: bool = False   # None allowed


@dataclass(frozen=True)
class OpSpec:
    name: str
    version: int
    title: str
    category: str
    params: tuple
    func: Callable
    description: str = ""
    flags: tuple = ()        # flags the op adds to the spectrum

    def defaults(self) -> dict:
        return {p.name: _copy(p.default) for p in self.params}

    def param(self, name) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def validate(self, params: dict) -> dict:
        """Fill defaults, reject unknown names, coerce simple types."""
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise TypeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        out = self.defaults()
        for p in self.params:
            if p.name not in params:
                continue
            v = params[p.name]
            if v is None:
                if not p.optional:
                    raise ValueError(f"{self.name}: parameter {p.name!r} must not be None")
                out[p.name] = None
                continue
            if p.kind in ("float", "log_float"):
                v = float(v)
            elif p.kind == "int":
                if float(v) != int(v):
                    raise ValueError(f"{self.name}: {p.name!r} must be an integer")
                v = int(v)
            elif p.kind == "bool":
                v = bool(v)
            elif p.kind == "choice":
                if v not in p.choices:
                    raise ValueError(f"{self.name}: {p.name!r} must be one of {p.choices}, got {v!r}")
            elif p.kind == "range":
                v = [None if a is None else float(a) for a in v]
                if len(v) != 2:
                    raise ValueError(f"{self.name}: {p.name!r} must be (min, max)")
            elif p.kind == "ranges":
                v = [[float(a), float(b)] for a, b in v]
            elif p.kind == "anchors":
                v = [[float(a), None if b is None else float(b)] for a, b in v]
            if p.kind in ("float", "log_float", "int"):
                if p.min is not None and v < p.min:
                    raise ValueError(f"{self.name}: {p.name!r} must be >= {p.min}, got {v}")
                if p.max is not None and v > p.max:
                    raise ValueError(f"{self.name}: {p.name!r} must be <= {p.max}, got {v}")
            out[p.name] = v
        return out


def _copy(v):
    if isinstance(v, list):
        return [_copy(a) for a in v]
    return v


_REGISTRY: dict[str, OpSpec] = {}


def operation(name: str, version: int, title: str, category: str, params=(), description: str = "",
              flags=()):
    """Decorator registering ``func(spectrum, **params) -> Spectrum``."""

    def deco(func):
        sig_names = list(inspect.signature(func).parameters)[1:]
        declared = [p.name for p in params]
        if sig_names != declared:
            raise RuntimeError(f"operation {name}: signature {sig_names} != declared params {declared}")
        spec = OpSpec(name, version, title, category, tuple(params), func,
                      description or (func.__doc__ or "").strip(), tuple(flags))

        @functools.wraps(func)
        def wrapper(s: Spectrum, **kwargs) -> Spectrum:
            if not isinstance(s, Spectrum):
                raise TypeError(f"{name}: first argument must be a Spectrum")
            p = spec.validate(kwargs)
            out = func(s, **p)
            if spec.flags:
                out = out.with_flags(*spec.flags)
            return out

        wrapper.spec = spec
        _REGISTRY[name] = spec
        return wrapper

    return deco


def get_op(name: str) -> OpSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown operation {name!r}") from None


def list_ops(category: str | None = None) -> list[OpSpec]:
    return [s for s in _REGISTRY.values() if category is None or s.category == category]


def apply_op(name: str, s: Spectrum, params: dict) -> Spectrum:
    spec = get_op(name)
    p = spec.validate(params)
    out = spec.func(s, **p)
    if spec.flags:
        out = out.with_flags(*spec.flags)
    return out


# --------------------------------------------------------------------- warnings
WARNINGS_KEY = "_step_warnings"


def warn(s: Spectrum, message: str) -> Spectrum:
    """Attach a processing warning to the spectrum produced by the current step."""
    meta = dict(s.meta)
    meta[WARNINGS_KEY] = list(meta.get(WARNINGS_KEY, [])) + [message]
    return s.replace(meta=meta)


def pop_warnings(s: Spectrum) -> tuple[Spectrum, list[str]]:
    if WARNINGS_KEY not in s.meta:
        return s, []
    meta = dict(s.meta)
    w = meta.pop(WARNINGS_KEY)
    return s.replace(meta=meta), list(w)


@dataclass
class OpCall:
    """A concrete call; used by the script generator."""
    name: str
    params: dict = field(default_factory=dict)

    def code(self, var: str = "s") -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{var} = ops.{self.name}({var}{', ' if args else ''}{args})"
