"""Composite model = sum of components, with lmfit parameter handling."""

from __future__ import annotations

import math

import lmfit
import numpy as np

from .components import COMPONENT_TYPES, Component, ParamSetting


class ModelError(ValueError):
    pass


class Model:
    def __init__(self, components: list | None = None):
        self.components: list[Component] = list(components or [])

    # ------------------------------------------------------------------ building
    def add(self, kind: str, prefix: str | None = None, options: dict | None = None,
            label: str = "", **values) -> Component:
        """Add a component; ``values`` set initial values of local parameters."""
        if prefix is None:
            prefix = self._auto_prefix(kind)
        comp = Component(kind, prefix, dict(options or {}), {}, label)
        for k, v in values.items():
            if k not in comp.settings:
                raise ModelError(f"{kind} has no parameter {k!r} (has {comp.local_names()})")
            comp.settings[k].value = float(v)
        self.components.append(comp)
        self.check()
        return comp

    def _auto_prefix(self, kind: str) -> str:
        if kind == "formula":
            return ""
        cat = COMPONENT_TYPES[kind].category
        stem = "p" if cat == "peak" else "bg"
        used = {c.prefix for c in self.components}
        i = 1
        while f"{stem}{i}_" in used:
            i += 1
        return f"{stem}{i}_"

    def remove(self, prefix_or_index) -> Component:
        if isinstance(prefix_or_index, int):
            return self.components.pop(prefix_or_index)
        for i, c in enumerate(self.components):
            if c.prefix == prefix_or_index:
                return self.components.pop(i)
        raise KeyError(prefix_or_index)

    def component(self, prefix: str) -> Component:
        for c in self.components:
            if c.prefix == prefix:
                return c
        raise KeyError(prefix)

    def check(self) -> None:
        seen = {}
        for c in self.components:
            for n in c.full_names():
                if n in seen:
                    raise ModelError(f"Parametername {n!r} kommt in zwei Komponenten vor "
                                     f"({seen[n]} und {c.display_name}); Präfix ändern")
                seen[n] = c.display_name

    def param_names(self) -> list:
        return [n for c in self.components for n in c.full_names()]

    def setting(self, full_name: str) -> ParamSetting:
        for c in self.components:
            for local in c.local_names():
                if c.full_name(local) == full_name:
                    return c.settings[local]
        raise KeyError(full_name)

    def make_params(self) -> lmfit.Parameters:
        self.check()
        if not self.components:
            raise ModelError("das Modell hat keine Komponenten")
        params = lmfit.Parameters()
        exprs = []
        for c in self.components:
            for local in c.local_names():
                s = c.settings[local]
                name = c.full_name(local)
                value = s.value
                if not s.expr and not (s.min <= value <= s.max):
                    raise ModelError(f"Startwert von {name} ({value}) liegt außerhalb der Grenzen "
                                     f"[{s.min}, {s.max}]")
                params.add(name, value=value, min=s.min, max=s.max, vary=s.vary and not s.expr)
                if s.expr:
                    exprs.append((name, s.expr))
        for name, expr in exprs:
            try:
                params[name].expr = expr
            except (NameError, SyntaxError) as exc:
                raise ModelError(f"Ausdruck für {name} ungültig: {exc}") from None
        return params

    def apply_values(self, values: dict) -> None:
        """Write fitted values back as new start values."""
        for c in self.components:
            for local in c.local_names():
                n = c.full_name(local)
                if n in values:
                    c.settings[local].value = float(values[n])

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def values_of(params) -> dict:
        if isinstance(params, lmfit.Parameters):
            return {k: p.value for k, p in params.items()}
        return dict(params)

    def evaluate(self, x, params, variables: dict | None = None) -> np.ndarray:
        v = self.values_of(params)
        x = np.asarray(x, float)
        total = np.zeros_like(x)
        for c in self.components:
            total = total + c.evaluate(x, v, variables)
        return total

    def evaluate_components(self, x, params, variables: dict | None = None) -> dict:
        v = self.values_of(params)
        x = np.asarray(x, float)
        return {c.prefix or c.display_name: c.evaluate(x, v, variables) for c in self.components}

    @property
    def independent_variables(self) -> list:
        """Names of additional independent variables required by formulas."""
        out = []
        for c in self.components:
            if c.kind == "formula":
                for name in c.formula.independent[1:]:
                    if name not in out:
                        out.append(name)
        return out

    def initial_values(self) -> dict:
        return self.values_of(self.make_params())

    # ------------------------------------------------------------------ description
    def describe(self) -> str:
        lines = []
        for c in self.components:
            if c.kind == "formula":
                lines.append(f"{c.display_name}: {c.options.get('expression')}")
            else:
                opt = ""
                if c.kind in ("polynomial", "linear") and c.options.get("x0"):
                    opt = f" (x0 = {c.options['x0']:g})"
                if c.kind == "polynomial":
                    opt = f" Grad {c.options.get('order', 1)}" + opt
                lines.append(f"{c.display_name}{opt}: {c.type.formula_text}")
        return "\n".join(lines)

    @property
    def peaks(self) -> list:
        return [c for c in self.components if c.is_peak]

    def to_dict(self) -> dict:
        return {"format": "ezspec.model", "version": 1,
                "components": [c.to_dict() for c in self.components]}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls([Component.from_dict(c) for c in d.get("components", [])])

    def copy(self) -> "Model":
        return Model.from_dict(self.to_dict())

    def __repr__(self):
        return f"Model({[c.display_name for c in self.components]})"


def is_finite_bound(v) -> bool:
    return v is not None and not math.isinf(v)
