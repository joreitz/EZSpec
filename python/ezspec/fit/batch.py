"""Series fits and global fits with shared parameters.

* :func:`fit_series` fits the same model to many spectra in turn; start
  values can be propagated from one fit to the next (spectra series,
  maps, temperature series).
* :func:`fit_global` fits several spectra *simultaneously*. Parameters named
  in ``shared`` are common to all spectra (e.g. a common line width or a
  common decay time); all others get a ``_d<i>`` suffix. The stacked problem
  goes through the regular fit engine, so it gets the same statistics,
  covariance (including correlations between shared and per-spectrum
  parameters), refinement and warnings as a single fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import lmfit
import numpy as np

from ..models.model import Model
from ..spectrum import SigmaSource, Spectrum
from .engine import FitOptions, fit
from .result import FitResult


# ============================================================================ series
@dataclass
class SeriesResult:
    names: list
    results: list
    errors: list
    index: np.ndarray = field(default_factory=lambda: np.array([]))

    def parameter(self, name: str):
        """(values, standard errors) of one parameter across the series (NaN where the fit failed)."""
        v = np.full(len(self.results), np.nan)
        e = np.full(len(self.results), np.nan)
        for i, r in enumerate(self.results):
            if r is None:
                continue
            if name in r.params:
                v[i] = r.params[name].value
                e[i] = np.nan if r.params[name].stderr is None else r.params[name].stderr
            else:
                comp, _, key = name.partition(".")
                d = r.derived_table().get(comp, {}).get(key)
                if d is not None:
                    v[i] = d[0]
                    e[i] = np.nan if d[1] is None else d[1]
        return v, e

    def parameter_names(self) -> list:
        for r in self.results:
            if r is not None:
                return list(r.params) + [f"{d.component}.{d.name}" for d in r.derived]
        return []

    def rows(self) -> list:
        names = self.parameter_names()
        out = []
        for i, (n, r, err) in enumerate(zip(self.names, self.results, self.errors)):
            row = {"index": float(self.index[i]) if len(self.index) else i, "name": n,
                   "success": bool(r is not None and r.success), "error": err or ""}
            if r is not None:
                st = r.stats
                row.update(redchi=st.redchi, s_res=st.s_res, aicc=st.aicc,
                           n_warnings=sum(w.severity != "info" for w in r.warnings))
            for p in names:
                v, e = self.parameter(p)
                row[p] = v[i]
                row[p + "_stderr"] = e[i]
            out.append(row)
        return out

    def to_csv(self, path, delimiter=","):
        import csv
        rows = self.rows()
        if not rows:
            return
        keys = list(rows[0])
        for r in rows[1:]:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter=delimiter)
            w.writeheader()
            w.writerows(rows)


def fit_series(spectra, model: Model, options: FitOptions | dict | None = None, propagate: bool = True,
               names=None, index=None, progress=None, cancel=None) -> SeriesResult:
    """Fit ``model`` to each spectrum. With ``propagate`` the fitted values of a
    successful fit become the start values of the next one."""
    current = model.copy()
    results, errors = [], []
    names = list(names) if names is not None else [s.meta.get("name", f"#{i}") for i, s in enumerate(spectra)]
    for i, s in enumerate(spectra):
        if cancel is not None and cancel():
            break
        try:
            r = fit(s, current, options)
            results.append(r)
            errors.append(None)
            if propagate and r.success:
                current.apply_values(r.values)
        except Exception as exc:  # noqa: BLE001 - recorded per spectrum
            results.append(None)
            errors.append(f"{type(exc).__name__}: {exc}")
        if progress is not None:
            progress(i + 1, len(spectra))
    idx = np.asarray(index if index is not None else np.arange(len(results)), float)
    return SeriesResult(names[:len(results)], results, errors, idx[:len(results)])


# ============================================================================ global
class _GlobalComponent:
    """Peak component of dataset i viewed through the global parameter names."""

    def __init__(self, gm, i, comp):
        self.gm, self.i, self.comp = gm, i, comp
        self.prefix = f"d{i}.{comp.prefix}" if comp.prefix else f"d{i}."
        self.kind = comp.kind
        self.is_peak = comp.is_peak
        self.display_name = f"{comp.display_name} (#{i})"
        self.type = comp.type
        self.group = i

    def _local(self, values):
        return self.gm.values_for(values, self.i)

    def derived(self, values):
        return self.comp.derived(self._local(values))

    def evaluate(self, x, values, variables=None):
        return self.comp.evaluate(x, self._local(values), variables)


class GlobalModel:
    """Duck-typed :class:`Model` over stacked data (aux variable ``__dataset``)."""

    def __init__(self, models: list, shared: list):
        self.models = [m.copy() for m in models]
        self.shared = list(shared)
        names0 = set(self.models[0].param_names())
        for s in self.shared:
            for m in self.models:
                if s not in m.param_names():
                    raise ValueError(f"shared parameter {s!r} is missing in one of the models")
        self._names0 = names0

    def name(self, base: str, i: int) -> str:
        return base if base in self.shared else f"{base}_d{i}"

    def _rename_expr(self, expr, i, bases):
        if not expr:
            return expr
        return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b",
                      lambda m: self.name(m.group(1), i) if m.group(1) in bases else m.group(1), expr)

    def make_params(self) -> lmfit.Parameters:
        params = lmfit.Parameters()
        exprs = []
        for i, m in enumerate(self.models):
            p_i = m.make_params()
            bases = set(p_i)
            for base, par in p_i.items():
                full = self.name(base, i)
                if full in params:
                    continue
                params.add(full, value=par.value, min=par.min, max=par.max, vary=par.vary and not par.expr)
                if par.expr:
                    exprs.append((full, self._rename_expr(par.expr, i, bases)))
        for full, e in exprs:
            params[full].expr = e
        return params

    def param_names(self) -> list:
        return list(self.make_params())

    def values_for(self, values: dict, i: int) -> dict:
        return {base: values[self.name(base, i)] for base in self.models[i].param_names()}

    @staticmethod
    def values_of(params) -> dict:
        return Model.values_of(params)

    def evaluate(self, x, params, variables=None) -> np.ndarray:
        values = self.values_of(params)
        x = np.asarray(x, float)
        idx = np.asarray(variables["__dataset"]).astype(int)
        out = np.empty_like(x)
        for i, m in enumerate(self.models):
            sel = idx == i
            if not sel.any():
                continue
            sub = {k: np.asarray(v)[sel] for k, v in variables.items() if k != "__dataset"}
            out[sel] = m.evaluate(x[sel], self.values_for(values, i), sub)
        return out

    def evaluate_components(self, x, params, variables=None) -> dict:
        values = self.values_of(params)
        idx = np.asarray(variables["__dataset"]).astype(int)
        out = {}
        for i, m in enumerate(self.models):
            sel = idx == i
            for c in m.components:
                arr = np.zeros(len(x))
                arr[sel] = c.evaluate(np.asarray(x)[sel], self.values_for(values, i))
                out[f"d{i}.{c.prefix or c.display_name}"] = arr
        return out

    @property
    def peaks(self) -> list:
        return [_GlobalComponent(self, i, c) for i, m in enumerate(self.models) for c in m.peaks]

    @property
    def independent_variables(self) -> list:
        return ["__dataset"]

    def describe(self) -> str:
        lines = [f"Global fit over {len(self.models)} spectra; shared: {', '.join(self.shared) or '—'}"]
        for i, m in enumerate(self.models):
            lines += [f"#{i}: " + line for line in m.describe().splitlines()]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"format": "ezspec.global_model", "version": 1, "shared": self.shared,
                "models": [m.to_dict() for m in self.models]}

    @classmethod
    def from_dict(cls, d) -> "GlobalModel":
        return cls([Model.from_dict(m) for m in d["models"]], d.get("shared", []))

    def copy(self) -> "GlobalModel":
        return GlobalModel.from_dict(self.to_dict())

    def check(self):
        for m in self.models:
            m.check()


def stack_spectra(spectra) -> Spectrum:
    """Concatenate spectra (fit masks applied) with an aux variable ``__dataset``."""
    xs, ys, sig, idx = [], [], [], []
    sources = {s.sigma_source for s in spectra}
    has_sigma = [s.sigma is not None for s in spectra]
    if any(has_sigma) and not all(has_sigma):
        raise ValueError("either all spectra or none must have σ")
    if len(sources) > 1:
        raise ValueError(f"different σ sources: {sorted(s.value for s in sources)}")
    flags = set()
    for i, s in enumerate(spectra):
        m = s.fit_mask
        xs.append(s.x[m])
        ys.append(s.y[m])
        if s.sigma is not None:
            sig.append(s.sigma[m])
        idx.append(np.full(int(m.sum()), i, float))
        flags |= set(s.flags)
    return Spectrum(np.concatenate(xs), np.concatenate(ys), np.concatenate(sig) if sig else None,
                    sources.pop() if sig else SigmaSource.UNKNOWN, spectra[0].x_unit, spectra[0].y_unit,
                    spectra[0].x_label, spectra[0].y_label, None, {"name": "global"}, frozenset(flags),
                    {"var:__dataset": np.concatenate(idx)})


def fit_global(spectra, models, shared, options: FitOptions | dict | None = None) -> FitResult:
    """Simultaneous fit; ``models`` is one model per spectrum or a single template."""
    if isinstance(models, Model):
        models = [models.copy() for _ in spectra]
    if len(models) != len(spectra):
        raise ValueError("one model per spectrum required")
    gm = GlobalModel(models, shared)
    opt = options if isinstance(options, FitOptions) else FitOptions.from_dict(options)
    if opt.x_range:
        lo, hi = opt.x_range
        spectra = [s.replace(exclude=(s.exclude if s.exclude is not None else np.zeros(s.n, bool))
                             | (s.x < lo) | (s.x > hi)) for s in spectra]
        opt = FitOptions.from_dict({**opt.to_dict(), "x_range": None})
    stacked = stack_spectra(spectra)
    r = fit(stacked, gm, opt)
    r.extra["global"] = {"n_spectra": len(spectra), "shared": list(shared),
                         "points_per_spectrum": [int(s.fit_mask.sum()) for s in spectra]}
    return r


def split_global(result: FitResult, i: int) -> dict:
    """Curves of spectrum ``i`` from a global fit result."""
    idx = result._internals["variables"]["__dataset"].astype(int)
    sel = idx == i
    comps = {k.split(".", 1)[1]: v[sel] for k, v in result.components.items() if k.startswith(f"d{i}.")}
    return {"x": result.x[sel], "y": result.y[sel], "best_fit": result.best_fit[sel],
            "residuals": result.residuals[sel], "normalized_residuals": result.normalized_residuals[sel],
            "components": comps}
