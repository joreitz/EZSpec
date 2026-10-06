"""Model components: peaks, baseline terms, classic functions and formulas.

Peak profiles are area-normalised: ``area`` is the integral and directly
reportable; widths are FWHM. Derived quantities (height, total FWHM, ...)
are computed from the fitted parameters and get fully propagated
uncertainties in the fit result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from .. import lineshapes as ls
from .formula import Formula

INF = float("inf")


@dataclass(frozen=True)
class ParamDef:
    name: str
    default: float
    min: float = -INF
    max: float = INF
    label: str = ""


@dataclass(frozen=True)
class ComponentType:
    kind: str
    title: str
    category: str                       # peak | baseline | formula
    params: tuple
    func: Callable                      # func(x, options, **params) -> array
    formula_text: str = ""
    derived: Callable | None = None     # derived(options, **params) -> dict
    linear: bool = False                # linear in all parameters
    description: str = ""


COMPONENT_TYPES: dict[str, ComponentType] = {}


def _register(ct: ComponentType):
    COMPONENT_TYPES[ct.kind] = ct
    return ct


SQRT_2PI = math.sqrt(2 * math.pi)


# --------------------------------------------------------------------------- peaks
def _peak_params(*extra):
    return (ParamDef("area", 1.0, -INF, INF, "Fläche"),
            ParamDef("center", 0.0, -INF, INF, "Zentrum")) + tuple(extra)


def _d_gauss(o, area, center, fwhm):
    sigma = max(fwhm, 1e-300) * ls.FWHM_TO_SIGMA
    return {"area": area, "center": center, "height": area / (sigma * SQRT_2PI), "fwhm": fwhm}


def _d_lorentz(o, area, center, fwhm):
    return {"area": area, "center": center, "height": 2 * area / (math.pi * max(fwhm, 1e-300)),
            "fwhm": fwhm}


def _d_voigt(o, area, center, fwhm_g, fwhm_l):
    return {"area": area, "center": center,
            "height": ls.voigt(center, area, center, fwhm_g, fwhm_l),
            "fwhm": ls.voigt_fwhm_exact(fwhm_g, fwhm_l)}


def _d_pv(o, area, center, fwhm, eta):
    return {"area": area, "center": center, "height": ls.pseudo_voigt(center, area, center, fwhm, eta),
            "fwhm": fwhm}


def _d_tch(o, area, center, fwhm_g, fwhm_l):
    w, eta = ls.tch_width_eta(fwhm_g, fwhm_l)
    return {"area": area, "center": center, "height": ls.pseudo_voigt(center, area, center, w, eta),
            "fwhm": w, "eta": eta}


def _d_p7(o, area, center, fwhm, m):
    return {"area": area, "center": center, "height": ls.pearson7(center, area, center, fwhm, m),
            "fwhm": fwhm}


def _d_emg(o, area, mu, fwhm_g, tau):
    f = lambda t: ls.emg(t, area, mu, fwhm_g, tau)  # noqa: E731
    sig = fwhm_g * ls.FWHM_TO_SIGMA
    lo, hi = mu - 2 * sig, mu + 2 * sig + 3 * tau
    res = minimize_scalar(lambda t: -f(t), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12 * max(1.0, abs(mu)) + 1e-10 * (hi - lo)})
    xm = float(res.x)
    height = f(xm)
    fwhm = ls.numeric_fwhm(f, xm, 0.5 * (fwhm_g + tau))
    return {"area": area, "center": xm, "height": height, "fwhm": fwhm}


_register(ComponentType(
    "gaussian", "Gauß", "peak",
    _peak_params(ParamDef("fwhm", 1.0, 0.0, INF, "FWHM")),
    lambda x, o, area, center, fwhm: ls.gaussian(x, area, center, fwhm),
    "A/(σ√(2π))·exp(−(x−c)²/(2σ²)),  σ = FWHM/(2√(2 ln 2))", _d_gauss))
_register(ComponentType(
    "lorentzian", "Lorentz", "peak",
    _peak_params(ParamDef("fwhm", 1.0, 0.0, INF, "FWHM")),
    lambda x, o, area, center, fwhm: ls.lorentzian(x, area, center, fwhm),
    "A/π · γ/((x−c)² + γ²),  γ = FWHM/2", _d_lorentz))
_register(ComponentType(
    "voigt", "Voigt (exakt)", "peak",
    _peak_params(ParamDef("fwhm_g", 1.0, 0.0, INF, "FWHM Gauß"), ParamDef("fwhm_l", 1.0, 0.0, INF, "FWHM Lorentz")),
    lambda x, o, area, center, fwhm_g, fwhm_l: ls.voigt(x, area, center, fwhm_g, fwhm_l),
    "A·Re[w(z)]/(σ√(2π)),  z = (x−c+iγ)/(σ√2)  (Faddeeva-Funktion)", _d_voigt,
    description="Exakte Faltung von Gauß und Lorentz; Gesamt-FWHM wird numerisch exakt berechnet."))
_register(ComponentType(
    "pseudo_voigt", "Pseudo-Voigt", "peak",
    _peak_params(ParamDef("fwhm", 1.0, 0.0, INF, "FWHM"), ParamDef("eta", 0.5, 0.0, 1.0, "η (Lorentz-Anteil)")),
    lambda x, o, area, center, fwhm, eta: ls.pseudo_voigt(x, area, center, fwhm, eta),
    "η·L(x; A, c, FWHM) + (1−η)·G(x; A, c, FWHM)", _d_pv))
_register(ComponentType(
    "tch_pseudo_voigt", "Pseudo-Voigt (TCH)", "peak",
    _peak_params(ParamDef("fwhm_g", 1.0, 0.0, INF, "FWHM Gauß"), ParamDef("fwhm_l", 1.0, 0.0, INF, "FWHM Lorentz")),
    lambda x, o, area, center, fwhm_g, fwhm_l: ls.tch_pseudo_voigt(x, area, center, fwhm_g, fwhm_l),
    "Pseudo-Voigt mit Γ, η nach Thompson–Cox–Hastings (1987)", _d_tch,
    description="Näherung an den Voigt (max. Abweichung ≈1,3 % der Peakhöhe)."))
_register(ComponentType(
    "pearson7", "Pearson VII", "peak",
    _peak_params(ParamDef("fwhm", 1.0, 0.0, INF, "FWHM"), ParamDef("m", 2.0, 0.501, 1000.0, "Exponent m")),
    lambda x, o, area, center, fwhm, m: ls.pearson7(x, area, center, fwhm, m),
    "A·Γ(m)/(√π Γ(m−½) a)·[1 + ((x−c)/a)²]^(−m),  a = FWHM/(2√(2^(1/m)−1))", _d_p7))
_register(ComponentType(
    "emg", "Exp. mod. Gauß (EMG)", "peak",
    (ParamDef("area", 1.0, -INF, INF, "Fläche"), ParamDef("mu", 0.0, -INF, INF, "μ (Gauß-Zentrum)"),
     ParamDef("fwhm_g", 1.0, 0.0, INF, "FWHM Gauß"), ParamDef("tau", 1.0, 0.0, INF, "τ (Ausläufer)")),
    lambda x, o, area, mu, fwhm_g, tau: ls.emg(x, area, mu, fwhm_g, tau),
    "Gauß(μ, σ) ⊗ exp(−x/τ)/τ  (Ausläufer zu großen x)", _d_emg,
    description="'center' der abgeleiteten Größen ist die Position des Maximums."))


# --------------------------------------------------------------------------- baseline terms
def _poly_names(order):
    return tuple(f"c{k}" for k in range(order + 1))


def _poly_func(x, o, **p):
    x0 = o.get("x0", 0.0)
    t = np.asarray(x, float) - x0
    out = np.zeros_like(t)
    for k in reversed(range(o.get("order", 1) + 1)):
        out = out * t + p[f"c{k}"]
    return out


_register(ComponentType(
    "constant", "Konstante", "baseline", (ParamDef("c", 0.0, label="c"),),
    lambda x, o, c: np.full(np.shape(x), c, float), "c", linear=True))
_register(ComponentType(
    "linear", "Gerade", "baseline",
    (ParamDef("intercept", 0.0, label="Achsenabschnitt bei x0"), ParamDef("slope", 0.0, label="Steigung")),
    lambda x, o, intercept, slope: intercept + slope * (np.asarray(x, float) - o.get("x0", 0.0)),
    "intercept + slope·(x − x0)", linear=True))
_register(ComponentType(
    "polynomial", "Polynom", "baseline", (),     # parameters depend on options["order"]
    _poly_func, "Σ c_k (x − x0)^k", linear=True))


# --------------------------------------------------------------------------- formula
_register(ComponentType(
    "formula", "Formel", "formula", (), None, "frei definierbar"))


# --------------------------------------------------------------------------- instance
@dataclass
class ParamSetting:
    value: float
    min: float = -INF
    max: float = INF
    vary: bool = True
    expr: str | None = None

    def to_dict(self):
        def enc(v):
            return None if v is None else (str(v) if math.isinf(v) else v)
        return {"value": self.value, "min": enc(self.min), "max": enc(self.max), "vary": self.vary,
                "expr": self.expr}

    @classmethod
    def from_dict(cls, d):
        def dec(v, default):
            if v is None:
                return default
            return float(v)
        return cls(float(d["value"]), dec(d.get("min"), -INF), dec(d.get("max"), INF),
                   bool(d.get("vary", True)), d.get("expr") or None)


@dataclass
class Component:
    kind: str
    prefix: str = ""
    options: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)   # local name -> ParamSetting
    label: str = ""

    def __post_init__(self):
        if self.kind not in COMPONENT_TYPES:
            raise ValueError(f"unknown component kind {self.kind!r}")
        self._formula = None
        for name, pd in zip(self.local_names(), self._param_defs()):
            if name not in self.settings:
                self.settings[name] = ParamSetting(pd.default, pd.min, pd.max)
        for name in list(self.settings):
            if name not in self.local_names():
                del self.settings[name]

    @property
    def type(self) -> ComponentType:
        return COMPONENT_TYPES[self.kind]

    @property
    def formula(self) -> Formula | None:
        if self.kind != "formula":
            return None
        if self._formula is None or self._formula.expression != self.options.get("expression", ""):
            self._formula = Formula(self.options.get("expression", ""),
                                    tuple(self.options.get("independent", ("x",))))
        return self._formula

    def _param_defs(self):
        if self.kind == "polynomial":
            return tuple(ParamDef(n, 0.0) for n in _poly_names(int(self.options.get("order", 1))))
        if self.kind == "formula":
            defaults = self.options.get("defaults", {})
            return tuple(ParamDef(n, float(defaults.get(n, 1.0))) for n in self.formula.parameters)
        return self.type.params

    def local_names(self) -> tuple:
        if self.kind == "polynomial":
            return _poly_names(int(self.options.get("order", 1)))
        if self.kind == "formula":
            return self.formula.parameters
        return tuple(p.name for p in self.type.params)

    def full_name(self, local: str) -> str:
        return f"{self.prefix}{local}"

    def full_names(self) -> tuple:
        return tuple(self.full_name(n) for n in self.local_names())

    def evaluate(self, x, values: dict, variables: dict | None = None):
        """Evaluate with a mapping of *full* parameter names to values.

        ``variables`` supplies additional independent variables by name for
        formulas with several predictors; the first declared independent
        variable defaults to ``x``.
        """
        local = {n: values[self.full_name(n)] for n in self.local_names()}
        if self.kind == "formula":
            f = self.formula
            args = []
            for i, name in enumerate(f.independent):
                if variables and name in variables:
                    args.append(variables[name])
                elif i == 0:
                    args.append(x)
                else:
                    raise KeyError(f"unabhängige Variable {name!r} fehlt in den Daten")
            return f(*args, **local)
        return np.asarray(self.type.func(x, self.options, **local), float)

    def derived(self, values: dict) -> dict:
        if self.type.derived is None:
            return {}
        local = {n: values[self.full_name(n)] for n in self.local_names()}
        return self.type.derived(self.options, **local)

    @property
    def is_peak(self) -> bool:
        return self.type.category == "peak"

    @property
    def display_name(self) -> str:
        base = self.label or self.type.title
        return f"{base} [{self.prefix.rstrip('_')}]" if self.prefix else base

    def to_dict(self) -> dict:
        return {"kind": self.kind, "prefix": self.prefix, "options": self.options, "label": self.label,
                "settings": {k: v.to_dict() for k, v in self.settings.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Component":
        return cls(d["kind"], d.get("prefix", ""), dict(d.get("options", {})),
                   {k: ParamSetting.from_dict(v) for k, v in d.get("settings", {}).items()},
                   d.get("label", ""))
