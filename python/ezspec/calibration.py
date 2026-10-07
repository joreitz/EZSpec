"""Fitted models as calibrations, e.g. a laser's wavelength λ(I, T).

A fit of y = f(x, v₁, …) with additional independent variables (aux columns
``var:<name>``) can be turned into a *calibration*: a plain-JSON description
of the model, its best-fit values and their covariance. Evaluated with the
extra variables held fixed (e.g. T = 25 °C) it is a 1-D mapping x → y, the
"slice" of the surface, which :func:`ezspec.ops.calibrate_x` applies to the
x axis of other datasets.

Uncertainty
-----------
``sigma`` is the delta-method standard uncertainty of the *mean* calibration
curve, σ_cal(x)² = J C Jᵀ with J = ∂f/∂θ at (x, v) and C the fit covariance.
It is a systematic uncertainty: the same parameter error shifts every point
of a calibrated axis coherently, so it must not be used as an independent
per-point σ_x. It does not contain the scatter of single calibration points
about the model (reported separately as the residual RMS) nor any model
error; a lack of fit shows up in the calibration fit's own statistics.
Uncertainties of the fixed variables (e.g. the temperature set point) are
added as (∂f/∂v · σ_v)².

Extrapolation
-------------
Polynomial surfaces extrapolate poorly. Points outside the convex hull of
the calibration points (in the coordinates normalised by their ranges) are
counted; for a single variable the x range is used.
"""

from __future__ import annotations

import re

import numpy as np

from .models import Model

FORMAT = "ezspec.calibration"
MAX_POINTS = 5000
_EPS = np.finfo(float).eps


class CalibrationError(ValueError):
    pass


def calibration_from_fit(result, spectrum, name: str = "") -> dict:
    """Plain-JSON calibration from a fit ``result`` of ``spectrum``."""
    if result.covariance is None:
        raise CalibrationError("the fit has no covariance matrix – calibration uncertainty cannot be determined")
    model = result._internals.get("model") or Model.from_dict(result.model_spec)
    variables = list(model.independent_variables)
    mask = np.asarray(result.mask, bool)
    cols = [np.asarray(spectrum.x, float)[mask]]
    for v in variables:
        key = f"var:{v}"
        if key not in spectrum.aux:
            raise CalibrationError(f"variable {v!r} missing from the dataset")
        cols.append(np.asarray(spectrum.aux[key], float)[mask])
    pts = np.column_stack(cols)
    if len(pts) > MAX_POINTS:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_POINTS).astype(int)]
    domain = {"x": [float(cols[0].min()), float(cols[0].max())]}
    for v, c in zip(variables, cols[1:]):
        domain[v] = [float(c.min()), float(c.max())]
    st = result.stats
    return {
        "format": FORMAT, "version": 1,
        "source": name or spectrum.meta.get("name", ""),
        "model": result.model_spec,
        "values": {k: float(v) for k, v in result.values.items()},
        "var_names": list(result.var_names),
        "covariance": np.asarray(result.covariance, float).tolist(),
        "covariance_mode": st.covariance_mode,
        "variables": variables,
        "domain": domain,
        "points": pts.tolist(),
        "n_points": int(mask.sum()),
        "rms_residual": float(st.rmse),
        "redchi": None if st.redchi is None else float(st.redchi),
        "x_label": spectrum.x_label, "x_unit": spectrum.x_unit,
        "y_label": spectrum.y_label, "y_unit": spectrum.y_unit,
        "expression": model.describe(),
    }


def parse_assignments(text: str) -> dict:
    """'T=25; p = 1.5e3' -> {'T': 25.0, 'p': 1500.0} (decimal comma accepted)."""
    out = {}
    text = (text or "").strip()
    if not text:
        return out
    for m in re.finditer(r"([A-Za-z_]\w*)\s*=\s*([-+]?(?:\d+[.,]?\d*|[.,]\d+)(?:[eE][-+]?\d+)?)", text):
        out[m.group(1)] = float(m.group(2).replace(",", "."))
    rest = re.sub(r"([A-Za-z_]\w*)\s*=\s*([-+]?(?:\d+[.,]?\d*|[.,]\d+)(?:[eE][-+]?\d+)?)", "", text)
    if rest.strip(" ;,\t"):
        raise CalibrationError(f"not understood: {rest.strip(' ;,')!r} (format: T=25; p=1.0)")
    return out


class Calibration:
    """Evaluate a calibration dict; ``values`` maps variable names to scalars or arrays."""

    def __init__(self, d: dict):
        if not d or d.get("format") != FORMAT:
            raise CalibrationError("not a valid calibration (adopt a fit as calibration first)")
        self.d = d
        self.model = Model.from_dict(d["model"])
        self.model.apply_values(d["values"])
        self.params = self.model.make_params()
        self.var_names = list(d["var_names"])
        cov = d.get("covariance")
        self.cov = None if cov is None else np.asarray(cov, float)
        self.variables = list(d.get("variables", []))
        self.domain = d.get("domain", {})
        self._hull = None

    # ------------------------------------------------------------------ evaluation
    def _vars(self, x, values):
        out = {}
        for v in self.variables:
            if v not in values:
                raise CalibrationError(f"value for {v!r} missing")
            out[v] = np.broadcast_to(np.asarray(values[v], float), np.shape(x)).astype(float)
        return out

    def evaluate(self, x, values=None, theta=None) -> np.ndarray:
        x = np.asarray(x, float)
        p = self.params if theta is None else self._setter()(theta)
        return np.asarray(self.model.evaluate(x, p, self._vars(x, values or {})), float)

    def _setter(self):
        from .fit.engine import _make_setter
        return _make_setter(self.params, self.var_names)

    def _deriv(self, fun, z, scale):
        """d fun / dz element-wise by central differences with a Richardson step."""
        h = _EPS ** (1 / 3) * scale

        def d(hh):
            return (fun(z + hh) - fun(z - hh)) / (2 * hh)

        return (4 * d(0.5 * h) - d(h)) / 3

    def _scale(self, name, z):
        """Finite-difference scale: the calibrated range of the coordinate."""
        lo, hi = self.domain.get(name, (None, None))
        if lo is not None and hi is not None and hi > lo:
            return np.full(np.shape(z), float(hi - lo))
        return np.maximum(np.abs(z), 1.0)

    def slope(self, x, values=None) -> np.ndarray:
        """∂f/∂x."""
        x = np.asarray(x, float)
        vals = values or {}
        return self._deriv(lambda z: self.evaluate(z, vals), x, self._scale("x", x))

    def partial(self, name, x, values) -> np.ndarray:
        """∂f/∂v for the extra variable ``name``."""
        x = np.asarray(x, float)
        v0 = np.broadcast_to(np.asarray(values[name], float), x.shape).astype(float)

        def f(z):
            return self.evaluate(x, {**values, name: z})

        return self._deriv(f, v0, self._scale(name, v0))

    def sigma(self, x, values=None) -> np.ndarray | None:
        """Standard uncertainty of the mean calibration curve (delta method)."""
        if self.cov is None or not self.var_names:
            return None
        from .fit.engine import _jacobian, _scales
        x = np.asarray(x, float)
        theta = np.array([self.params[n].value for n in self.var_names], float)
        se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        setter = self._setter()
        J = _jacobian(lambda t: self.model.evaluate(x, setter(t), self._vars(x, values or {})), theta,
                      _scales(self.params, self.var_names, se))
        var = np.einsum("ij,jk,ik->i", J, self.cov, J)
        return np.sqrt(np.clip(var, 0, None))

    # ------------------------------------------------------------------ domain
    def inside(self, x, values=None) -> np.ndarray:
        """True where (x, v…) lies inside the calibrated region."""
        x = np.asarray(x, float)
        names = ["x"] + self.variables
        q = np.column_stack([x] + [np.broadcast_to(np.asarray((values or {})[v], float), x.shape)
                                   for v in self.variables])
        lo = np.array([self.domain[n][0] for n in names], float)
        hi = np.array([self.domain[n][1] for n in names], float)
        span = np.where(hi > lo, hi - lo, 1.0)
        tol = 1e-9
        in_box = np.all((q >= lo - tol * span) & (q <= hi + tol * span), axis=1)
        if not self.variables:
            return in_box
        hull = self._hull_tri(lo, span)
        if hull is None:
            return in_box
        return in_box & (hull.find_simplex((q - lo) / span, tol=1e-9) >= 0)

    def _hull_tri(self, lo, span):
        if self._hull is None:
            pts = np.asarray(self.d.get("points") or [], float)
            self._hull = False
            if pts.ndim == 2 and len(pts) > pts.shape[1]:
                try:
                    from scipy.spatial import Delaunay
                    self._hull = Delaunay((pts - lo) / span)
                except Exception:  # noqa: BLE001 - degenerate (e.g. a single T): fall back to the box
                    self._hull = False
        return self._hull or None

    # ------------------------------------------------------------------ description
    def describe_slice(self, values: dict, x=None) -> str:
        """Human-readable local linearisation of the slice at the centre of ``x``."""
        if x is None:
            x = np.mean(self.domain.get("x", [0.0, 1.0]))
        x0 = float(np.median(np.atleast_1d(x)))
        f0 = float(self.evaluate(np.array([x0]), values)[0])
        k = float(self.slope(np.array([x0]), values)[0])
        s = self.sigma(np.array([x0]), values)
        unc = f" ± {s[0]:.3g}" if s is not None else ""
        fixed = ", ".join(f"{k_}={v:g}" for k_, v in values.items() if np.ndim(v) == 0)
        return (f"f({x0:.6g}{'; ' + fixed if fixed else ''}) = {f0:.8g}{unc}; "
                f"slope df/dx = {k:.6g}")

    def __repr__(self):
        return f"Calibration({self.d.get('source', '')!r}, variables={self.variables})"

