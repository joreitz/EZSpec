"""Baseline operations.

A baseline is stored as a *recipe* (method + parameters + anchors + masks),
never as a bare array: it is recomputed from the step input, so it stays
valid after re-cropping and can be applied to other spectra.

Each operation stores the baseline of this step in ``aux["baseline"]`` and the
sum of all baselines subtracted so far in ``aux["baseline_total"]``.

Whittaker parameter ``lam`` is in the pybaselines convention (per point).
Its effect depends on the sampling density: roughly, structures with periods
longer than ``2*pi*lam**(1/(2d))`` points are kept in the baseline
(:func:`whittaker_cutoff_points`). ``lam`` is therefore not transferable
between spectra of different point density.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from .._backend import as_f64, core
from ..spectrum import Spectrum
from .basic import _in_ranges
from .registry import ParamSpec as P
from .registry import operation, warn

_SUBTRACT = P("subtract", "bool", True, "abziehen", help="False: Baseline nur berechnen/anzeigen")


def whittaker_cutoff_points(lam: float, diff_order: int = 2) -> float:
    """Period (in points) at which the Whittaker smoother passes half amplitude.

    The transfer function is 1 / (1 + lam (2 sin(w/2))^(2d)); it equals 1/2 at
    2 sin(w/2) = lam^(-1/(2d)), i.e. period = 2 pi / w.
    """
    s = lam ** (-1.0 / (2 * diff_order)) / 2.0
    if s >= 1.0:
        return 2.0
    return 2.0 * np.pi / (2.0 * np.arcsin(s))


def _finish(s: Spectrum, b: np.ndarray, subtract: bool, recipe: dict) -> Spectrum:
    aux = dict(s.aux)
    aux["baseline"] = b
    total = aux.get("baseline_total", np.zeros(s.n))
    aux["baseline_total"] = total + b if subtract else total
    if subtract and "smoothed" in aux:          # display smoothing lives on the y scale
        aux["smoothed"] = aux["smoothed"] - b
    meta = dict(s.meta)
    meta.setdefault("baselines", [])
    meta["baselines"] = list(meta["baselines"]) + [recipe]
    out = s.replace(y=s.y - b if subtract else s.y, aux=aux, meta=meta)
    if subtract:
        out = out.with_flags("baseline_subtracted")
        out = _check_overcorrection(s, out, b)
    return out


def _check_overcorrection(s_in: Spectrum, out: Spectrum, b: np.ndarray) -> Spectrum:
    """Warn if the baseline lies more than 3 sigma above the data at many points
    (the baseline cuts into broad bands)."""
    sig = s_in.sigma if s_in.sigma is not None else core().der_snr(as_f64(s_in.y))
    if sig is None or not np.all(np.isfinite(sig)):
        return out
    r = s_in.y - b
    frac = float(np.mean(r < -3.0 * np.asarray(sig)))
    if frac > 0.02:
        out = warn(out, f"Baseline liegt an {100 * frac:.1f} % der Punkte > 3σ über den Daten – "
                        "Überkorrektur? (schneidet in breite Banden)")
    return out


def _fixed_weights(x, exclude_ranges, force_ranges):
    if not exclude_ranges and not force_ranges:
        return None
    fixed = np.full(len(x), np.nan)
    fixed[_in_ranges(x, force_ranges)] = 1.0
    fixed[_in_ranges(x, exclude_ranges)] = 0.0
    return fixed


# ============================================================================ anchors
def anchor_values(s: Spectrum, anchors, window: int):
    """Effective anchor ordinates: given y, or the median of the data within
    +/- ``window`` points of the nearest data point (window = 0: snap)."""
    xa, ya = [], []
    for ax, ay in sorted(anchors, key=lambda a: a[0]):
        if ay is None:
            i = int(np.argmin(np.abs(s.x - ax)))
            lo, hi = max(0, i - window), min(s.n, i + window + 1)
            ay = float(np.median(s.y[lo:hi]))
        if xa and ax == xa[-1]:
            continue
        xa.append(float(ax))
        ya.append(float(ay))
    return np.array(xa), np.array(ya)


@operation("baseline_anchors", 1, "Baseline: Ankerpunkte", "Baseline",
           params=[P("anchors", "anchors", [], "Anker [(x, y|None), ...]"),
                   P("interpolation", "choice", "pchip", "Interpolation",
                     choices=("pchip", "akima", "cubic", "linear"),
                     help="pchip/akima: formerhaltend; cubic: natürlicher Spline (kann überschwingen)"),
                   P("window", "int", 3, "Median-Fenster ± Punkte (0 = Snap)", min=0),
                   P("extrapolation", "choice", "constant", "außerhalb der Anker",
                     choices=("constant", "linear")),
                   _SUBTRACT])
def baseline_anchors(s: Spectrum, anchors, interpolation, window, extrapolation, subtract) -> Spectrum:
    """Baseline through user-placed anchor points."""
    xa, ya = anchor_values(s, anchors, window)
    if len(xa) < 2:
        raise ValueError("need at least 2 anchor points")
    xq = as_f64(s.x)
    if interpolation == "pchip":
        b = core().pchip(xa, ya, xq, extrapolation)
    elif interpolation == "cubic":
        b = core().natural_cubic(xa, ya, xq, extrapolation)
    elif interpolation == "linear":
        b = core().linear_interp(xa, ya, xq, extrapolation)
    else:
        if len(xa) < 3:
            b = core().linear_interp(xa, ya, xq, extrapolation)
        else:
            ak = Akima1DInterpolator(xa, ya)
            b = ak(xq, extrapolate=True)
            lo, hi = xq < xa[0], xq > xa[-1]
            if extrapolation == "constant":
                b[lo], b[hi] = ya[0], ya[-1]
            else:
                d = ak.derivative()
                b[lo] = ya[0] + d(xa[0]) * (xq[lo] - xa[0])
                b[hi] = ya[-1] + d(xa[-1]) * (xq[hi] - xa[-1])
    recipe = {"method": "anchors", "anchors": [[float(a), float(c)] for a, c in zip(xa, ya)],
              "interpolation": interpolation}
    out = _finish(s, np.asarray(b), subtract, recipe)
    if xa[0] > s.x[0] or xa[-1] < s.x[-1]:
        out = warn(out, f"Baseline außerhalb der Anker ({extrapolation}) fortgesetzt – "
                        "an den Rändern schlecht bestimmt.")
    return out


# ============================================================================ polynomial
def _vander(x, order):
    lo, hi = x.min(), x.max()
    t = (2 * x - (lo + hi)) / (hi - lo) if hi > lo else np.zeros_like(x)
    return np.vander(t, order + 1, increasing=True)


def polynomial_baseline(x, y, order, weights=None, method="fit", tol=1e-3, max_iter=250, num_std=1.0):
    """Polynomial baselines; ``modpoly``/``imodpoly`` follow pybaselines 1.2.1
    (Lieber & Mahadevan-Jansen 2003; Zhao et al. 2007). x is mapped to [-1, 1]."""
    V = _vander(x, order)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float).copy()
    sw = np.sqrt(w)
    pinv = np.linalg.pinv(sw[:, None] * V)
    yw = y.copy()
    coef = pinv @ (sw * yw)
    base = V @ coef
    n_iter = 0
    if method == "fit":
        return base, n_iter
    if method == "modpoly":
        for n_iter in range(1, max_iter + 1):
            old = base
            yw = np.minimum(yw, base)
            base = V @ (pinv @ (sw * yw))
            if np.linalg.norm(base - old) / max(np.linalg.norm(old), np.finfo(float).eps) < tol:
                break
        return base, n_iter
    # imodpoly
    dev = np.std(yw - base)
    w[base + dev < yw] = 0
    sw = np.sqrt(w)
    pinv = np.linalg.pinv(sw[:, None] * V)
    for n_iter in range(1, max_iter + 1):
        yw = np.minimum(yw, base + num_std * dev)
        base = V @ (pinv @ (sw * yw))
        new_dev = np.std(yw - base)
        if abs(dev - new_dev) / max(abs(new_dev), np.finfo(float).eps) < tol:
            break
        dev = new_dev
    return base, n_iter


@operation("baseline_polynomial", 1, "Baseline: Polynom", "Baseline",
           params=[P("order", "int", 2, "Grad", min=0, max=12),
                   P("method", "choice", "fit", "Verfahren", choices=("fit", "modpoly", "imodpoly"),
                     help="fit: LS-Fit (nur in 'ranges'); modpoly/imodpoly: iterativ unter die Daten"),
                   P("ranges", "ranges", [], "Baseline-Bereiche (leer = alle)"),
                   _SUBTRACT])
def baseline_polynomial(s: Spectrum, order, method, ranges, subtract) -> Spectrum:
    """Polynomial baseline, fitted to marked baseline regions or iteratively."""
    w = None
    if ranges:
        w = _in_ranges(s.x, ranges).astype(float)
        if w.sum() < order + 1:
            raise ValueError(f"need at least {order + 1} points in the baseline ranges")
    b, n_iter = polynomial_baseline(s.x, s.y, order, w, method)
    return _finish(s, b, subtract, {"method": "polynomial", "order": order, "variant": method,
                                    "iterations": n_iter})


# ============================================================================ Whittaker
_WH_PARAMS = [P("diff_order", "int", 2, "Differenzenordnung", min=1, max=3),
              P("max_iter", "int", 50, "max. Iterationen", min=0),
              P("tol", "float", 1e-3, "Toleranz", min=0.0),
              P("exclude_ranges", "ranges", [], "ausgeschlossene Bereiche (w = 0)"),
              P("force_ranges", "ranges", [], "erzwungene Baseline-Bereiche (w = 1)"),
              _SUBTRACT]


def _whittaker_finish(s, res, subtract, recipe, lam, d):
    b, w, hist, converged = res
    recipe.update(iterations=int(len(hist)), converged=bool(converged),
                  cutoff_points=whittaker_cutoff_points(lam, d))
    out = _finish(s, np.asarray(b), subtract, recipe)
    aux = dict(out.aux)
    aux["baseline_weights"] = np.asarray(w)
    out = out.replace(aux=aux)
    if not converged:
        out = warn(out, f"{recipe['method']}: nicht konvergiert nach {len(hist)} Iterationen.")
    return out


@operation("baseline_asls", 1, "Baseline: AsLS (Whittaker)", "Baseline",
           params=[P("lam", "log_float", 1e6, "λ (Glattheit)", min=0.0),
                   P("p", "float", 1e-2, "Asymmetrie p", min=1e-6, max=0.5)] + _WH_PARAMS)
def baseline_asls(s: Spectrum, lam, p, diff_order, max_iter, tol, exclude_ranges, force_ranges,
                  subtract) -> Spectrum:
    """Asymmetric least squares (Eilers & Boelens 2005); weights p above, 1-p below the baseline."""
    fixed = _fixed_weights(s.x, exclude_ranges, force_ranges)
    res = core().asls(as_f64(s.y), lam, p, diff_order, max_iter, tol, None,
                      None if fixed is None else as_f64(fixed))
    return _whittaker_finish(s, res, subtract, {"method": "asls", "lam": lam, "p": p}, lam, diff_order)


@operation("baseline_arpls", 1, "Baseline: arPLS (Whittaker)", "Baseline",
           params=[P("lam", "log_float", 1e5, "λ (Glattheit)", min=0.0)] + _WH_PARAMS)
def baseline_arpls(s: Spectrum, lam, diff_order, max_iter, tol, exclude_ranges, force_ranges,
                   subtract) -> Spectrum:
    """Asymmetrically reweighted penalized least squares (Baek et al. 2015)."""
    fixed = _fixed_weights(s.x, exclude_ranges, force_ranges)
    res = core().arpls(as_f64(s.y), lam, diff_order, max_iter, tol, None,
                       None if fixed is None else as_f64(fixed))
    return _whittaker_finish(s, res, subtract, {"method": "arpls", "lam": lam}, lam, diff_order)


# ============================================================================ SNIP / rubberband
def _lls(y):
    return np.log(np.log(np.sqrt(y + 1.0) + 1.0) + 1.0)


def _lls_inv(v):
    return (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0


@operation("baseline_snip", 1, "Baseline: SNIP", "Baseline",
           params=[P("max_half_window", "int", 20, "max. Halbfenster (Punkte)", min=1),
                   P("decreasing", "bool", False, "abnehmende Fenster"),
                   P("filter_order", "choice", 2, "Filterordnung", choices=(2, 4, 6, 8)),
                   P("lls", "bool", False, "LLS-Transformation (Zähldaten)"),
                   _SUBTRACT])
def baseline_snip(s: Spectrum, max_half_window, decreasing, filter_order, lls, subtract) -> Spectrum:
    """Statistics-sensitive non-linear iterative peak clipping (Ryan 1988, Morháč 1997).
    Choose the half window about the half width of the broadest peak."""
    y = as_f64(s.y)
    if lls:
        if np.min(y) < 0:
            raise ValueError("LLS transformation needs y >= 0")
        b = _lls_inv(core().snip(as_f64(_lls(y)), max_half_window, decreasing, filter_order))
    else:
        b = core().snip(y, max_half_window, decreasing, filter_order)
    return _finish(s, np.asarray(b), subtract,
                   {"method": "snip", "max_half_window": max_half_window, "filter_order": filter_order})


@operation("baseline_rubberband", 1, "Baseline: Rubberband", "Baseline",
           params=[_SUBTRACT])
def baseline_rubberband(s: Spectrum, subtract) -> Spectrum:
    """Lower convex hull of the data (rubberband), linearly interpolated."""
    if not s.is_strictly_increasing:
        raise ValueError("rubberband needs strictly increasing x")
    b = core().rubberband(as_f64(s.x), as_f64(s.y))
    return _finish(s, np.asarray(b), subtract, {"method": "rubberband"})
