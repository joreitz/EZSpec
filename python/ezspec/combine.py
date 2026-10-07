"""Combine several datasets into a new one by formulas (ratio, difference, parametric plots).

Example: two quantities recorded simultaneously, ``a`` and ``b``; plot ``c``
against the ratio ``a/b``::

    s = combine({"a": s1, "b": s2, "c": s3}, x_expr="a/b", y_expr="c")

Names usable in the expressions: every alias (= y values of that dataset on
the common grid), ``x`` (the common x values) and ``x_<alias>`` (the original
x values of that dataset, meaningful when aligning by index).

Alignment of the input datasets (``mode``):

* ``"auto"`` – identical x grids are used as they are; datasets of equal
  length whose x values differ by less than half a sampling step are matched
  point by point (recorded simultaneously); otherwise the datasets are
  interpolated onto the grid of the reference dataset within the common range.
* ``"exact"`` – require identical x grids.
* ``"index"`` – match point by point (equal length required); for data
  recorded simultaneously with slightly different abscissae.
* ``"interpolate"`` – linear interpolation onto the reference grid. This
  correlates neighbouring points; the result is flagged ``interpolated`` and
  later fits warn about it.

Uncertainties are propagated to first order (delta method) assuming
independent errors of the inputs: sigma_f^2 = sum_k (df/dy_k)^2 sigma_k^2,
with the derivatives taken per point from the full expression, so an input
used several times in one expression is treated correctly. If the new x
depends on uncertain data, its uncertainty is stored in ``aux["sigma_x"]``
and the dataset is flagged ``x_uncertain`` (ordinary least squares ignores
x errors; fits warn about it). The x and y results are correlated when they
share inputs; this correlation is not carried along.
"""

from __future__ import annotations

import keyword
import re

import numpy as np

from .models.formula import CONSTANTS, FUNCTIONS, Formula, FormulaError
from .spectrum import SigmaSource, Spectrum

_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


class CombineError(ValueError):
    pass


def default_aliases(n: int) -> list:
    return [chr(ord("a") + i) for i in range(n)] if n <= 26 else [f"d{i}" for i in range(n)]


def _check_alias(a: str):
    if not _ALIAS.match(a) or keyword.iskeyword(a) or a in FUNCTIONS or a in CONSTANTS or a == "x" \
            or a.startswith("x_"):
        raise CombineError(f"invalid alias {a!r} (letters/digits, not 'x', not a function name)")


def diagnose_alignment(spectra: list) -> dict:
    """What 'auto' would do, with a human-readable explanation."""
    ref = spectra[0]
    if all(s.n == ref.n and np.array_equal(s.x, ref.x) for s in spectra[1:]):
        return {"mode": "exact", "text": "identical x grid – points are combined directly"}
    if all(s.n == ref.n for s in spectra[1:]):
        dx = abs(ref.dx_median) if ref.n > 1 else np.inf
        dev = max(float(np.max(np.abs(s.x - ref.x))) for s in spectra[1:])
        if np.isfinite(dx) and dev < 0.5 * dx:
            return {"mode": "index", "text": f"same number of points, x differs by ≤ {dev:.3g} (< ½ step) – "
                                             "point-by-point matching (recorded simultaneously)"}
    lo = max(float(s.x.min()) for s in spectra)
    hi = min(float(s.x.max()) for s in spectra)
    if not lo < hi:
        return {"mode": "none", "text": "the x ranges do not overlap"}
    return {"mode": "interpolate", "text": f"different x grids – linear interpolation onto the grid of the "
                                           f"reference dataset within the common range [{lo:.6g}, {hi:.6g}] "
                                           "(correlates neighbouring points)"}


def _interp_with_sigma(s: Spectrum, xq: np.ndarray):
    """Linear interpolation of y and sigma (sigma of a convex combination of two independent points)."""
    idx = np.clip(np.searchsorted(s.x, xq, side="right") - 1, 0, s.n - 2)
    x0, x1 = s.x[idx], s.x[idx + 1]
    t = np.where(x1 > x0, (xq - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    y = (1 - t) * s.y[idx] + t * s.y[idx + 1]
    sig = None
    if s.sigma is not None:
        sig = np.sqrt(((1 - t) * s.sigma[idx]) ** 2 + (t * s.sigma[idx + 1]) ** 2)
    excl = None
    if s.exclude is not None:
        excl = s.exclude[idx] | s.exclude[idx + 1]
    return y, sig, excl


def align(spectra: list, mode: str = "auto", reference: int = 0) -> dict:
    """Bring datasets onto a common grid. Returns x, per-dataset y/sigma/x, exclusion mask, info."""
    if not spectra:
        raise CombineError("no datasets")
    if mode == "auto":
        mode = diagnose_alignment(spectra)["mode"]
        if mode == "none":
            raise CombineError("the x ranges of the datasets do not overlap")
    ref = spectra[reference]
    warnings = []
    reported = mode
    if mode == "exact":
        if not all(s.n == ref.n and np.array_equal(s.x, ref.x) for s in spectra):
            raise CombineError("x grids not identical – choose alignment 'Index' or 'Interpolation'")
        mode = "index"
    if mode == "index":
        if not all(s.n == ref.n for s in spectra):
            raise CombineError("point-by-point matching requires the same number of points in all datasets")
        x = ref.x.copy()
        ys = [s.y.copy() for s in spectra]
        sigs = [None if s.sigma is None else s.sigma.copy() for s in spectra]
        xs = [s.x.copy() for s in spectra]
        excl = np.zeros(ref.n, bool)
        for s in spectra:
            if s.exclude is not None:
                excl |= s.exclude
        dev = max((float(np.max(np.abs(s.x - ref.x))) for s in spectra), default=0.0)
        if dev > 0:
            warnings.append(f"Point-by-point matching with differing x values (max. {dev:.3g}); x of the "
                            f"reference dataset is used")
        interpolated = False
    elif mode == "interpolate":
        lo = max(float(s.x.min()) for s in spectra)
        hi = min(float(s.x.max()) for s in spectra)
        if not lo < hi:
            raise CombineError("the x ranges of the datasets do not overlap")
        for s in spectra:
            if not s.is_strictly_increasing:
                raise CombineError("interpolation requires strictly increasing x values in all datasets")
        m = (ref.x >= lo) & (ref.x <= hi)
        x = ref.x[m]
        ys, sigs, xs = [], [], []
        excl = np.zeros(len(x), bool)
        for i, s in enumerate(spectra):
            if i == reference:
                ys.append(s.y[m].copy())
                sigs.append(None if s.sigma is None else s.sigma[m].copy())
                if s.exclude is not None:
                    excl |= s.exclude[m]
            else:
                y, sig, e = _interp_with_sigma(s, x)
                ys.append(y)
                sigs.append(sig)
                if e is not None:
                    excl |= e
            xs.append(x.copy())
        dropped = ref.n - len(x)
        warnings.append("Interpolation onto the reference grid: neighbouring points are correlated, fit errors "
                        "are underestimated")
        if dropped:
            warnings.append(f"{dropped} points outside the common x range discarded")
        interpolated = True
    else:
        raise CombineError(f"unknown alignment {mode!r}")
    return {"x": x, "ys": ys, "sigmas": sigs, "xs": xs, "exclude": excl, "mode": reported,
            "interpolated": interpolated, "warnings": warnings}


def _compile(expr: str, names: list) -> Formula:
    try:
        f = Formula(expr, independent=names)
    except FormulaError as e:
        raise CombineError(f"expression {expr!r}: {e}") from None
    if f.parameters:
        raise CombineError(f"expression {expr!r}: unknown names {', '.join(f.parameters)} "
                           f"(available: {', '.join(names)})")
    return f


def _propagate(f: Formula, args: list, sigmas: list, data_idx: list):
    """sigma of f by central differences w.r.t. each uncertain input array (per point)."""
    var = np.zeros(len(args[0]))
    used_uncertain = False
    for k in data_idx:
        sig = sigmas[k]
        if sig is None:
            continue
        a = args[k]
        h = 1e-6 * np.maximum(np.abs(a), sig)
        h = np.where(h > 0, h, 1e-12)
        up = list(args)
        dn = list(args)
        up[k] = a + h
        dn[k] = a - h
        d = (f(*up) - f(*dn)) / (2 * h)
        if np.any(d != 0):
            used_uncertain = True
        var = var + (d * sig) ** 2
    return np.sqrt(var), used_uncertain


def combine(datasets: dict, x_expr: str = "x", y_expr: str = "a", mode: str = "auto", reference: str | None = None,
            x_label: str = "", y_label: str = "", name: str = "") -> tuple:
    """Evaluate ``x_expr`` and ``y_expr`` on aligned datasets.

    ``datasets`` maps alias -> Spectrum. Returns (Spectrum, report dict).
    """
    aliases = list(datasets)
    for a in aliases:
        _check_alias(a)
    spectra = [datasets[a] for a in aliases]
    ref_i = aliases.index(reference) if reference else 0
    al = align(spectra, mode, ref_i)
    names = ["x"] + aliases + [f"x_{a}" for a in aliases]
    fx = _compile(x_expr, names)
    fy = _compile(y_expr, names)
    args = [al["x"]] + al["ys"] + al["xs"]
    sigmas = [None] + al["sigmas"] + [None] * len(aliases)
    data_idx = list(range(1, 1 + len(aliases)))
    with np.errstate(all="ignore"):
        X = fx(*args)
        Y = fy(*args)
    report = {"alignment": al["mode"], "warnings": list(al["warnings"]), "n_input": int(len(al["x"]))}
    ref_spec = spectra[ref_i]

    y_inputs = [a for a, k in zip(aliases, data_idx) if a in _names_in(y_expr)]
    have_sigma_y = all(al["sigmas"][aliases.index(a)] is not None for a in y_inputs) and bool(y_inputs)
    with np.errstate(all="ignore"):
        sy, _ = _propagate(fy, args, sigmas, data_idx) if have_sigma_y else (None, False)
        sx, x_unc = _propagate(fx, args, sigmas, data_idx)
    if y_inputs and not have_sigma_y and any(al["sigmas"][aliases.index(a)] is not None for a in y_inputs):
        report["warnings"].append("Not all datasets used in y have σ – result without σ")

    good = np.isfinite(X) & np.isfinite(Y)
    if sy is not None:
        good &= np.isfinite(sy) & (sy > 0)
    n_bad = int((~good).sum())
    if n_bad:
        report["warnings"].append(f"{n_bad} points undefined (e.g. division by 0, log ≤ 0) – discarded")
    if sy is not None:
        poorly = good & (sy > np.abs(Y))
        if poorly.sum():
            report["warnings"].append(f"{int(poorly.sum())} points with σ_y > |y| (e.g. denominator near 0) – "
                                      "poorly determined")
    flags = set()
    for s in spectra:
        flags |= set(s.flags)
    if al["interpolated"]:
        flags.add("interpolated")
    aux = {}
    if x_unc and np.any(sx[good] > 0):
        flags.add("x_uncertain")
        aux["sigma_x"] = sx[good]
        rel = float(np.nanmedian(sx[good] / np.maximum(np.abs(X[good]), 1e-300)))
        report["warnings"].append(f"x depends on uncertain measured values (median σ_x/|x| = {rel:.2g}); "
                                  "an ordinary fit ignores σ_x")
    if x_expr.strip() != "x":
        order = np.argsort(X[good], kind="stable")
        nonmono = np.any(np.diff(X[good]) < 0)
    else:
        order = np.arange(int(good.sum()))
        nonmono = False
    xg, yg = X[good][order], Y[good][order]
    sg = None if sy is None else sy[good][order]
    eg = al["exclude"][good][order]
    if al["mode"] in ("exact", "index") and "acq_index" in ref_spec.aux:
        aux["acq_index"] = np.asarray(ref_spec.aux["acq_index"])[good]   # keep acquisition order
    aux = {k: v[order] for k, v in aux.items()}
    if nonmono:
        report["warnings"].append("New x values are not monotonic in acquisition order – points were "
                                  "sorted by x (displayed as points)")
    report["n_output"] = int(len(xg))
    ref = spectra[ref_i]
    meta = {"name": name or f"{y_expr} vs {x_expr}",
            "derived": {"sources": [{"alias": a, "name": s.meta.get("name", "")} for a, s in zip(aliases, spectra)],
                        "x_expr": x_expr, "y_expr": y_expr, "alignment": al["mode"],
                        "warnings": report["warnings"]},
            "plot_style": "scatter" if x_expr.strip() != "x" else "line"}
    out = Spectrum(xg, yg, sg, SigmaSource.KNOWN if sg is not None else SigmaSource.UNKNOWN,
                   x_unit=ref.x_unit if x_expr.strip() == "x" else "",
                   y_unit="", x_label=x_label or (ref.x_label if x_expr.strip() == "x" else x_expr),
                   y_label=y_label or y_expr, exclude=eg if eg.any() else None, meta=meta,
                   flags=frozenset(flags), aux=aux)
    if sg is not None:
        srcs = {s.sigma_source for a, s in zip(aliases, spectra) if a in y_inputs}
        for cand in (SigmaSource.POISSON_DATA, SigmaSource.ESTIMATED_DERSNR, SigmaSource.ESTIMATED_REGION,
                     SigmaSource.CONSTANT, SigmaSource.KNOWN):
            if cand in srcs:            # report the weakest source among the inputs
                out = out.replace(sigma_source=cand)
                break
        if srcs != {SigmaSource.KNOWN}:
            report["warnings"].append("σ of the inputs: " + ", ".join(sorted(x.label for x in srcs))
                                      + " – the propagated σ inherits this origin")
    return out, report


def _names_in(expr: str) -> set:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
