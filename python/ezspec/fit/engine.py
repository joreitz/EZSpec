"""Fit engine (lmfit/SciPy as reference engine) with explicit statistics semantics.

The engine decides – and reports – three things that common tools leave
implicit:

1. **sigma source**: known/estimated sigma, Poisson (sigma^2 = model) or
   unknown. Only with sigma known is chi-square a chi-square.
2. **covariance mode**: *absolute* (sigma taken at face value) or *scaled*
   by the reduced chi-square / residual variance. With unknown sigma scaling
   is required (it *is* the estimate of sigma^2). With known sigma the
   default is absolute, because scaling would hide a bad model behind
   inflated error bars; scaling can be requested and is then flagged.
3. **diagnostics**: parameters at bounds, high correlations, rank-deficient
   Jacobians, systematic residual structure, and processing steps upstream
   that invalidate the noise model (smoothing, interpolation).
"""

from __future__ import annotations

import copy
import datetime as _dt
import time
from dataclasses import asdict, dataclass

import lmfit
import numpy as np
import scipy
from scipy import stats as sps

from .._backend import backend_name
from ..models.model import Model
from ..spectrum import SigmaSource, Spectrum
from . import stats as S
from .result import DerivedResult, FitResult, FitStatistics, FitWarning, ParamResult

METHODS = {
    "leastsq": "Levenberg–Marquardt (MINPACK)",
    "least_squares": "Trust Region Reflective (SciPy)",
    "nelder": "Nelder–Mead (+ LM polish)",
    "powell": "Powell (+ LM polish)",
    "differential_evolution": "Differential Evolution, global (+ LM polish)",
    "basinhopping": "Basin-Hopping, global (+ LM polish)",
    "ampgo": "AMPGO, global (+ LM polish)",
}
GLOBAL_OR_SCALAR = {"nelder", "powell", "differential_evolution", "basinhopping", "ampgo"}


@dataclass
class FitOptions:
    method: str = "leastsq"
    weighting: str = "auto"          # auto | sigma | none | poisson_model
    covariance: str = "auto"         # auto | absolute | scaled
    x_range: list | None = None      # [xmin, xmax]
    loss: str = "linear"             # least_squares only: linear | soft_l1 | huber | cauchy | arctan
    max_nfev: int | None = None
    polish: bool = True
    poisson_max_iter: int = 50
    tol: float = 1e-10               # ftol = xtol for leastsq / least_squares (MINPACK default 1.5e-8)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


class FitError(RuntimeError):
    pass


# ============================================================================ helpers
def _select(spectrum: Spectrum, opt: FitOptions):
    mask = spectrum.fit_mask.copy()
    if opt.x_range:
        lo, hi = opt.x_range
        if lo is not None:
            mask &= spectrum.x >= lo
        if hi is not None:
            mask &= spectrum.x <= hi
    return mask


def _bound_flags(v, lo, hi, se):
    at, near = None, False
    for side, b in (("min", lo), ("max", hi)):
        if b is None or not np.isfinite(b):
            continue
        dist = abs(v - b)
        scale = max(abs(b), abs(v), 1e-300)
        if dist <= 1e-6 * scale or (se is not None and np.isfinite(se) and se > 0 and dist <= 1e-3 * se):
            at = side
        elif se is not None and np.isfinite(se) and dist < 2 * se:
            near = True
    return at, near


_EPS = np.finfo(float).eps


def _jacobian(fun, theta, scale):
    """d fun / d theta by central differences with one Richardson step (error O(h^4)).

    ``scale`` is the natural scale of each parameter (|value| or its standard
    error); the step is eps^(1/5) * scale, which balances truncation and
    round-off error.
    """
    f0 = np.asarray(fun(theta), float)
    J = np.empty((f0.size, theta.size))
    for i in range(theta.size):
        h = _EPS ** 0.2 * scale[i]

        def d(hh):
            tp = theta.copy()
            tm = theta.copy()
            tp[i] += hh
            tm[i] -= hh
            return (np.asarray(fun(tp), float) - np.asarray(fun(tm), float)) / (2.0 * hh)

        J[:, i] = (4.0 * d(0.5 * h) - d(h)) / 3.0
    return J


def _scales(params, names, se=None):
    """Natural scale per parameter: min(|value|, standard error) where both are known.

    The objective varies smoothly on the scale of the standard error, so it is
    a good finite-difference scale; |value| caps it for poorly determined
    parameters.
    """
    out = []
    for i, n in enumerate(names):
        v = abs(params[n].value)
        e = se[i] if se is not None and np.isfinite(se[i]) and se[i] > 0 else 0.0
        cands = [c for c in (v, e) if c > 0]
        sc = min(cands) if cands else 1e-6
        # floor relative to |value| so that theta + h != theta in floating point
        out.append(max(sc, 1e-6 * v))
    return np.array(out)


def _make_setter(params, names):
    """Function theta -> dict of all parameter values with constraints applied.

    Bounds are removed on the working copy so that finite differences at a
    bound are not clipped (lmfit clips values to [min, max]).
    """
    work = params.copy()
    for p in work.values():
        p.min, p.max = -np.inf, np.inf

    def values(theta):
        for n, v in zip(names, theta):
            work[n].value = float(v)
        work.update_constraints()
        return {k: p.value for k, p in work.items()}

    return values


def propagate(fn, params, names, covar):
    """Value and standard error of fn(values) by the delta method."""
    theta = np.array([params[n].value for n in names], float)
    setter = _make_setter(params, names)
    v0 = fn(setter(theta))
    if covar is None:
        return v0, None
    se = np.sqrt(np.clip(np.diag(covar), 0, None))
    g = _jacobian(lambda t: np.atleast_1d(fn(setter(t))), theta, _scales(params, names, se))[0]
    var = float(g @ covar @ g)
    return v0, (np.sqrt(var) if var >= 0 else float("nan"))


def _refine(resid, theta, jac, lo, hi, max_iter=30):
    """Gauss–Newton refinement with an accurate Jacobian and step halving.

    Only steps that stay inside the bounds and do not increase the objective
    are accepted. Returns the refined theta and the number of accepted steps.
    """
    r = resid(theta)
    obj = float(r @ r)
    accepted = 0
    for _ in range(max_iter):
        J = jac(theta)
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]
        step = 1.0
        ok = False
        while step > 1e-3:
            cand = theta + step * delta
            if np.all(cand >= lo) and np.all(cand <= hi):
                rc = resid(cand)
                oc = float(rc @ rc)
                if np.isfinite(oc) and oc <= obj:
                    ok = True
                    break
            step *= 0.5
        if not ok:
            break
        change = np.max(np.abs(cand - theta) / np.maximum(np.abs(theta), 1e-300))
        theta, r, obj = cand, rc, oc
        accepted += 1
        if change < 1e-15:
            break
    return theta, accepted


def _covariance_from_jacobian(J, scale_factor):
    """C = (J^T J)^-1 * scale_factor via SVD; None if J is numerically rank deficient."""
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return None, float("inf")
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    if not np.isfinite(cond) or cond > 1e15:
        return None, cond
    C = (Vt.T / s**2) @ Vt * scale_factor
    return 0.5 * (C + C.T), cond


# ============================================================================ engine
def fit(spectrum: Spectrum, model: Model, options: FitOptions | dict | None = None) -> FitResult:
    """Fit ``model`` to ``spectrum``. Start values, bounds, fixed parameters and
    constraints are taken from the model's parameter settings."""
    opt = options if isinstance(options, FitOptions) else FitOptions.from_dict(options)
    if opt.method not in METHODS:
        raise FitError(f"unknown method {opt.method!r}")
    t_start = time.perf_counter()
    mask = _select(spectrum, opt)
    x = spectrum.x[mask]
    y = spectrum.y[mask]
    variables = {k[4:]: np.asarray(v)[mask] for k, v in spectrum.aux.items() if k.startswith("var:")}
    warnings: list[FitWarning] = []

    # ---------------------------------------------------------------- weighting
    weighting = opt.weighting
    if weighting == "auto":
        weighting = "sigma" if spectrum.sigma is not None else "none"
    if weighting == "sigma":
        if spectrum.sigma is None:
            raise FitError("weighting 'sigma' selected, but the data have no σ "
                           "(import a σ column or apply 'Estimate noise → σ')")
        sigma = spectrum.sigma[mask].astype(float)
        source = spectrum.sigma_source
        sigma_label = source.label
    elif weighting == "none":
        sigma = None
        source = SigmaSource.UNKNOWN
        sigma_label = SigmaSource.UNKNOWN.label + " (unit weights)"
    elif weighting == "effective_variance":
        if spectrum.sigma is None or "sigma_x" not in spectrum.aux:
            raise FitError("effective variance requires σ_y and σ_x (e.g. from 'Combine datasets')")
        sigma = spectrum.sigma[mask].astype(float)
        sigma_x = np.asarray(spectrum.aux["sigma_x"], float)[mask]
        source = spectrum.sigma_source
        sigma_label = source.label + " + σ_x (effective variance, Orear 1982)"
    elif weighting == "poisson_model":
        sigma = None
        source = None
        sigma_label = "Poisson σ² = model (IRLS ≙ Poisson ML)"
        if np.any(y < 0):
            warnings.append(FitWarning("POISSON_NEGATIVE", "Poisson weighting with negative data – "
                                       "are these really counts (no subtracted baseline)?"))
        if "baseline_subtracted" in spectrum.flags or "normalized" in spectrum.flags:
            warnings.append(FitWarning("POISSON_PROCESSED", "Poisson statistics apply only to unprocessed "
                                       "counts (baseline subtracted/normalized)."))
    else:
        raise FitError(f"unknown weighting {weighting!r}")
    sigma_known = weighting in ("sigma", "poisson_model", "effective_variance")

    # ---------------------------------------------------------------- covariance mode
    cov_mode = opt.covariance
    if cov_mode == "auto":
        cov_mode = "absolute" if sigma_known else "scaled"
    if cov_mode == "absolute" and not sigma_known:
        warnings.append(FitWarning("ABSOLUTE_WITHOUT_SIGMA", "Absolute covariance without known σ is "
                                   "meaningless – scaling by the residual variance instead.", "warning"))
        cov_mode = "scaled"
    if cov_mode == "scaled" and sigma_known:
        warnings.append(FitWarning("SCALED_KNOWN_SIGMA", "Covariance scaled by χ²_ν although σ is known: "
                                   "a misfit then shows up as larger error bars instead of as a "
                                   "warning.", "warning"))
    scale_covar = cov_mode == "scaled"

    # ---------------------------------------------------------------- upstream processing
    flag_warnings = {
        "smoothed": ("SMOOTHED_INPUT", "Fit on smoothed data: noise is correlated, peaks are broadened/"
                     "lowered; uncertainties are strongly underestimated (O'Haver). Fit unsmoothed data!",
                     "warning"),
        "interpolated": ("INTERPOLATED_INPUT", "Fit on interpolated/resampled data: neighbouring "
                         "points are correlated, effective N is smaller; χ² and errors are not exact.", "warning"),
        "normalized": ("NORMALIZED_INPUT", "Data normalized: amplitudes/areas refer to the "
                       "normalized scale.", "info"),
        "x_uncertain": ("X_UNCERTAIN", "x values have an uncertainty (σ_x, e.g. from combining "
                        "series). Ordinary least squares ignores σ_x; weighting 'effective variance' "
                        "accounts for it.", "warning"),
        "baseline_subtracted": ("BASELINE_UNCERTAINTY", "Baseline subtracted beforehand: its uncertainty is "
                                "not included in the errors (systematic; e.g. vary λ or "
                                "fit a linear baseline simultaneously).", "info"),
    }
    for flag, (code, msg, sev) in flag_warnings.items():
        if flag in spectrum.flags:
            warnings.append(FitWarning(code, msg, sev))
    from ..sweeps import count_sweeps, is_multivalued
    if is_multivalued(spectrum) and not model.independent_variables:   # y(x, v): repeats in x are expected
        c = count_sweeps(spectrum)
        warnings.append(FitWarning("MULTI_SWEEP", f"Data contain several sweeps ({c['up']} up, "
                                   f"{c['down']} down): x is ambiguous. Evaluate up and down sweeps separately "
                                   "(step 'Select sweeps (up/down)' or 'Average sweeps (same direction)').", "warning"))
    cal = spectrum.meta.get("calibration")
    if "x_calibrated" in spectrum.flags and cal:
        warnings.append(FitWarning("X_CALIBRATION", f"x axis calibrated ({cal.get('source', '')}): the systematic "
                                   f"calibration uncertainty σ_x,cal ≈ {cal.get('sigma_median', float('nan')):.3g} "
                                   f"(max. {cal.get('sigma_max', float('nan')):.3g}) is not included in the errors; "
                                   "position parameters inherit it in full.", "info"))
    if sigma_known and source is SigmaSource.POISSON_DATA:
        warnings.append(FitWarning("NEYMAN_WEIGHTS", "σ = √y (Neyman) biases amplitudes and areas "
                                   "downwards, even at high count rates (Humphrey et al. 2009). Prefer "
                                   "weighting 'Poisson: σ² = model'.", "warning"))
    if opt.loss != "linear":
        if opt.method != "least_squares":
            raise FitError("robust loss functions require method 'least_squares'")
        warnings.append(FitWarning("ROBUST_LOSS", f"Robust loss function '{opt.loss}': covariance and χ² "
                                   "are not the usual Gaussian quantities.", "warning"))

    # ---------------------------------------------------------------- parameters
    params = model.make_params()
    nvary = sum(1 for p in params.values() if p.vary)
    if nvary == 0:
        raise FitError("no free parameters")
    if len(x) <= nvary:
        raise FitError(f"too few data points ({len(x)}) for {nvary} free parameters")
    if opt.method == "differential_evolution":
        bad = [n for n, p in params.items() if p.vary and not (np.isfinite(p.min) and np.isfinite(p.max))]
        if bad:
            raise FitError(f"Differential Evolution requires finite bounds for: {', '.join(bad)}")
    init_values = {k: p.value for k, p in params.items()}
    init_fit = model.evaluate(x, params, variables)
    if not np.all(np.isfinite(init_fit)):
        raise FitError("model returns NaN/Inf at the start values – check start values or bounds")

    # ---------------------------------------------------------------- minimisation
    def run_once(start_params, w):
        def residual(p):
            return (y - model.evaluate(x, p, variables)) * w

        mini = lmfit.Minimizer(residual, start_params, nan_policy="raise", scale_covar=scale_covar)
        kws = {}
        if opt.max_nfev:
            kws["max_nfev"] = int(opt.max_nfev)
        if opt.method == "least_squares" and opt.loss != "linear":
            kws["loss"] = opt.loss
        if opt.method in ("leastsq", "least_squares") and opt.tol:
            kws.update(ftol=opt.tol, xtol=opt.tol)
        res = mini.minimize(method=opt.method, **kws)
        method_used = opt.method
        if opt.method in GLOBAL_OR_SCALAR and opt.polish:
            mini = lmfit.Minimizer(residual, res.params, nan_policy="raise", scale_covar=scale_covar)
            res2 = mini.minimize(method="leastsq", ftol=opt.tol, xtol=opt.tol)
            res2.nfev += res.nfev
            res = res2
            method_used = f"{opt.method} + leastsq"
        return mini, res, residual, method_used

    try:
        if weighting == "poisson_model":
            w = 1.0 / np.sqrt(np.maximum(y, 1.0))
            current = params
            for it in range(opt.poisson_max_iter):
                mini, res, residual, method_used = run_once(current, w)
                fbest = model.evaluate(x, res.params, variables)
                if np.any(fbest <= 0):
                    raise FitError("Poisson weighting: model ≤ 0 at some points")
                w_new = 1.0 / np.sqrt(fbest)
                change = np.max(np.abs(w_new - w) / w)
                w = w_new
                current = res.params
                if change < 1e-8:
                    break
            else:
                warnings.append(FitWarning("POISSON_NOT_CONVERGED", "Poisson IRLS did not converge."))
            # final fit with converged weights so covariance corresponds to sigma^2 = model
            mini, res, residual, method_used = run_once(current, w)
            sigma = 1.0 / w
        elif weighting == "effective_variance":
            sigma_y = sigma
            w = 1.0 / sigma_y
            current = params
            span = float(np.ptp(x)) or 1.0
            for _ in range(opt.poisson_max_iter):
                mini, res, residual, method_used = run_once(current, w)
                h = 1e-6 * span
                fp = (model.evaluate(x + h, res.params, variables) - model.evaluate(x - h, res.params, variables)) \
                    / (2 * h)
                w_new = 1.0 / np.sqrt(sigma_y ** 2 + (fp * sigma_x) ** 2)
                change = np.max(np.abs(w_new - w) / w)
                w = w_new
                current = res.params
                if change < 1e-8:
                    break
            else:
                warnings.append(FitWarning("EV_NOT_CONVERGED", "Effective-variance iteration did not converge."))
            mini, res, residual, method_used = run_once(current, w)
            sigma = 1.0 / w
            warnings = [w_ for w_ in warnings if w_.code != "X_UNCERTAIN"]
            warnings.append(FitWarning("EFFECTIVE_VARIANCE", "σ_x accounted for via effective variance (first-order "
                                       "approximation to errors-in-variables; good for small σ_x).", "info"))
        else:
            w = 1.0 / sigma if sigma is not None else np.ones_like(y)
            mini, res, residual, method_used = run_once(params, w)
    except FitError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FitError(f"fit failed: {type(exc).__name__}: {exc}") from exc

    # ---------------------------------------------------------------- refinement & covariance
    best = res.params
    names = list(res.var_names)
    lmfit_se = np.array([best[nm].stderr if best[nm].stderr else np.nan for nm in names], float)
    lo = np.array([best[nm].min for nm in names], float)
    hi = np.array([best[nm].max for nm in names], float)
    setter = _make_setter(best, names)

    def wresid(t):
        return (y - model.evaluate(x, setter(t), variables)) * w

    theta = np.array([best[nm].value for nm in names], float)
    scales = _scales(best, names, lmfit_se)
    jac = lambda t: _jacobian(wresid, t, scales)  # noqa: E731
    refined_steps = 0
    if opt.loss == "linear":
        try:
            theta, refined_steps = _refine(wresid, theta, jac, lo, hi)
        except Exception:  # noqa: BLE001 - refinement is optional
            refined_steps = 0
        for nm, v in zip(names, theta):
            best[nm].value = float(v)
        best.update_constraints()
    f = model.evaluate(x, best, variables)
    r = y - f
    zres = r * w
    n = len(y)
    p = res.nvarys
    dof = n - p
    rss = float(np.sum(r * r))
    wobj = float(np.sum(zres * zres))
    chi2 = wobj if sigma_known else None
    covar = None
    cond = float("nan")
    try:
        J = jac(theta)
        covar, cond = _covariance_from_jacobian(J, (wobj / dof if dof > 0 else np.nan) if scale_covar else 1.0)
    except Exception:  # noqa: BLE001
        covar = None
    if opt.loss != "linear":
        # robust loss: the Gauss covariance of the plain residuals does not apply; keep lmfit's estimate
        covar = res.covar if getattr(res, "errorbars", False) else None
    if covar is not None and not np.all(np.isfinite(covar)):
        covar = None
    if covar is None:
        cov_mode_final = "unavailable"
        warnings.append(FitWarning("NO_COVARIANCE", "Covariance matrix not available (Jacobian "
                                   "rank-deficient/singular) – at least one parameter is not determined "
                                   "by the data.", "warning"))
    else:
        cov_mode_final = cov_mode
    corr = None
    if covar is not None:
        d = np.sqrt(np.diag(covar))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = covar / np.outer(d, d)
    se_vary = dict(zip(names, np.sqrt(np.clip(np.diag(covar), 0, None)))) if covar is not None else {}

    neg2ll = deviance = None
    if weighting == "poisson_model":
        with np.errstate(divide="ignore", invalid="ignore"):
            neg2ll = float(2 * np.sum(f - np.where(y > 0, y * np.log(f), 0.0)))
            deviance = float(2 * np.sum(np.where(y > 0, y * np.log(y / f), 0.0) - (y - f)))
    ic = S.information_criteria(n, p, rss, chi2, sigma_known, neg2ll)
    r2 = S.r_squared(y, f, p)

    redchi = band = pval = None
    if sigma_known and dof > 0:
        redchi = chi2 / dof
        half = np.sqrt(2.0 / dof)
        band = (1 - half, 1 + half)
        pval = float(sps.chi2.sf(chi2, dof))
        if abs(redchi - 1) > 3 * half:
            if redchi > 1:
                warnings.append(FitWarning("REDCHI_HIGH", f"χ²_ν = {redchi:.3g} is well above 1 "
                                           f"(3σ band ±{3 * half:.2g}): the model does not describe the data "
                                           "within σ, or σ is underestimated.", "warning"))
            else:
                warnings.append(FitWarning("REDCHI_LOW", f"χ²_ν = {redchi:.3g} is well below 1: σ "
                                           "overestimated or overfitting.", "warning"))
    if dof < 5:
        warnings.append(FitWarning("FEW_DOF", f"only ν = {dof} degrees of freedom", "warning"))

    zdiag = zres if sigma_known else r / (np.sqrt(rss / dof) if dof > 0 and rss > 0 else 1.0)
    runs = S.runs_test(zdiag)
    if np.isfinite(runs.get("p_too_few", np.nan)) and runs["p_too_few"] < 0.01:
        warnings.append(FitWarning("RUNS_TEST", f"Runs test: {runs['runs']} instead of ~{runs['expected']:.0f} "
                                   f"runs of residual signs (p = {runs['p_too_few']:.2g}) – systematic "
                                   "deviation, model incomplete?", "warning"))
    lag1 = S.lag1_autocorrelation(zdiag)
    if np.isfinite(lag1) and abs(lag1) > 3 / np.sqrt(n):
        warnings.append(FitWarning("AUTOCORR", f"Residuals autocorrelated (ρ₁ = {lag1:.2f} > 3/√N)",
                                   "info" if "RUNS_TEST" in {w_.code for w_ in warnings} else "warning"))

    if not np.isfinite(cond) or cond > 1e12:
        warnings.append(FitWarning("ILL_CONDITIONED", f"Jacobian (nearly) rank-deficient (condition number "
                                   f"{cond:.2g}): at least one parameter combination is poorly determined "
                                   "by the data.", "warning"))

    # ---------------------------------------------------------------- parameters
    presults = {}
    for name, par in best.items():
        if covar is None:
            se = None
        elif name in se_vary:
            se = float(se_vary[name])
        elif par.expr:
            se = float(propagate(lambda v, nm=name: v[nm], best, names, covar)[1])
        else:
            se = None
        if se is not None and not np.isfinite(se):
            se = None
        at, near = (None, False)
        if par.vary:
            at, near = _bound_flags(par.value, par.min, par.max, se)
        presults[name] = ParamResult(name, float(par.value), None if se is None else float(se),
                                     float(init_values[name]), bool(par.vary), float(par.min),
                                     float(par.max), par.expr, at, near)
        if at:
            warnings.append(FitWarning("AT_BOUND", f"{name} is at a bound ({at} = "
                                       f"{par.min if at == 'min' else par.max:g}); standard error not "
                                       "reliable – check model/bounds.", "warning"))
        elif near:
            warnings.append(FitWarning("NEAR_BOUND", f"{name}: bound lies within 2 SE – symmetric "
                                       "error misleading, use the profile CI.", "info"))
    for a, b, c in _high_corr(names, corr):
        warnings.append(FitWarning("HIGH_CORRELATION", f"ρ({a}, {b}) = {c:+.3f}: parameters strongly correlated, "
                                   "individually poorly determined (consider constraints).", "info"))

    # ---------------------------------------------------------------- derived quantities
    derived = []
    peaks = model.peaks
    for comp in peaks:
        keys = list(comp.derived(_make_setter(best, names)(theta)).keys())
        for key in keys:
            val, se = propagate(lambda v, c=comp, k=key: c.derived(v)[k], best, names, covar)
            derived.append(DerivedResult(comp.prefix.rstrip("_") or comp.display_name, key, float(val),
                                         None if se is None else float(se)))
    groups = {}
    for comp in peaks:      # area fractions within each spectrum (group) – global fits have several
        groups.setdefault(getattr(comp, "group", 0), []).append(comp)
    for members in groups.values():
        if len(members) < 2:
            continue

        def frac(v, i, members=members):
            areas = [c.derived(v)["area"] for c in members]
            return areas[i] / sum(areas)
        for i, comp in enumerate(members):
            val, se = propagate(lambda v, i=i: frac(v, i), best, names, covar)
            derived.append(DerivedResult(comp.prefix.rstrip("_"), "area_fraction", float(val),
                                         None if se is None else float(se)))

    if not res.success:
        warnings.append(FitWarning("NOT_CONVERGED", f"Optimizer reports no success: {res.message}", "error"))

    stats = FitStatistics(
        n_points=n, n_varys=p, dof=dof,
        sigma_source=(source.value if source is not None else "poisson_model"),
        sigma_label=sigma_label, weighting=weighting, covariance_mode=cov_mode_final,
        rss=rss, s_res=float(np.sqrt(rss / dof)) if dof > 0 else float("nan"),
        rmse=float(np.sqrt(rss / n)), r2=r2["r2"], adj_r2=r2["adj_r2"],
        aic=ic["aic"], aicc=ic["aicc"], bic=ic["bic"], ic_form=ic["form"], ic_k=ic["k"],
        chi2=chi2, redchi=redchi, redchi_band=band, chi2_pvalue=pval, runs=runs,
        lag1_autocorr=lag1, durbin_watson=S.durbin_watson(zdiag), normality=S.normality(zdiag),
        jacobian_condition=cond, poisson_deviance=deviance,
    )
    if np.isnan(ic["aicc"]):
        warnings.append(FitWarning("AICC_UNDEFINED", "AICc undefined (K ≥ N − 1).", "info"))

    full_sigma = None if sigma is None else np.asarray(sigma, float)
    provenance = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ezspec": _version(), "lmfit": lmfit.__version__, "scipy": scipy.__version__,
        "numpy": np.__version__, "backend": backend_name(),
        "model_description": model.describe(), "seconds": time.perf_counter() - t_start,
        "spectrum_flags": sorted(spectrum.flags), "gauss_newton_steps": refined_steps,
        "baselines": copy.deepcopy(list(spectrum.meta.get("baselines", []))),
    }
    result = FitResult(
        success=bool(res.success), message=str(res.message), method=method_used, nfev=int(res.nfev),
        params=presults, var_names=names, covariance=covar, correlation=corr, stats=stats,
        derived=derived, warnings=warnings, x=x, y=y, sigma=full_sigma, best_fit=f, init_fit=init_fit,
        residuals=r, normalized_residuals=zdiag, components=model.evaluate_components(x, best, variables),
        mask=mask, model_spec=model.to_dict(), options=opt.to_dict(),
        data_hash=_data_hash(x, y, full_sigma, weighting), provenance=provenance,
        baseline=(np.asarray(spectrum.aux["baseline_total"], float)[mask]
                  if "baseline_total" in spectrum.aux else None),
    )
    result._internals = {"minimizer": mini, "lmfit_result": res, "best_params": best, "model": model.copy(),
                         "x": x, "y": y,
                         "w": w, "variables": variables, "scale_covar": scale_covar, "spectrum": spectrum, "options": opt}
    return result


def _high_corr(names, corr, threshold=0.9):
    out = []
    if corr is None:
        return out
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > threshold:
                out.append((names[i], names[j], float(corr[i, j])))
    return out


def _data_hash(x, y, sigma, weighting):
    import hashlib
    h = hashlib.sha256()
    for a in (x, y, sigma):
        h.update(b"none" if a is None else np.ascontiguousarray(a, float).tobytes())
    h.update(weighting.encode())
    return h.hexdigest()


def _version():
    from .. import __version__
    return __version__


def fit_xy(x, y, model: Model, sigma=None, options=None, **kw) -> FitResult:
    """Convenience: fit arrays directly."""
    from ..spectrum import spectrum
    return fit(spectrum(x, y, sigma, **kw), model, options)
