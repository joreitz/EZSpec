"""Uncertainty estimates beyond the linearised covariance.

* :func:`profile_ci` – profile-likelihood confidence intervals (asymmetric).
  For each parameter, the others are re-optimised along a grid until the
  objective rises by the threshold. With *absolute* covariance (sigma known)
  the threshold is Delta chi2 = chi2_1 quantile (1.0 for 68.27 %); with
  *scaled* covariance (sigma unknown) it is the F-based threshold
  chi2_min/nu * F(1, nu) quantile, as in lmfit's ``conf_interval``.
* :func:`bootstrap` – residual bootstrap (homoscedastic in normalised
  residuals) or wild bootstrap with Rademacher signs (heteroscedastic;
  MacKinnon 2012). Residuals are rescaled by sqrt(N/nu) to undo the
  shrinkage of fitted residuals. Pairs bootstrap is not offered: on a fixed
  x grid it would change the design.
"""

from __future__ import annotations

import contextlib
import io

import lmfit
import numpy as np
from scipy import stats as sps
from scipy.optimize import brentq


def _refit(fr, fixed: dict, start, y=None):
    model = fr._internals["model"]
    x = fr._internals["x"]
    w = fr._internals["w"]
    yy = fr._internals["y"] if y is None else y
    params = start.copy()
    for k, v in fixed.items():
        params[k].value = v
        params[k].vary = False

    variables = fr._internals.get("variables")

    def residual(p):
        return (yy - model.evaluate(x, p, variables)) * w

    if not any(p.vary for p in params.values()):
        r = residual(params)
        return float(r @ r), params
    mini = lmfit.Minimizer(residual, params, nan_policy="omit", scale_covar=False)
    res = mini.minimize(method="leastsq")
    return float(res.chisqr), res.params


def profile_ci(fr, names=None, levels=(0.6827, 0.9545), max_expand: int = 30, progress=None) -> dict:
    """Profile-likelihood confidence intervals for the varying parameters."""
    best = fr._internals["best_params"]
    it = fr._internals
    chi0 = float(np.sum((it["y"] - it["model"].evaluate(it["x"], best, it.get("variables"))) ** 2
                        * it["w"] ** 2))
    dof = fr.stats.dof
    scaled = fr._internals["scale_covar"]
    names = list(names or fr.var_names)
    out = {}
    total = len(names) * len(levels) * 2
    done = 0
    for name in names:
        p = fr.params[name]
        se = p.stderr if p.stderr and np.isfinite(p.stderr) else max(abs(p.value) * 0.1, 1e-6)
        intervals = {}
        for level in levels:
            if scaled:
                delta = chi0 / dof * sps.f.ppf(level, 1, dof)
            else:
                delta = sps.chi2.ppf(level, 1)
            cache = {}

            def h(v):
                if v not in cache:
                    chi, _ = _refit(fr, {name: v}, best)
                    cache[v] = chi - chi0 - delta
                return cache[v]

            bounds = []
            for sign in (-1, +1):
                limit = p.min if sign < 0 else p.max
                step = se
                inner = p.value
                outer = None
                for _ in range(max_expand):
                    cand = p.value + sign * step
                    if np.isfinite(limit) and (cand - limit) * sign >= 0:
                        cand = limit
                    try:
                        val = h(cand)
                    except Exception:  # noqa: BLE001
                        val = np.nan
                    if np.isfinite(val) and val > 0:
                        outer = cand
                        break
                    if cand == limit:
                        break
                    inner = cand
                    step *= 2
                if outer is None:
                    bounds.append(None)          # interval not closed (bound or flat profile)
                else:
                    try:
                        bounds.append(float(brentq(h, inner, outer, xtol=1e-6 * se, rtol=1e-10)))
                    except Exception:  # noqa: BLE001
                        bounds.append(None)
                done += 1
                if progress:
                    progress(done, total)
            intervals[level] = tuple(bounds)
        out[name] = intervals
    summary = []
    for name, iv in out.items():
        parts = []
        for level, (lo, hi) in iv.items():
            los = "open" if lo is None else f"{lo:.6g}"
            his = "open" if hi is None else f"{hi:.6g}"
            parts.append(f"{100 * level:.2f} %: [{los}, {his}]")
        summary.append(f"{name} = {fr.params[name].value:.6g}:  " + "   ".join(parts))
    mode = ("Δχ² = χ²₁ quantile (σ known)" if not scaled else "F-test threshold (σ from residuals)")
    summary.append(f"Threshold: {mode}; 'open' = interval not closed (bound or flat profile)")
    result = {"intervals": {k: {str(l): v for l, v in iv.items()} for k, iv in out.items()},
              "threshold": mode, "summary": summary}
    fr.extra["profile_ci"] = result
    return result


def bootstrap(fr, n_samples: int = 500, kind: str = "residual", seed: int | None = 0,
              progress=None, cancel=None) -> dict:
    """Residual or wild bootstrap of the fit; returns percentile intervals."""
    rng = np.random.default_rng(seed)
    x = fr._internals["x"]
    w = fr._internals["w"]
    model = fr._internals["model"]
    best = fr._internals["best_params"]
    f = model.evaluate(x, best, fr._internals.get("variables"))
    y = fr._internals["y"]
    n = len(y)
    dof = fr.stats.dof
    scale = np.sqrt(n / dof)
    z = (y - f) * w * scale                  # normalised, rescaled residuals
    names = list(fr.var_names)
    peaks = model.peaks
    samples, dsamples, failures = [], [], 0
    for b in range(n_samples):
        if cancel is not None and cancel():
            break
        if kind == "residual":
            zb = rng.choice(z - z.mean(), size=n, replace=True)
        elif kind == "wild":
            zb = z * rng.choice([-1.0, 1.0], size=n)
        else:
            raise ValueError("kind must be 'residual' or 'wild'")
        yb = f + zb / w
        try:
            _, pb = _refit(fr, {}, best, y=yb)
        except Exception:  # noqa: BLE001
            failures += 1
            continue
        vals = {k: p.value for k, p in pb.items()}
        samples.append([vals[k] for k in names])
        dsamples.append({f"{c.prefix.rstrip('_')}.{k}": v for c in peaks for k, v in c.derived(vals).items()})
        if progress:
            progress(b + 1, n_samples)
    arr = np.array(samples)
    res = {"kind": kind, "n": int(len(arr)), "failures": failures, "params": {}, "derived": {}}
    summary = [f"{'Residual' if kind == 'residual' else 'Wild (Rademacher)'} bootstrap, "
               f"{len(arr)} successful replicates, {failures} failures"]

    def describe(vals):
        q = np.percentile(vals, [2.5, 15.865, 50, 84.135, 97.5])
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                "ci68": (float(q[1]), float(q[3])), "ci95": (float(q[0]), float(q[4])),
                "median": float(q[2])}

    if len(arr) > 1:
        for i, k in enumerate(names):
            d = describe(arr[:, i])
            res["params"][k] = d
            summary.append(f"{k}: SD = {d['std']:.4g} (linear: {fr.params[k].stderr or float('nan'):.4g}), "
                           f"68 % [{d['ci68'][0]:.6g}, {d['ci68'][1]:.6g}], 95 % [{d['ci95'][0]:.6g}, "
                           f"{d['ci95'][1]:.6g}]")
        for key in (dsamples[0] if dsamples else {}):
            res["derived"][key] = describe(np.array([d[key] for d in dsamples]))
    res["summary"] = summary
    fr.extra["bootstrap"] = res
    return res


def mcmc(fr, steps: int = 3000, burn: int = 1000, thin: int = 10, nwalkers: int | None = None,
         seed: int | None = 0, max_derived: int = 2000) -> dict:
    """Posterior sampling with emcee (via lmfit). Optional dependency: ``pip install emcee``.

    Priors are flat within the parameter bounds (improper for unbounded
    parameters). With sigma known the likelihood uses it; with sigma unknown
    the noise level is sampled as ``__lnsigma``. Reports medians and the
    central 68.27 % / 95.45 % credible intervals, also for derived peak
    quantities, plus acceptance fraction and autocorrelation times.
    """
    try:
        import emcee  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("MCMC requires the package 'emcee' (pip install emcee)") from exc
    it = fr._internals
    model, x, y, w, variables = it["model"], it["x"], it["y"], it["w"], it.get("variables")
    best = it["best_params"].copy()
    is_weighted = fr.stats.chi2 is not None

    def residual(p):
        return (y - model.evaluate(x, p, variables)) * w

    nvar = sum(1 for p in best.values() if p.vary)
    walkers = nwalkers or max(2 * nvar + 2, 32)
    mini = lmfit.Minimizer(residual, best, nan_policy="omit")
    note = io.StringIO()
    with contextlib.redirect_stdout(note):          # lmfit prints emcee's autocorrelation warning
        res = mini.emcee(params=best, steps=steps, burn=burn, thin=thin, nwalkers=walkers,
                         is_weighted=is_weighted, seed=seed, progress=False)
    cols = list(res.var_names)
    arr = np.asarray(res.chain).reshape(-1, len(cols))
    q = (2.275, 15.865, 50.0, 84.135, 97.725)
    out = {"params": {}, "derived": {}, "n_samples": int(arr.shape[0]), "nwalkers": walkers,
           "acceptance": float(np.mean(res.acceptance_fraction)), "is_weighted": is_weighted}
    acor = getattr(res, "acor", None)
    summary = [f"emcee: {walkers} walkers, {steps} steps (burn-in {burn}, thinning {thin}), "
               f"{arr.shape[0]} samples, acceptance fraction {out['acceptance']:.2f}; flat priors within the bounds"]
    if note.getvalue().strip():
        summary.append("Note: chain possibly too short for reliable autocorrelation times – use more steps.")
        out["warning"] = note.getvalue().strip()
    for j, name in enumerate(cols):
        v = np.percentile(arr[:, j], q)
        d = {"median": float(v[2]), "ci68": (float(v[1]), float(v[3])), "ci95": (float(v[0]), float(v[4]))}
        if acor is not None and j < len(acor):
            d["autocorr_time"] = float(acor[j])
        out["params"][name] = d
        summary.append(f"{name}: median {v[2]:.6g}, 68 % [{v[1]:.6g}, {v[3]:.6g}], 95 % [{v[0]:.6g}, {v[4]:.6g}]"
                       + (f", τ_int ≈ {d['autocorr_time']:.0f}" if "autocorr_time" in d else ""))
    peaks = model.peaks
    if peaks:
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.shape[0], size=min(max_derived, arr.shape[0]), replace=False)
        work = best.copy()
        samples = {}
        for i in idx:
            for j, name in enumerate(cols):
                if name in work:
                    work[name].value = float(arr[i, j])
            work.update_constraints()
            vals = {k: p.value for k, p in work.items()}
            for c in peaks:
                for k, v in c.derived(vals).items():
                    samples.setdefault(f"{c.prefix.rstrip('_')}.{k}", []).append(v)
        for key, vals in samples.items():
            v = np.percentile(vals, q)
            out["derived"][key] = {"median": float(v[2]), "ci68": (float(v[1]), float(v[3])),
                                   "ci95": (float(v[0]), float(v[4]))}
    out["summary"] = summary
    fr.extra["mcmc"] = out
    return out
