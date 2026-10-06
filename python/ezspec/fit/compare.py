"""Model comparison by information criteria."""

from __future__ import annotations

import numpy as np

from . import stats as S


class ComparisonError(ValueError):
    pass


def compare(results: dict) -> dict:
    """Compare fits of *different models to identical data*.

    ``results`` maps a model name to a FitResult. Comparison is refused when
    the fitted points, sigma or weighting differ (a common silent error:
    AIC values from different x ranges, baselines or weights are not
    comparable). Returns Delta AICc, Delta BIC (relative to the best model)
    and Akaike weights. Absolute AIC values carry no meaning.

    Note: deciding on "one more peak" by an F-test or likelihood-ratio test is
    not valid because a zero amplitude lies on the boundary of the parameter
    space (Protassov et al. 2002); calibrate such decisions by simulation.
    """
    if len(results) < 2:
        raise ComparisonError("mindestens zwei Fits nötig")
    names = list(results)
    hashes = {results[k].data_hash for k in names}
    if len(hashes) != 1:
        raise ComparisonError("Fits beruhen nicht auf identischen Daten (x-Bereich, Baseline, σ oder "
                              "Gewichtung verschieden) – Vergleich verweigert")
    forms = {results[k].stats.ic_form for k in names}
    if len(forms) != 1:
        raise ComparisonError("verschiedene Likelihood-Formen – Vergleich verweigert")
    use_aicc = all(np.isfinite(results[k].stats.aicc) for k in names)
    crit = np.array([results[k].stats.aicc if use_aicc else results[k].stats.aic for k in names])
    bic = np.array([results[k].stats.bic for k in names])
    w = S.akaike_weights(crit)
    rows = []
    for i, k in enumerate(names):
        rows.append({"model": k, "k": results[k].stats.ic_k, "criterion": "AICc" if use_aicc else "AIC",
                     "delta_aic": float(crit[i] - crit.min()), "akaike_weight": float(w[i]),
                     "delta_bic": float(bic[i] - bic.min()),
                     "redchi": results[k].stats.redchi, "s_res": results[k].stats.s_res})
    rows.sort(key=lambda r: r["delta_aic"])
    return {"rows": rows, "form": forms.pop(),
            "note": "Δ < 2: kaum Unterschied; 4–7: deutlich weniger gestützt; > 10: praktisch keine "
                    "Unterstützung (Burnham & Anderson). Für 'einen Peak mehr' per Simulation kalibrieren."}


def _objective(result) -> float:
    """-2 ln L up to a model-independent constant (for nested-model statistics)."""
    st = result.stats
    if st.weighting == "poisson_model":
        f = result.best_fit
        y = result.y
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(2 * np.sum(f - np.where(y > 0, y * np.log(f), 0.0)))
    if st.chi2 is not None:
        return float(st.chi2)
    return float(st.n_points * np.log(st.rss / st.n_points))


def simulate_nested_test(null, alt, n_sim: int = 200, seed: int | None = 0, progress=None, cancel=None) -> dict:
    """Simulation-calibrated likelihood-ratio test for nested models ("one more peak").

    ``null`` and ``alt`` are FitResults of the smaller and the larger model on
    identical data. The statistic is T = (-2 ln L)_null - (-2 ln L)_alt
    (Delta chi2 for known sigma, N ln(RSS0/RSS1) for unknown sigma). Its null
    distribution is obtained by a parametric bootstrap: data are simulated from
    the fitted null model with the fitted noise model (Gaussian sigma_i, or s
    if sigma is unknown, or Poisson), and both models are refitted to every
    replicate. p = (1 + #{T* >= T_obs}) / (n_sim + 1).

    The nominal chi2(Delta p) p-value is reported for comparison only: it is
    invalid when the extra component's amplitude sits on the boundary of the
    parameter space (Protassov et al. 2002). The additional component starts
    at its fitted position in every replicate, so the test is local (no
    look-elsewhere correction).
    """
    from scipy import stats as sps

    from .engine import fit

    if null.data_hash != alt.data_hash:
        raise ComparisonError("Fits beruhen nicht auf identischen Daten")
    dp = alt.stats.n_varys - null.stats.n_varys
    if dp <= 0:
        raise ComparisonError("Alternative muss mehr freie Parameter haben als das Nullmodell")
    rng = np.random.default_rng(seed)
    t_obs = _objective(null) - _objective(alt)
    spec = null._internals["spectrum"]
    mask = null.mask
    f0 = null.best_fit
    weighting = null.stats.weighting
    m_null = null._internals["model"].copy()
    m_alt = alt._internals["model"].copy()
    m_null.apply_values(null.values)
    m_alt.apply_values(alt.values)
    opt_null = null._internals["options"]
    opt_alt = alt._internals["options"]
    if weighting == "poisson_model":
        def draw():
            return rng.poisson(np.clip(f0, 0, None)).astype(float)
    elif null.stats.chi2 is not None:
        sig = null.sigma

        def draw():
            return f0 + rng.normal(size=f0.size) * sig
    else:
        s = null.stats.s_res

        def draw():
            return f0 + rng.normal(scale=s, size=f0.size)
    t_sim = []
    failed = 0
    for i in range(n_sim):
        if cancel is not None and cancel():
            break
        y = spec.y.copy()
        y[mask] = draw()
        s_i = spec.replace(y=y)
        try:
            r0 = fit(s_i, m_null, opt_null)
            r1 = fit(s_i, m_alt, opt_alt)
            t = _objective(r0) - _objective(r1)
            if np.isfinite(t):
                t_sim.append(t)
            else:
                failed += 1
        except Exception:  # noqa: BLE001
            failed += 1
        if progress is not None:
            progress(i + 1, n_sim)
    t_sim = np.array(t_sim)
    p = float((1 + np.sum(t_sim >= t_obs)) / (len(t_sim) + 1))
    p_nominal = float(sps.chi2.sf(t_obs, dp))
    summary = [f"T_obs = {t_obs:.4g} (Δp = {dp}); simulierte p = {p:.3g} aus {len(t_sim)} Replikaten "
               f"({failed} fehlgeschlagen)",
               f"nominell χ²({dp}): p = {p_nominal:.3g} – am Parameterrand nicht gültig, nur zum Vergleich",
               "Test lokal: die zusätzliche Komponente startet an ihrer gefitteten Position."]
    return {"t_obs": float(t_obs), "t_sim": t_sim.tolist(), "p_value": p, "p_nominal": p_nominal, "dp": dp,
            "n": int(len(t_sim)), "failed": failed, "summary": summary}
