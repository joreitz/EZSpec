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
