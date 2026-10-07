"""Goodness-of-fit statistics and residual diagnostics.

Conventions (documented because packages differ):

* Known sigma (column, estimate, constant, Poisson model):
  chi2 = sum(((y - f)/sigma)^2), reduced chi2 = chi2/nu with nu = N - p, and
  the 1-sigma band of the reduced chi2 for a correct model, 1 +/- sqrt(2/nu)
  (Andrae et al. 2010 caution that nu is not exactly defined for nonlinear
  models). AIC = chi2 + 2K, BIC = chi2 + K ln N with K = p.
* Unknown sigma (unit weights): no chi-square is reported. RSS, the residual
  standard error s = sqrt(RSS/nu) and RMSE = sqrt(RSS/N). sigma is profiled
  out of the Gaussian likelihood: AIC = N ln(RSS/N) + 2K, BIC = N ln(RSS/N)
  + K ln N with K = p + 1 (sigma counts as an estimated parameter; Burnham &
  Anderson 2002). lmfit uses K = p; AIC differences agree, AICc differs.
* AICc = AIC + 2K(K+1)/(N-K-1); undefined for K >= N - 1.
* Only differences of AIC/BIC between models fitted to *identical* data with
  identical weighting are meaningful.
* R^2 = 1 - RSS/TSS (unweighted) is descriptive only; it is not a valid
  criterion for comparing nonlinear models (Spiess & Neumeyer 2010).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def runs_test(residuals) -> dict:
    """Wald–Wolfowitz runs test on the signs of the residuals (zeros dropped).

    Too few runs indicate systematic deviations (the model misses structure);
    the one-sided p-value for "too few runs" is reported (as GraphPad does).
    Normal approximation; reliable for n+ and n- >= ~10.
    """
    r = np.asarray(residuals, float)
    s = np.sign(r[r != 0])
    n = len(s)
    n_pos = int(np.sum(s > 0))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"runs": 1 if n else 0, "expected": float("nan"), "z": float("nan"),
                "p_too_few": float("nan"), "n_pos": n_pos, "n_neg": n_neg}
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2.0 * n_pos * n_neg / n + 1.0
    var = 2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n) / (n * n * (n - 1.0))
    z = (runs - mu) / math.sqrt(var) if var > 0 else float("nan")
    return {"runs": runs, "expected": mu, "z": z, "p_too_few": float(sps.norm.cdf(z)),
            "n_pos": n_pos, "n_neg": n_neg}


def lag1_autocorrelation(r) -> float:
    r = np.asarray(r, float)
    d = r - r.mean()
    den = np.sum(d * d)
    return float(np.sum(d[1:] * d[:-1]) / den) if den > 0 else float("nan")


def durbin_watson(r) -> float:
    r = np.asarray(r, float)
    den = np.sum(r * r)
    return float(np.sum(np.diff(r) ** 2) / den) if den > 0 else float("nan")


def information_criteria(n: int, p: int, rss: float, chi2: float | None, sigma_known: bool,
                         neg2loglik: float | None = None) -> dict:
    """AIC, AICc, BIC with the form appropriate to the sigma source."""
    if neg2loglik is not None:
        k = p
        base = neg2loglik
        form = "Poisson: −2 ln L + 2K, K = p"
    elif sigma_known:
        k = p
        base = chi2
        form = "σ known: χ² + 2K, K = p"
    else:
        k = p + 1
        base = n * math.log(rss / n) if rss > 0 else float("-inf")
        form = "σ unknown (profiled out): N·ln(RSS/N) + 2K, K = p + 1"
    aic = base + 2 * k
    bic = base + k * math.log(n)
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")
    return {"aic": aic, "aicc": aicc, "bic": bic, "k": k, "form": form}


def r_squared(y, f, p: int) -> dict:
    y = np.asarray(y, float)
    rss = float(np.sum((y - f) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    n = len(y)
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n - p > 0 else float("nan")
    return {"r2": r2, "adj_r2": adj}


def normality(z) -> dict:
    """Skewness, excess kurtosis and D'Agostino–Pearson test of normalised residuals.

    Informative only: with large N tiny deviations become 'significant'.
    """
    z = np.asarray(z, float)
    out = {"skewness": float(sps.skew(z)), "excess_kurtosis": float(sps.kurtosis(z))}
    if len(z) >= 20:
        out["p_normal"] = float(sps.normaltest(z).pvalue)
    else:
        out["p_normal"] = float("nan")
    return out


def qq_points(z) -> tuple:
    """Theoretical vs. sample quantiles for a normal QQ plot."""
    z = np.sort(np.asarray(z, float))
    n = len(z)
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)   # Blom
    return sps.norm.ppf(probs), z


def akaike_weights(values) -> np.ndarray:
    v = np.asarray(values, float)
    d = v - np.nanmin(v)
    w = np.exp(-0.5 * d)
    return w / np.nansum(w)
