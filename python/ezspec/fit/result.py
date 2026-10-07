"""Fit result containers and the human-readable report."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ParamResult:
    name: str
    value: float
    stderr: float | None
    init_value: float
    vary: bool
    min: float
    max: float
    expr: str | None = None
    at_bound: str | None = None      # "min" | "max" | None
    near_bound: bool = False         # bound within 2 standard errors

    @property
    def reliable_stderr(self) -> bool:
        return (self.stderr is not None and np.isfinite(self.stderr) and self.at_bound is None
                and not self.near_bound)

    @property
    def rel_stderr(self) -> float:
        if self.stderr is None or self.value == 0:
            return float("nan")
        return abs(self.stderr / self.value)


@dataclass
class DerivedResult:
    component: str
    name: str
    value: float
    stderr: float | None


@dataclass
class FitWarning:
    code: str
    message: str
    severity: str = "warning"        # info | warning | error


@dataclass
class FitStatistics:
    n_points: int
    n_varys: int
    dof: int
    sigma_source: str
    sigma_label: str
    weighting: str
    covariance_mode: str             # absolute | scaled | unavailable
    rss: float
    s_res: float                     # sqrt(RSS / nu)
    rmse: float                      # sqrt(RSS / N)
    r2: float
    adj_r2: float
    aic: float
    aicc: float
    bic: float
    ic_form: str
    ic_k: int
    chi2: float | None = None
    redchi: float | None = None
    redchi_band: tuple | None = None
    chi2_pvalue: float | None = None
    runs: dict = field(default_factory=dict)
    lag1_autocorr: float = float("nan")
    durbin_watson: float = float("nan")
    normality: dict = field(default_factory=dict)
    jacobian_condition: float = float("nan")
    poisson_deviance: float | None = None       # 2 sum[y ln(y/f) - (y - f)], ~ chi2_nu for Poisson data

    @property
    def sigma_known(self) -> bool:
        return self.chi2 is not None


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if isinstance(v, (np.floating,)):
        return _clean(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_clean(a) for a in v.tolist()]
    if isinstance(v, dict):
        return {k: _clean(a) for k, a in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(a) for a in v]
    return v


@dataclass
class FitResult:
    success: bool
    message: str
    method: str
    nfev: int
    params: dict
    var_names: list
    covariance: np.ndarray | None
    correlation: np.ndarray | None
    stats: FitStatistics
    derived: list
    warnings: list
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray | None
    best_fit: np.ndarray
    init_fit: np.ndarray
    residuals: np.ndarray
    normalized_residuals: np.ndarray
    components: dict
    mask: np.ndarray
    model_spec: dict
    options: dict
    data_hash: str
    provenance: dict
    extra: dict = field(default_factory=dict)       # bootstrap / profile CI results
    _internals: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ access
    @property
    def values(self) -> dict:
        return {k: p.value for k, p in self.params.items()}

    def value(self, name):
        return self.params[name].value

    def stderr(self, name):
        return self.params[name].stderr

    def correl(self, a: str, b: str) -> float:
        i, j = self.var_names.index(a), self.var_names.index(b)
        return float(self.correlation[i, j])

    def high_correlations(self, threshold: float = 0.9) -> list:
        out = []
        if self.correlation is None:
            return out
        n = len(self.var_names)
        for i in range(n):
            for j in range(i + 1, n):
                r = self.correlation[i, j]
                if abs(r) > threshold:
                    out.append((self.var_names[i], self.var_names[j], float(r)))
        return sorted(out, key=lambda t: -abs(t[2]))

    def derived_table(self) -> dict:
        out = {}
        for d in self.derived:
            out.setdefault(d.component, {})[d.name] = (d.value, d.stderr)
        return out

    # ------------------------------------------------------------------ export
    def to_dict(self, include_curves: bool = True) -> dict:
        d = {
            "format": "ezspec.fit_result", "version": 1,
            "success": self.success, "message": self.message, "method": self.method, "nfev": self.nfev,
            "params": {k: asdict(p) for k, p in self.params.items()},
            "var_names": list(self.var_names),
            "covariance": None if self.covariance is None else self.covariance,
            "correlation": None if self.correlation is None else self.correlation,
            "statistics": asdict(self.stats),
            "derived": [asdict(a) for a in self.derived],
            "warnings": [asdict(w) for w in self.warnings],
            "model": self.model_spec, "options": self.options, "data_hash": self.data_hash,
            "provenance": self.provenance, "extra": self.extra,
        }
        if include_curves:
            d["curves"] = {"x": self.x, "y": self.y, "sigma": self.sigma, "best_fit": self.best_fit,
                           "residuals": self.residuals, "normalized_residuals": self.normalized_residuals,
                           **{f"component:{k}": v for k, v in self.components.items()}}
        return _clean(d)

    def to_json(self, include_curves: bool = True, indent: int = 1) -> str:
        return json.dumps(self.to_dict(include_curves), indent=indent, ensure_ascii=False)

    def report(self) -> str:
        return format_report(self)

    def __repr__(self):
        return f"<FitResult {self.method} success={self.success} N={self.stats.n_points} p={self.stats.n_varys}>"


# =============================================================================== report
def fmt_value(v: float, e: float | None, digits: int = 2) -> str:
    """Value with uncertainty, rounded to ``digits`` significant digits of the uncertainty."""
    if v is None or not np.isfinite(v):
        return str(v)
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.6g}"
    exp = math.floor(math.log10(e)) - (digits - 1)
    if -5 <= exp <= 5 and abs(v) < 1e7:
        dec = max(0, -exp)
        return f"{round(v, -exp):.{dec}f} ± {round(e, -exp):.{dec}f}"
    return f"{v:.{max(digits, 3)}e} ± {e:.{digits - 1}e}"


def _g(v, spec=".6g"):
    if v is None:
        return "–"
    try:
        if not np.isfinite(v):
            return str(v)
    except TypeError:
        return str(v)
    return format(v, spec)


def format_report(r: FitResult) -> str:
    st = r.stats
    L = []
    L.append("=" * 72)
    L.append("EZSpec fit report")
    L.append("=" * 72)
    L.append(f"Method: {r.method}   Success: {'yes' if r.success else 'NO'}   Function evaluations: {r.nfev}")
    if r.message:
        L.append(f"Message: {r.message}")
    L.append(f"Points N = {st.n_points}   free parameters p = {st.n_varys}   degrees of freedom ν = {st.dof}")
    L.append(f"σ source: {st.sigma_label}   Weighting: {st.weighting}")
    cov = {"absolute": "absolute (σ taken as known)",
           "scaled": "scaled by √χ²_ν or s (σ estimated from the residuals)",
           "unavailable": "NOT available"}[st.covariance_mode]
    L.append(f"Covariance: {cov}")
    L.append("")
    L.append("Parameters")
    L.append("-" * 72)
    for p in r.params.values():
        flag = []
        if not p.vary and not p.expr:
            flag.append("fixed")
        if p.expr:
            flag.append(f"= {p.expr}")
        if p.at_bound:
            flag.append(f"AT BOUND ({p.at_bound}) – SE not reliable")
        elif p.near_bound:
            flag.append("bound < 2 SE away – SE not reliable")
        rel = f" ({100 * p.rel_stderr:.2g} %)" if p.stderr and np.isfinite(p.rel_stderr) and p.rel_stderr < 10 else ""
        L.append(f"  {p.name:<18} {fmt_value(p.value, p.stderr):<30}{rel:<10} {'; '.join(flag)}")
    if r.derived:
        L.append("")
        L.append("Derived quantities (uncertainty propagated with the full covariance)")
        L.append("-" * 72)
        for comp, vals in r.derived_table().items():
            items = ", ".join(f"{k} = {fmt_value(v, e)}" for k, (v, e) in vals.items())
            L.append(f"  {comp}: {items}")
    L.append("")
    L.append("Goodness of fit")
    L.append("-" * 72)
    if st.chi2 is not None:
        lo, hi = st.redchi_band
        L.append(f"  χ² = {_g(st.chi2)}   χ²_ν = {_g(st.redchi, '.4g')}   "
                 f"(expected 1, 1σ band [{lo:.3g}, {hi:.3g}])")
        L.append(f"  P(χ² ≥ observed) ≈ {_g(st.chi2_pvalue, '.3g')}  (approximation: linear, Gaussian, σ exact)")
        if st.poisson_deviance is not None:
            L.append(f"  Poisson deviance D = {_g(st.poisson_deviance)}   D/ν = {_g(st.poisson_deviance / st.dof, '.4g')}"
                     "  (Pearson χ² above)")
    else:
        L.append("  χ²: not defined (σ unknown) – instead:")
    L.append(f"  RSS = {_g(st.rss)}   s = √(RSS/ν) = {_g(st.s_res)}   RMSE = √(RSS/N) = {_g(st.rmse)}")
    L.append(f"  AIC = {_g(st.aic, '.6g')}   AICc = {_g(st.aicc, '.6g')}   BIC = {_g(st.bic, '.6g')}")
    L.append(f"     Form: {st.ic_form}; only differences on identical data are meaningful")
    L.append(f"  R² = {_g(st.r2, '.6f')}   R²_adj = {_g(st.adj_r2, '.6f')}   "
             "(descriptive – not for model comparison)")
    L.append("")
    L.append("Residual diagnostics")
    L.append("-" * 72)
    ru = st.runs
    if ru:
        L.append(f"  Runs test: {ru.get('runs')} runs (expected {_g(ru.get('expected'), '.1f')}), "
                 f"z = {_g(ru.get('z'), '.2f')}, p(too few) = {_g(ru.get('p_too_few'), '.3g')}")
    L.append(f"  Lag-1 autocorrelation = {_g(st.lag1_autocorr, '.3f')}   "
             f"Durbin–Watson ≈ {_g(st.durbin_watson, '.3f')}")
    nm = st.normality
    if nm:
        L.append(f"  Skewness = {_g(nm.get('skewness'), '.3f')}   Excess kurtosis = {_g(nm.get('excess_kurtosis'), '.3f')}"
                 f"   p(normal, D'Agostino) = {_g(nm.get('p_normal'), '.3g')}")
    L.append(f"  Jacobian condition number = {_g(st.jacobian_condition, '.3g')}")
    hc = r.high_correlations()
    if hc:
        L.append("")
        L.append("High correlations |ρ| > 0.9")
        for a, b, c in hc:
            L.append(f"  ρ({a}, {b}) = {c:+.4f}")
    if r.warnings:
        L.append("")
        L.append("Notes")
        L.append("-" * 72)
        for w in r.warnings:
            tag = {"info": "i", "warning": "!", "error": "✗"}.get(w.severity, "!")
            L.append(f"  [{tag}] {w.message}")
    for key, title in (("profile_ci", "Profile likelihood confidence intervals"),
                       ("bootstrap", "Bootstrap"),
                       ("mcmc", "MCMC-Posterior (emcee)"),
                       ("baseline_systematics", "Systematic uncertainty from the baseline")):
        if key in r.extra:
            L.append("")
            L.append(title)
            L.append("-" * 72)
            L.extend("  " + line for line in r.extra[key].get("summary", []))
    L.append("")
    L.append("Model")
    L.append("-" * 72)
    L.extend("  " + line for line in r.provenance.get("model_description", "").splitlines())
    L.append(f"Data hash: {r.data_hash[:16]}…   Backend: {r.provenance.get('backend')}   "
             f"lmfit {r.provenance.get('lmfit')}")
    return "\n".join(L)
