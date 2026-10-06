"""Tabular export of fit results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def parameter_rows(result) -> list:
    rows = []
    for p in result.params.values():
        rows.append({
            "parameter": p.name, "value": p.value, "stderr": "" if p.stderr is None else p.stderr,
            "rel_stderr": "" if p.stderr is None or p.value == 0 else abs(p.stderr / p.value),
            "vary": p.vary, "expr": p.expr or "", "min": p.min, "max": p.max, "init": p.init_value,
            "at_bound": p.at_bound or "", "stderr_reliable": p.reliable_stderr,
            "covariance": result.stats.covariance_mode,
        })
    for d in result.derived:
        rows.append({"parameter": f"{d.component}.{d.name}", "value": d.value,
                     "stderr": "" if d.stderr is None else d.stderr,
                     "rel_stderr": "" if d.stderr is None or d.value == 0 else abs(d.stderr / d.value),
                     "vary": "", "expr": "abgeleitet", "min": "", "max": "", "init": "", "at_bound": "",
                     "stderr_reliable": "", "covariance": result.stats.covariance_mode})
    return rows


def statistics_rows(result) -> list:
    st = result.stats
    rows = [("N", st.n_points), ("p (frei)", st.n_varys), ("nu", st.dof), ("sigma_source", st.sigma_label),
            ("weighting", st.weighting), ("covariance", st.covariance_mode)]
    if st.chi2 is not None:
        rows += [("chi2", st.chi2), ("redchi", st.redchi), ("redchi_band_lo", st.redchi_band[0]),
                 ("redchi_band_hi", st.redchi_band[1]), ("chi2_pvalue_approx", st.chi2_pvalue)]
    rows += [("RSS", st.rss), ("s_res", st.s_res), ("RMSE", st.rmse), ("AIC", st.aic), ("AICc", st.aicc),
             ("BIC", st.bic), ("IC_form", st.ic_form), ("R2_descriptive", st.r2), ("adjR2_descriptive", st.adj_r2),
             ("runs", st.runs.get("runs")), ("runs_expected", st.runs.get("expected")),
             ("runs_p_too_few", st.runs.get("p_too_few")), ("lag1_autocorr", st.lag1_autocorr),
             ("durbin_watson", st.durbin_watson), ("jacobian_condition", st.jacobian_condition)]
    return rows


def write_parameters_csv(result, path, delimiter: str = ",") -> None:
    rows = parameter_rows(result)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)


def write_statistics_csv(result, path, delimiter: str = ",") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(["quantity", "value"])
        w.writerows(statistics_rows(result))


def write_curves_csv(result, path, delimiter: str = ",") -> None:
    cols = {"x": result.x, "y": result.y}
    if result.sigma is not None:
        cols["sigma"] = result.sigma
    cols.update(fit=result.best_fit, residual=result.residuals, normalized_residual=result.normalized_residuals)
    for k, v in result.components.items():
        cols[f"component_{k.rstrip('_')}"] = v
    names = list(cols)
    data = np.column_stack([np.asarray(cols[n], float) for n in names])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(names)
        for row in data:
            w.writerow([repr(float(v)) for v in row])


def write_all(result, stem) -> list:
    """Write report (.txt), parameters, statistics, curves (.csv) and full JSON next to ``stem``."""
    stem = Path(stem)
    out = []
    p = stem.with_name(stem.name + "_report.txt")
    p.write_text(result.report(), encoding="utf-8")
    out.append(p)
    for suffix, fn in (("_parameters.csv", write_parameters_csv), ("_statistics.csv", write_statistics_csv),
                       ("_curves.csv", write_curves_csv)):
        p = stem.with_name(stem.name + suffix)
        fn(result, p)
        out.append(p)
    p = stem.with_name(stem.name + "_fit.json")
    p.write_text(result.to_json(), encoding="utf-8")
    out.append(p)
    return out
