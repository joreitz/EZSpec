"""Systematic uncertainty from the choice of baseline.

A baseline subtracted before the fit is treated as exact by the fit, so its
uncertainty is not part of the statistical errors. Peak areas in particular
depend directly on the baseline (Liland 2015). This module repeats
processing and fit with the baseline parameter varied (lambda x 0.3 / 1 / 3
for Whittaker methods, half window x 0.7 / 1 / 1.4 for SNIP, polynomial
order -1 / 0 / +1) and reports the spread as a *separate*, systematic
uncertainty: half the range of the variants.
"""

from __future__ import annotations

import numpy as np

from ..pipeline import Pipeline
from .engine import fit

VARIATIONS = {
    "baseline_asls": ("lam", "factor", (0.3, 1.0, 3.0)),
    "baseline_arpls": ("lam", "factor", (0.3, 1.0, 3.0)),
    "baseline_snip": ("max_half_window", "factor", (0.7, 1.0, 1.4)),
    "baseline_polynomial": ("order", "offset", (-1, 0, 1)),
}


def baseline_systematics(raw, pipeline: Pipeline, model, options=None, step_id: str | None = None,
                         variations=None) -> dict:
    """Refit with varied baseline parameters; returns per-parameter values and the systematic error."""
    steps = [s for s in pipeline if s.enabled and s.op in VARIATIONS]
    if step_id is not None:
        steps = [s for s in steps if s.id == step_id]
    if not steps:
        raise ValueError("keine variierbare Baseline (AsLS, arPLS, SNIP, Polynom) in der Pipeline")
    step = steps[-1]
    pname, mode, values = VARIATIONS[step.op]
    if variations is not None:
        values = tuple(variations)
    nominal = step.params[pname]
    variants = []
    for v in values:
        new = nominal * v if mode == "factor" else nominal + v
        if pname in ("max_half_window", "order"):
            new = int(round(new))
            if new < (0 if pname == "order" else 1):
                continue
        p = pipeline.copy()
        p.update(step.id, {pname: new})
        run = p.run(raw)
        if not run.ok:
            continue
        try:
            r = fit(run.final, model, options)
        except Exception:  # noqa: BLE001 - a failing variant is reported as missing
            continue
        variants.append((new, r))
    if len(variants) < 2:
        raise ValueError("zu wenige erfolgreiche Varianten")
    names = list(variants[0][1].params)
    dnames = [f"{d.component}.{d.name}" for d in variants[0][1].derived]
    table = {}
    for n in names + dnames:
        vals = []
        for _, r in variants:
            if n in r.params:
                vals.append(r.params[n].value)
            else:
                comp, _, key = n.partition(".")
                vals.append(r.derived_table().get(comp, {}).get(key, (np.nan, None))[0])
        vals = np.array(vals, float)
        table[n] = {"values": vals.tolist(), "systematic": float(0.5 * (np.nanmax(vals) - np.nanmin(vals)))}
    summary = [f"Baseline-Systematik: {step.title}, {pname} ∈ {[v for v, _ in variants]} "
               f"(nominal {nominal}); systematischer Fehler = halbe Spannweite"]
    nominal_r = next((r for v, r in variants if v == nominal), variants[len(variants) // 2][1])
    for n in names + dnames:
        stat = None
        if n in nominal_r.params:
            stat = nominal_r.params[n].stderr
        else:
            comp, _, key = n.partition(".")
            stat = nominal_r.derived_table().get(comp, {}).get(key, (None, None))[1]
        syst = table[n]["systematic"]
        if stat:
            summary.append(f"{n}: ± {stat:.3g} (stat.) ± {syst:.3g} (syst., Baseline)")
        else:
            summary.append(f"{n}: ± {syst:.3g} (syst., Baseline)")
    return {"step": step.id, "parameter": pname, "variants": [v for v, _ in variants], "table": table,
            "summary": summary}
