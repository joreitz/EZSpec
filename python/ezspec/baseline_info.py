"""Human-readable statement of the baseline used for a result.

Every baseline operation stores a complete *recipe* in ``meta["baselines"]``
(method, all parameters, ranges, effective anchors, polynomial coefficients
with standard errors, convergence). A fit copies these recipes into its
provenance, and the report, the results panel, the exports and figure
parameter boxes state them, so a published result always says which
baseline was subtracted – or that none was, and whether the model contains a
background term instead.
"""

from __future__ import annotations

from .numfmt import fmt_num, fmt_value


def _rng(ranges) -> str:
    return ", ".join(f"[{fmt_num(a)}, {fmt_num(b)}]" for a, b in ranges)


def _unit(rec) -> str:
    u = rec.get("x_unit") or ""
    return f" {u}" if u else ""


def describe_baseline(rec: dict, short: bool = False) -> str:
    """One recipe as text; ``short`` gives a compact form for figure legends/boxes."""
    m = rec.get("method", "?")
    title = rec.get("title") or f"Baseline: {m}"
    u = _unit(rec)
    if m == "polynomial":
        head = f"{title}, order {rec.get('order')} ({rec.get('variant', 'fit')})"
        if short:
            return head
        parts = [head]
        lo, hi = rec.get("domain", [None, None])
        if lo is not None:
            parts.append(f"b(x) = Σ c_k t^k with t = (2x − {fmt_num(lo + hi)})/{fmt_num(hi - lo)}")
        coef, se = rec.get("coefficients") or [], rec.get("coefficient_stderr")
        if coef:
            parts.append(", ".join(f"c{k} = {fmt_value(c, None if se is None else se[k])}"
                                   for k, c in enumerate(coef)))
        if se is not None:
            parts.append("standard errors from the scatter of the data in the baseline ranges")
        elif rec.get("variant") in ("modpoly", "imodpoly"):
            parts.append(f"iterative ({rec.get('iterations')} iterations), no statistical uncertainty")
        if rec.get("ranges"):
            parts.append(f"baseline ranges {_rng(rec['ranges'])}{u}")
        body = "; ".join(parts)
    elif m == "anchors":
        head = f"{title} ({rec.get('interpolation')}, {len(rec.get('anchors', []))} anchors)"
        if short:
            return head
        pts = ", ".join(f"({fmt_num(a)}, {fmt_num(b)})" for a, b in rec.get("anchors", []))
        body = (f"{head}; median window ±{rec.get('window')} points, extrapolation {rec.get('extrapolation')}; "
                f"effective anchors (x, y): {pts}")
    elif m in ("asls", "arpls"):
        lam = f"λ = {fmt_num(rec.get('lam'), 3)}"
        head = f"{title}, {lam}" + (f", p = {fmt_num(rec.get('p'), 3)}" if m == "asls" else "")
        if short:
            return head
        cut = rec.get("cutoff_points")
        cut_x = rec.get("cutoff_x")
        parts = [head, f"difference order {rec.get('diff_order')}",
                 "λ per point (pybaselines convention)"]
        if cut is not None:
            parts.append(f"cut-off period ≈ {fmt_num(cut, 3)} points"
                         + (f" ≈ {fmt_num(cut_x, 3)}{u}" if cut_x is not None else ""))
        conv = "converged" if rec.get("converged") else "NOT converged"
        parts.append(f"{conv} after {rec.get('iterations')} iterations (tol {fmt_num(rec.get('tol'), 3)})")
        if rec.get("exclude_ranges"):
            parts.append(f"excluded ranges {_rng(rec['exclude_ranges'])}{u}")
        if rec.get("force_ranges"):
            parts.append(f"forced ranges {_rng(rec['force_ranges'])}{u}")
        body = "; ".join(parts)
    elif m == "snip":
        hw = rec.get("max_half_window")
        head = f"{title}, max. half-window {hw} points"
        if short:
            return head
        parts = [head + (f" ≈ {fmt_num(rec['half_window_x'], 3)}{u}" if rec.get("half_window_x") else ""),
                 f"filter order {rec.get('filter_order')}",
                 "decreasing windows" if rec.get("decreasing") else "increasing windows"]
        if rec.get("lls"):
            parts.append("LLS transform")
        body = "; ".join(parts)
    else:
        body = title
        if short:
            return body
    if rec.get("x_range"):
        body += f"; data range [{fmt_num(rec['x_range'][0])}, {fmt_num(rec['x_range'][1])}]{u}"
    body += "; subtracted" if rec.get("subtracted", True) else "; computed but NOT subtracted (display only)"
    return body


def background_components(model_spec: dict | None) -> list:
    from .models import COMPONENT_TYPES
    out = []
    for c in (model_spec or {}).get("components", []):
        t = COMPONENT_TYPES.get(c.get("kind"))
        if t is not None and t.category == "baseline":
            out.append(f"{c.get('prefix', '').rstrip('_') or c.get('kind')} ({t.title})")
    return out


def baseline_statement(recipes: list | None, model_spec: dict | None = None) -> list:
    """Lines stating the baseline treatment of a result (always non-empty)."""
    lines = []
    subtracted = [r for r in (recipes or []) if r.get("subtracted", True)]
    for i, rec in enumerate(recipes or [], 1):
        lines.append(f"[{i}] {describe_baseline(rec)}")
    if not subtracted:
        lines.append("No baseline was subtracted before the fit.")
    bg = background_components(model_spec)
    if bg:
        lines.append("Background fitted within the model: " + ", ".join(bg) + " (parameters above).")
    elif not subtracted:
        lines.append("The model contains no background term.")
    if subtracted:
        lines.append("The uncertainty of the subtracted baseline is not included in the parameter errors "
                     "(systematic; see Analysis → Baseline systematics).")
    return lines


def baseline_short(recipes: list | None, model_spec: dict | None = None, prefix: bool = True) -> str:
    """Compact one-line statement (results table, figure parameter box), e.g.
    'Baseline: arPLS (Whittaker), λ = 1.00e+07 (subtracted)'."""
    def strip(t):
        return t[len("Baseline: "):] if t.startswith("Baseline: ") else t
    sub = [strip(describe_baseline(r, short=True)) for r in (recipes or []) if r.get("subtracted", True)]
    bg = background_components(model_spec)
    if sub:
        text = " + ".join(sub) + " (subtracted)" + (f"; background in model: {', '.join(bg)}" if bg else "")
    elif bg:
        text = f"none subtracted; background in model: {', '.join(bg)}"
    else:
        text = "none (no baseline subtracted, no background in the model)"
    return ("Baseline: " + text) if prefix else text
