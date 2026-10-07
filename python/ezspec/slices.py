"""Slices of y(x, v) data and surface fits for display.

Data with an additional independent variable v (aux ``var:<name>``), e.g.
laser wavelength vs. current at several temperatures, cannot be drawn as one
line over x. They are grouped by v and every group is drawn with the fitted
surface evaluated at the group's v ("slice", e.g. λ(I) at T = 25 °C).

Grouping: if the v values form well separated clusters (set points with a
small read-back scatter) each cluster is a group; otherwise v is split into
quantile bins and the curve of a bin is drawn at the bin mean, which is then
only representative for points close to that value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Group:
    value: float
    mask: np.ndarray
    lo: float
    hi: float
    exact: bool          # True: cluster of (nearly) equal values; False: quantile bin

    def label(self, name: str) -> str:
        if self.exact:
            return f"{name} = {self.value:.4g}"
        return f"{name} ≈ {self.value:.4g} ({self.lo:.4g}…{self.hi:.4g})"


def group_values(v, max_groups: int = 12, n_bins: int = 6) -> list:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if not ok.any():
        return []
    span = float(np.ptp(v[ok]))
    if span == 0:
        return [Group(float(v[ok][0]), ok.copy(), float(v[ok][0]), float(v[ok][0]), True)]
    u = np.sort(v[ok])
    gaps = np.diff(u)
    cut = gaps > max(10 * float(np.median(gaps)), 1e-9 * span)
    edges = np.concatenate([[u[0]], u[1:][cut]])      # first value of every cluster
    k = len(edges)
    if 2 <= k <= max_groups:
        starts = np.concatenate([[0], np.flatnonzero(cut) + 1])
        ends = np.concatenate([starts[1:], [len(u)]])
        within = max(float(u[e - 1] - u[s]) for s, e in zip(starts, ends))
        between = float(np.min(u[starts[1:]] - u[ends[:-1] - 1]))
        if within < 0.5 * between:
            groups = []
            for s, e in zip(starts, ends):
                lo, hi = float(u[s]), float(u[e - 1])
                m = ok & (v >= lo) & (v <= hi)
                groups.append(Group(float(v[m].mean()), m, lo, hi, True))
            return groups
    q = np.unique(np.quantile(u, np.linspace(0, 1, n_bins + 1)))
    groups = []
    for i, (lo, hi) in enumerate(zip(q[:-1], q[1:])):
        m = ok & (v >= lo) & ((v < hi) if i < len(q) - 2 else (v <= hi))
        if m.any():
            groups.append(Group(float(v[m].mean()), m, float(lo), float(hi), False))
    return groups


def slice_variable(spectrum, model=None) -> str | None:
    """The variable to slice by: the single extra variable of the model, or –
    without such a model – the first extra column if x values repeat (several
    curves over the same x grid)."""
    names = [k[4:] for k in spectrum.aux if k.startswith("var:")]
    if not names:
        return None
    if model is not None:
        used = [v for v in model.independent_variables if v in names]
        if len(used) == 1:
            return used[0]
        if used:
            return None
    if spectrum.n and len(np.unique(spectrum.x)) < 0.9 * spectrum.n:
        return names[0]
    return None


def slice_curves(spectrum, model=None, values=None, var=None, n_dense: int = 300) -> list:
    """Groups of the data with the model evaluated along each group's x range.

    Returns dicts with label, value, mask/index (points of the group; index
    sorted by x), x, y, sigma (data of the group) and xd, yd (model curve, or
    None without model/values)."""
    var = var or slice_variable(spectrum, model)
    if var is None:
        return []
    v = np.asarray(spectrum.aux[f"var:{var}"], float)
    out = []
    for g in group_values(v):
        idx = np.flatnonzero(g.mask)
        idx = idx[np.argsort(spectrum.x[idx], kind="stable")]
        x = spectrum.x[idx]
        item = {"label": g.label(var), "value": g.value, "var": var, "mask": g.mask, "index": idx,
                "x": x, "y": spectrum.y[idx],
                "sigma": None if spectrum.sigma is None else spectrum.sigma[idx], "xd": None, "yd": None}
        if model is not None and values is not None and len(x) > 0:
            others = [n for n in model.independent_variables if n != var]
            if not others:
                lo, hi = float(x.min()), float(x.max())
                xd = np.linspace(lo, hi, n_dense) if hi > lo else np.array([lo])
                try:
                    item["xd"] = xd
                    item["yd"] = np.asarray(model.evaluate(xd, values, {var: np.full(xd.shape, g.value)}), float)
                except Exception:  # noqa: BLE001 - display only
                    item["xd"] = item["yd"] = None
        out.append(item)
    return out


def grouped_by_slices(slices, mask, x, y):
    """Fit arrays (x, y of the points ``mask``) reordered group by group, sorted
    by x within a group and separated by NaN so lines break between groups."""
    pos = np.full(len(mask), -1)
    pos[np.flatnonzero(mask)] = np.arange(int(np.sum(mask)))
    xs, ys = [], []
    for sl in slices:
        p = pos[sl["index"]]
        p = p[p >= 0]
        if len(p):
            xs += [x[p], [np.nan]]
            ys += [y[p], [np.nan]]
    if not xs:
        return x, y
    return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])
