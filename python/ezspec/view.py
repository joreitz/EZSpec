"""How the datasets of a project are looked at (no Qt).

:class:`ViewState` holds what is not part of the analysis but of its
presentation: which spectra stay in the plot besides the active one, their
colours, and whether they are stacked or normalised. It is stored in the
project (``Project.meta['view']``) and never changes data, pipelines or fits:
stacking and normalising are display transforms ``y' = a·y + b`` that are
undone again when a tool reports a position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VIEW_KEY = "view"
_trapezoid = getattr(np, "trapezoid", None) or np.trapz     # numpy < 2

# Colours of the datasets: the first one is the neutral data colour, none is close to the fit red.
PALETTE = ["#2b2b2b", "#3f90da", "#ffa90e", "#2e9c6a", "#832db6", "#a96b59", "#e05fa5", "#717581",
           "#17a7b5", "#b9ac70"]

NORMALIZE = {
    "none": "Off",
    "max": "Maximum = 1",
    "range": "Min–max (0 … 1)",
    "area": "Area = 1",
}


@dataclass
class ViewState:
    shown: list = field(default_factory=list)      # ids of datasets kept in the plot besides the active one
    colors: dict = field(default_factory=dict)     # dataset id -> "#rrggbb"
    stacked: bool = False
    gap: float = 1.0                               # stack spacing in units of the tallest shown spectrum
    normalize: str = "none"                        # key of NORMALIZE
    regions: list = field(default_factory=list)    # zoom regions [[x0, x1], …] shown next to the overview

    def to_dict(self) -> dict:
        return {"shown": list(self.shown), "colors": dict(self.colors), "stacked": bool(self.stacked),
                "gap": float(self.gap), "normalize": self.normalize,
                "regions": [[float(a), float(b)] for a, b in self.regions]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ViewState":
        d = d or {}
        norm = d.get("normalize", "none")
        try:
            gap = float(d.get("gap", 1.0))
        except (TypeError, ValueError):
            gap = 1.0
        regions = []
        for r in d.get("regions", []) or []:
            try:
                a, b = float(r[0]), float(r[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(a) and np.isfinite(b) and a != b:
                regions.append([min(a, b), max(a, b)])
        return cls(shown=[str(i) for i in d.get("shown", []) or []],
                   colors={str(k): str(v) for k, v in (d.get("colors") or {}).items()},
                   stacked=bool(d.get("stacked", False)), gap=gap if np.isfinite(gap) and gap >= 0 else 1.0,
                   normalize=norm if norm in NORMALIZE else "none", regions=regions)

    @property
    def transformed(self) -> bool:
        """True if what is drawn differs from the data (stacked or normalised)."""
        return self.stacked or self.normalize != "none"

    def prune(self, ids) -> None:
        """Forget datasets that no longer exist."""
        ids = set(ids)
        self.shown = [i for i in self.shown if i in ids]
        self.colors = {k: v for k, v in self.colors.items() if k in ids}


def next_color(used) -> str:
    """First palette colour that is not in use; the least used one when all are taken."""
    used = [str(c).lower() for c in used]
    for c in PALETTE:
        if c not in used:
            return c
    return min(PALETTE, key=used.count)


def _values(s):
    """y values that count for scaling: finite and not excluded (a masked spike must not set the scale)."""
    m = s.fit_mask
    if not m.any():
        m = np.isfinite(s.x) & np.isfinite(s.y)
    return s.x[m], s.y[m]


def normalization(s, mode: str) -> tuple:
    """``(a, b)`` so that ``a·y + b`` is the spectrum normalised by ``mode``."""
    x, y = _values(s)
    if mode == "none" or len(y) == 0:
        return 1.0, 0.0
    if mode == "max":
        top = float(np.max(y))
        if top <= 0:
            top = float(np.max(np.abs(y)))
        return (1.0 / top, 0.0) if top > 0 else (1.0, 0.0)
    if mode == "range":
        lo, hi = float(np.min(y)), float(np.max(y))
        return (1.0 / (hi - lo), -lo / (hi - lo)) if hi > lo else (1.0, -lo)
    if mode == "area":
        o = np.argsort(x, kind="stable")
        area = abs(float(_trapezoid(y[o], x[o]))) if len(y) > 1 else 0.0
        return (1.0 / area, 0.0) if area > 0 else (1.0, 0.0)
    raise ValueError(f"unknown normalisation: {mode}")


def display_transforms(spectra, view: ViewState) -> dict:
    """Display transform ``(a, b)`` per dataset: ``y' = a·y + b``.

    ``spectra`` is a list of ``(id, Spectrum)`` in drawing order (bottom of a
    stack first). Without stacking and normalisation every transform is the
    identity."""
    out = {}
    heights = []
    for key, s in spectra:
        a, b = normalization(s, view.normalize)
        out[key] = (a, b)
        _x, y = _values(s)
        heights.append(a * float(np.max(y) - np.min(y)) if len(y) else 0.0)
    if view.stacked and len(spectra) > 1:
        step = view.gap * max(heights, default=0.0)
        for i, (key, _s) in enumerate(spectra):
            a, b = out[key]
            out[key] = (a, b + i * step)
    return out


def y_label_suffix(view: ViewState) -> str:
    parts = []
    if view.normalize != "none":
        parts.append("normalised")
    if view.stacked:
        parts.append("stacked")
    return f" ({', '.join(parts)})" if parts else ""
