"""Mapping between peak components and draggable plot handles."""

from __future__ import annotations

import numpy as np

WIDTH_KEYS = {
    "gaussian": ("fwhm",), "lorentzian": ("fwhm",), "pseudo_voigt": ("fwhm",), "pearson7": ("fwhm",),
    "voigt": ("fwhm_g", "fwhm_l"), "tch_pseudo_voigt": ("fwhm_g", "fwhm_l"), "emg": ("fwhm_g", "tau"),
}


def _center_key(comp):
    return "mu" if comp.kind == "emg" else "center"


def _others(model, comp, values, x):
    total = model.evaluate(np.atleast_1d(x), values)
    own = comp.evaluate(np.atleast_1d(x), values)
    return float(total[0] - own[0])


def handles(model, values: dict) -> list:
    """Top and width handle positions for each peak component."""
    out = []
    for comp in model.peaks:
        try:
            d = comp.derived(values)
            xc = d["center"]
            h = d["height"]
            fw = d["fwhm"]
            base = _others(model, comp, values, xc)
            top = (xc, base + h)
            width = (xc + 0.5 * fw, base + 0.5 * h) if np.isfinite(fw) and fw > 0 else None
        except Exception:  # noqa: BLE001 - invalid start values: no handle
            continue
        out.append({"prefix": comp.prefix, "top": top, "width": width})
    return out


def drag(model, comp, values: dict, handle: str, x: float, y: float) -> dict:
    """New local start values for ``comp`` after dragging a handle to (x, y)."""
    loc = {n: values[comp.full_name(n)] for n in comp.local_names()}
    d = comp.derived(values)
    new = dict(loc)
    if handle == "top":
        shift = x - d["center"]
        new[_center_key(comp)] = loc[_center_key(comp)] + shift
        moved = dict(values)
        moved[comp.full_name(_center_key(comp))] = new[_center_key(comp)]
        base = _others(model, comp, moved, x)
        h_new = y - base
        if d["height"] != 0 and np.isfinite(h_new):
            new["area"] = loc["area"] * h_new / d["height"]
    else:
        fw_new = 2.0 * (x - d["center"])
        if fw_new > 0 and d["fwhm"] > 0:
            k = fw_new / d["fwhm"]
            for key in WIDTH_KEYS[comp.kind]:
                new[key] = loc[key] * k
            new["area"] = loc["area"] * k          # pure width scaling keeps the height
    return new
