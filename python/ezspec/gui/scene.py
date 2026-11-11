"""What the plot shows for a dataset, as plain data (no Qt).

:func:`build_scene` turns a dataset, its pipeline run and the selected step
into a :class:`Scene`: the curves, the labels and the interactive tools that
belong to them. The plot widget only draws scenes, so the same description
can be rendered by several views of the same data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import units as U
from ..export.figure import curves_for
from ..ops.baseline import anchor_values
from ..slices import grouped_by_slices, slice_curves, slice_variable
from ..sweeps import display_order, display_order_masked
from ..view import ViewState, display_transforms, y_label_suffix
from . import peak_edit

_EMPTY = (np.array([]), np.array([]))


@dataclass
class Overlay:
    """Another dataset drawn together with the active one: its processed data in its colour."""
    id: str
    name: str
    color: str
    x: np.ndarray
    y: np.ndarray
    scatter: bool = False


@dataclass
class Scene:
    """Curves are ``(x, y)`` tuples in drawing order, ``None`` when absent."""
    name: str = ""                       # dataset name (legend entry of the data among overlays)
    color: str | None = None             # colour of the data curve (None: the default)
    overlays: tuple = ()                 # Overlay per further dataset in the plot
    y_map: tuple = (1.0, 0.0)            # display transform y' = a·y + b that all curves and handles carry
    data: tuple = _EMPTY
    input: tuple | None = None           # input of the selected step
    excluded: tuple | None = None
    baseline: tuple | None = None
    smoothed: tuple | None = None
    fit: tuple | None = None
    fit_stale: bool = False
    components: tuple = ()               # (name, x, y)
    preview: tuple | None = None         # model at its start values
    residuals: tuple | None = None       # (x, y, normalized)
    residual_label: str | None = None    # overrides "Res." / "Res./σ"
    x_label: str = ""
    y_label: str = ""
    scatter: bool = False
    slices: tuple = ()                   # dicts (label, x, y, xd, yd): y(x, v) data per value of v
    # interactive tools
    step: object = None                  # selected pipeline Step
    anchors_eff: tuple | None = None     # effective anchor positions (x, y)
    peaks: tuple = ()                    # dicts {prefix, top: (x, y), width: (x, y) | None}
    fit_range: list | None = None


def data_y(scene, y):
    """Position on the y axis of a (stacked / normalised) scene -> y value of the data."""
    a, b = scene.y_map
    return (y - b) / a


def _disp(s, *arrays):
    """Arrays of spectrum ``s`` in drawing order: acquisition order for
    back-and-forth (multi-sweep) data, so up and down ramps are drawn as
    separate traces instead of a zigzag between them."""
    o = display_order(s)
    if o is None:
        return arrays
    return tuple(np.asarray(a)[o] for a in arrays)


def handle_values(ds, fit_is_current: bool):
    """Parameter values the peak handles show: the fit if it is current, else the start values."""
    if fit_is_current:
        return ds.fit_result.values
    try:
        return ds.model.initial_values()
    except Exception:  # noqa: BLE001 - invalid model: no handles
        return None


def build_scene(ds, run, step_id=None, fit_is_current: bool = False) -> Scene:
    """Scene of dataset ``ds`` with pipeline run ``run``.

    With a step selected: its input and output. Otherwise the processed data
    with fit, components, residuals and peak handles."""
    if ds is None:
        return Scene()
    step = None
    if step_id is not None:
        try:
            step = ds.pipeline.get(step_id)
        except KeyError:
            step = None
    if step is not None:
        return _step_scene(ds, run, step)
    return _result_scene(ds, run, fit_is_current)


def _step_scene(ds, run, step) -> Scene:
    res = run.result(step.id)
    inp = res.input if res.input is not None else ds.raw
    out = res.output
    sc = Scene(x_label=U.PLAIN_LABELS.get(inp.x_unit, inp.x_label), y_label=inp.y_label, step=step)
    if out is None:
        sc.data = _disp(inp, inp.x, inp.y)
    elif step.op.startswith("baseline_"):
        corrected = (out.x, out.y) if step.params.get("subtract", True) else (out.x, out.y - out.aux["baseline"])
        sc.data = _disp(inp, inp.x, inp.y)
        sc.baseline = _disp(out, out.x, out.aux["baseline"])
        sc.residuals = (*_disp(out, *corrected), False)
        sc.residual_label = "corrected"
        if step.op == "baseline_anchors":
            try:
                sc.anchors_eff = anchor_values(inp, step.params.get("anchors", []), step.params.get("window", 3))
            except Exception:  # noqa: BLE001
                sc.anchors_eff = None
    elif "smoothed" in out.aux and "smoothed" not in inp.aux:
        sc.data = _disp(out, out.x, out.y)
        sc.smoothed = _disp(out, out.x, out.aux["smoothed"])
    elif step.op == "estimate_noise" or step.op == "exclude":
        excl = out.exclude if out.exclude is not None else np.zeros(out.n, bool)
        sc.data = _disp(out, out.x, out.y)
        sc.excluded = (out.x[excl], out.y[excl])
    else:
        sc.data = _disp(out, out.x, out.y)
        sc.input = _disp(inp, inp.x, inp.y)
    return sc


def _result_scene(ds, run, fit_is_current) -> Scene:
    s = run.final
    mask = s.fit_mask
    sc = Scene(data=_disp(s, s.x, s.y), x_label=U.PLAIN_LABELS.get(s.x_unit, s.x_label), y_label=s.y_label,
               scatter=s.meta.get("plot_style") == "scatter", fit_stale=not fit_is_current,
               fit_range=ds.fit_options.x_range)
    if (~mask).any():
        sc.excluded = (s.x[~mask], s.y[~mask])
    if "smoothed" in s.aux:
        sc.smoothed = _disp(s, s.x, s.aux["smoothed"])
    r = ds.fit_result
    comps = []
    if r is not None:
        c = curves_for(s, None, r, r._internals.get("model"), n_dense=1500)
        sc.fit = c["fit_dense"][:2]
        comps = [(k.split(":", 1)[1].rstrip("_"), *v[:2]) for k, v in c.items()
                 if k.startswith("component_dense:")]
        if len(comps) < 2:
            comps = []
        rx, ry = r.x, r.normalized_residuals
        o = display_order_masked(s, r.mask) if len(r.mask) == s.n else None
        if o is not None:
            rx, ry = rx[o], ry[o]
        sc.residuals = (rx, ry, r.stats.chi2 is not None)
    if ds.model.components and (r is None or not fit_is_current) and s.n > 1:
        try:
            xd = np.linspace(s.x.min(), s.x.max(), 1500)
            sc.preview = (xd, ds.model.evaluate(xd, ds.model.initial_values()))
        except Exception:  # noqa: BLE001 - invalid model: no preview
            sc.preview = None
    # y(x, v) data such as a calibration surface: one group/curve per value of v
    fit_model = r._internals.get("model") if r is not None else None
    var = slice_variable(s, fit_model if fit_model is not None else ds.model)
    if var is not None:
        sc.slices = tuple(slice_curves(s, fit_model, r.values if r is not None else None, var))
        sc.fit, comps, sc.preview = None, [], None
        if sc.residuals is not None and len(r.mask) == s.n:
            sc.residuals = (*grouped_by_slices(sc.slices, r.mask, r.x, r.normalized_residuals), sc.residuals[2])
    sc.components = tuple(comps)
    vals = handle_values(ds, fit_is_current)
    if vals is not None and ds.model.peaks:
        sc.peaks = tuple(peak_edit.handles(ds.model, vals))
    return sc


# ============================================================================ several datasets
def _map_y(sc: Scene, a: float, b: float) -> Scene:
    """Apply the display transform ``y' = a·y + b`` to everything drawn in the main panel."""
    if a == 1.0 and b == 0.0:
        return sc

    def f(y):
        return a * np.asarray(y, float) + b

    def xy(c):
        return None if c is None else (c[0], f(c[1]))
    for name in ("data", "input", "excluded", "baseline", "smoothed", "fit", "preview", "anchors_eff"):
        setattr(sc, name, xy(getattr(sc, name)))
    sc.components = tuple((n, x, f(y)) for n, x, y in sc.components)
    sc.slices = tuple({**sl, "y": f(sl["y"]), "yd": None if sl.get("yd") is None else f(sl["yd"])}
                      for sl in sc.slices)
    sc.peaks = tuple({**pk, "top": (pk["top"][0], float(f(pk["top"][1]))),
                      "width": None if pk.get("width") is None else (pk["width"][0], float(f(pk["width"][1])))}
                     for pk in sc.peaks)
    sc.y_map = (a, b)
    return sc


def build_view(ds, run, entries=(), view: ViewState | None = None, step_id=None,
               fit_is_current: bool = False) -> Scene:
    """Scene of the active dataset ``ds`` together with the other datasets in the plot.

    ``entries`` lists ``(dataset, run, colour)`` of every dataset shown, in
    drawing order and including ``ds``. With a step selected only the active
    dataset is drawn – a step is edited on one spectrum, untransformed."""
    sc = build_scene(ds, run, step_id, fit_is_current)
    if ds is None:
        return sc
    entries = list(entries)
    sc.name = ds.name
    sc.color = next((c for d, _r, c in entries if d is ds), None)
    if sc.step is not None:
        return sc
    view = view or ViewState()
    if not any(d is ds for d, _r, _c in entries):
        entries.append((ds, run, sc.color))
    transforms = display_transforms([(d.id, r.final) for d, r, _c in entries], view)
    overlays = []
    for d, r, color in entries:
        if d is ds:
            continue
        s = r.final
        a, b = transforms[d.id]
        x, y = _disp(s, s.x, s.y)
        overlays.append(Overlay(d.id, d.name, color, x, a * y + b, s.meta.get("plot_style") == "scatter"))
    sc.overlays = tuple(overlays)
    _map_y(sc, *transforms[ds.id])
    sc.y_label += y_label_suffix(view)
    return sc
