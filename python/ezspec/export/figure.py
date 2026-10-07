"""Publication figures from a declarative, JSON-serialisable figure specification.

The interactive plot (pyqtgraph) is for working; the publication figure is
rendered by matplotlib from a *FigureSpec* – a plain dict. Every edit in the
GUI is a ``set_path(spec, "panels.0.xlabel", value)`` call, which makes edits
undoable and the figure reproducible from a script.

Sizes are stored in mm, fonts and lines in pt. PDF/SVG are vector output with
embedded TrueType fonts (``pdf.fonttype = 42``, text stays editable).
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from .. import units as U

MM = 1.0 / 25.4

# Column widths as published in author guidelines (no guarantee – check the
# current guide of the journal before submission).
PRESETS = {
    "elsevier_1": {"title": "Elsevier – 1 column (90 mm)", "width_mm": 90, "height_mm": 68, "font_size": 8},
    "elsevier_15": {"title": "Elsevier – 1.5 columns (140 mm)", "width_mm": 140, "height_mm": 95, "font_size": 8},
    "elsevier_2": {"title": "Elsevier – 2 columns (190 mm)", "width_mm": 190, "height_mm": 110, "font_size": 8},
    "acs_1": {"title": "ACS – 1 column (3.25 in)", "width_mm": 82.55, "height_mm": 62, "font_size": 8},
    "acs_2": {"title": "ACS – 2 columns (7 in)", "width_mm": 177.8, "height_mm": 100, "font_size": 8},
    "aps_1": {"title": "APS/AIP – 1 column (8.6 cm)", "width_mm": 86, "height_mm": 65, "font_size": 8},
    "aps_2": {"title": "APS/AIP – 2 columns (17.8 cm)", "width_mm": 178, "height_mm": 100, "font_size": 8},
    "nature_1": {"title": "Nature – 1 column (89 mm)", "width_mm": 89, "height_mm": 67, "font_size": 7},
    "nature_2": {"title": "Nature – 2 columns (183 mm)", "width_mm": 183, "height_mm": 100, "font_size": 7},
    "presentation": {"title": "Presentation 16:9", "width_mm": 254, "height_mm": 142.9, "font_size": 16},
}
PRESET_NOTE = "Dimensions as given in the author guidelines (without guarantee) – check the current requirements before submission."

_FALLBACK_CYCLE = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300",
                   "#b9ac70", "#717581", "#92dadd"]


def color_cycle(name: str = "petroff10") -> list:
    """Colour cycle; 'petroff10' is the colour-vision-deficiency-safe cycle shipped with matplotlib."""
    import matplotlib.style as mstyle
    try:
        cyc = mstyle.library[name]["axes.prop_cycle"]
        return [c["color"] for c in cyc]
    except (KeyError, TypeError):
        if name == "tab10":
            import matplotlib
            return [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["tab10"].colors]
        return list(_FALLBACK_CYCLE)


# ============================================================================ spec helpers
def new_spec(preset: str = "elsevier_1") -> dict:
    p = PRESETS.get(preset, PRESETS["elsevier_1"])
    return {
        "format": "ezspec.figure", "version": 1, "preset": preset,
        "width_mm": p["width_mm"], "height_mm": p["height_mm"], "font_size": p["font_size"],
        "font_family": "sans-serif", "line_width": 1.0, "axes_line_width": 0.6, "tick_direction": "in",
        "colors": "petroff10", "panel_labels": False, "dpi": 600, "title": "",
        "panels": [],
    }


def new_panel(height_ratio: float = 3.0, **kw) -> dict:
    p = {"height_ratio": height_ratio, "xlabel": "", "ylabel": "", "xlim": None, "ylim": None,
         "xscale": "linear", "yscale": "linear", "invert_x": False, "legend": "best", "grid": False,
         "minor_ticks": True, "zero_line": False, "secondary_x": None, "traces": [], "annotations": [],
         "param_box": None}
    p.update(kw)
    return p


def new_trace(source: str, curve: str, kind: str = "line", **kw) -> dict:
    t = {"source": source, "curve": curve, "kind": kind, "label": "", "color": None, "lw": None,
         "ls": "-", "marker": "o", "ms": 2.0, "alpha": 1.0, "zorder": None, "visible": True,
         "errorbars": False, "offset": 0.0, "scale": 1.0}
    t.update(kw)
    return t


def _split(path: str):
    return [int(p) if p.isdigit() else p for p in path.split(".")]


def get_path(spec, path: str):
    node = spec
    for key in _split(path):
        node = node[key]
    return node


def set_path(spec, path: str, value):
    """Set a nested value; returns the previous value (for undo)."""
    keys = _split(path)
    node = spec
    for key in keys[:-1]:
        node = node[key]
    old = copy.deepcopy(node[keys[-1]]) if (isinstance(node, list) and keys[-1] < len(node)) or \
        (isinstance(node, dict) and keys[-1] in node) else None
    node[keys[-1]] = value
    return old


# ============================================================================ curves
def curves_for(processed, raw=None, result=None, model=None, n_dense: int = 2000) -> dict:
    """All plottable curves of one dataset: name -> (x, y, yerr or None)."""
    from ..sweeps import display_order, display_order_masked

    def ordered(order, *arrays):
        # back-and-forth (multi-sweep) data are listed in acquisition order so
        # that line traces follow the ramps instead of zigzagging between them
        if order is None:
            return arrays
        return tuple(None if a is None else np.asarray(a)[order] for a in arrays)

    c = {}
    po = display_order(processed)
    c["processed"] = ordered(po, processed.x, processed.y, processed.sigma)
    if raw is not None:
        c["raw"] = ordered(display_order(raw), raw.x, raw.y, raw.sigma)
    for k in ("baseline", "baseline_total", "smoothed"):
        if k in processed.aux:
            c[k] = ordered(po, processed.x, processed.aux[k], None)
    if "baseline_total" in processed.aux:
        c["processed_plus_baseline"] = ordered(po, processed.x, processed.y + processed.aux["baseline_total"], None)
    if result is not None:
        ro = display_order_masked(processed, result.mask) if len(result.mask) == processed.n else None
        c["fit"] = ordered(ro, result.x, result.best_fit, None)
        c["residuals"] = ordered(ro, result.x, result.residuals, None)
        c["normalized_residuals"] = ordered(ro, result.x, result.normalized_residuals, None)
        for k, v in result.components.items():
            c[f"component:{k}"] = ordered(ro, result.x, v, None)
        mdl = model if model is not None else result._internals.get("model")
        if mdl is not None and not mdl.independent_variables and len(result.x) > 1:
            xd = np.linspace(result.x.min(), result.x.max(), max(n_dense, 4 * len(result.x)))
            vals = result.values
            c["fit_dense"] = (xd, mdl.evaluate(xd, vals), None)
            for k, v in mdl.evaluate_components(xd, vals).items():
                c[f"component_dense:{k}"] = (xd, v, None)
        else:
            c["fit_dense"] = c["fit"]
    # y(x, v) data (e.g. a calibration surface): one curve per value of v
    from ..slices import slice_curves
    mdl = model if model is not None else (result._internals.get("model") if result is not None else None)
    for it in slice_curves(processed, mdl, result.values if result is not None else None):
        c[f"slice_data:{it['label']}"] = (it["x"], it["y"], it["sigma"])
        if it["xd"] is not None:
            c[f"slice_fit:{it['label']}"] = (it["xd"], it["yd"], None)
    return c


def make_resolver(curves: dict) -> Callable:
    """curves: source id -> curves dict (from :func:`curves_for`)."""
    def resolve(source, curve):
        try:
            return curves[source][curve]
        except KeyError:
            raise KeyError(f"curve {curve!r} of {source!r} not available") from None
    return resolve


def default_spec(source: str, curves: dict, x_label: str = "", y_label: str = "", preset: str = "elsevier_1",
                 residuals: bool = True, components: bool = True, normalized: bool = True,
                 name: str = "Data") -> dict:
    spec = new_spec(preset)
    main = new_panel(3.0, xlabel=x_label, ylabel=y_label)
    n = len(curves["processed"][0])
    slices = [k.split(":", 1)[1] for k in curves if k.startswith("slice_data:")]
    if slices:
        cyc = color_cycle(spec.get("colors", "petroff10"))
        for i, lab in enumerate(slices):
            col = cyc[i % len(cyc)]
            fit_key = f"slice_fit:{lab}"
            main["traces"].append(new_trace(source, f"slice_data:{lab}", "scatter", color=col, ms=2.5, zorder=1,
                                            label="" if fit_key in curves else lab))
            if fit_key in curves:
                main["traces"].append(new_trace(source, fit_key, "line", label=lab, color=col, lw=1.0, zorder=3))
    else:
        main["traces"].append(new_trace(source, "processed", "scatter" if n <= 400 else "line", label=name,
                                        color="#555555", lw=0.8, ms=2.0, zorder=1))
    if "fit" in curves and not slices:
        comps = [k for k in curves if k.startswith("component_dense:")]
        if components and len(comps) > 1:
            for k in comps:
                main["traces"].append(new_trace(source, k, "fill", label=k.split(":", 1)[1].rstrip("_"),
                                                alpha=0.35, zorder=2))
        main["traces"].append(new_trace(source, "fit_dense", "line", label="Fit", color="#bd1f01", lw=1.2,
                                        zorder=3))
    spec["panels"].append(main)
    if residuals and "fit" in curves:
        main["xlabel"] = ""
        rk = "normalized_residuals" if normalized else "residuals"
        rp = new_panel(1.0, xlabel=x_label, ylabel="Res./σ" if normalized else "Residuals", legend=None,
                       zero_line=True, minor_ticks=True)
        rp["traces"].append(new_trace(source, rk, "scatter" if n <= 400 or slices else "line", color="#555555",
                                      lw=0.6, ms=1.5))
        spec["panels"].append(rp)
    return spec


def axis_label_for(spectrum) -> str:
    return U.AXIS_LABELS.get(spectrum.x_unit, spectrum.x_label or "x")


# ============================================================================ rendering
def _rc(spec: dict) -> dict:
    fs = spec["font_size"]
    cyc = color_cycle(spec.get("colors", "petroff10"))
    from cycler import cycler
    serif = spec.get("font_family") == "serif"
    return {
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs, "legend.fontsize": fs * 0.9,
        "xtick.labelsize": fs * 0.9, "ytick.labelsize": fs * 0.9, "font.family": spec.get("font_family"),
        "mathtext.fontset": "stix" if serif else "dejavusans",
        "lines.linewidth": spec["line_width"], "axes.linewidth": spec["axes_line_width"],
        "xtick.major.width": spec["axes_line_width"], "ytick.major.width": spec["axes_line_width"],
        "xtick.minor.width": spec["axes_line_width"] * 0.7, "ytick.minor.width": spec["axes_line_width"] * 0.7,
        "xtick.direction": spec["tick_direction"], "ytick.direction": spec["tick_direction"],
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "axes.prop_cycle": cycler(color=cyc), "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "ezspec",
        "savefig.dpi": spec.get("dpi", 600), "figure.dpi": 100,
    }


def _secondary(ax, sec):
    src, dst, laser = sec["from"], sec["to"], sec.get("laser_nm")

    def fwd(v):
        with np.errstate(all="ignore"):
            return U.convert(np.asarray(v, float), src, dst, laser)

    def inv(v):
        with np.errstate(all="ignore"):
            return U.convert(np.asarray(v, float), dst, src, laser)
    sx = ax.secondary_xaxis("top", functions=(fwd, inv))
    sx.set_xlabel(sec.get("label") or U.AXIS_LABELS.get(dst, dst))
    return sx


def _draw_panel(ax, panel, resolve, cycle):
    ci = 0
    for tr in panel["traces"]:
        if not tr.get("visible", True):
            continue
        x, y, err = resolve(tr["source"], tr["curve"])
        y = np.asarray(y, float) * tr.get("scale", 1.0) + tr.get("offset", 0.0)
        color = tr.get("color")
        if color is None:
            color = cycle[ci % len(cycle)]
            ci += 1
        kw = {"color": color, "alpha": tr.get("alpha", 1.0), "label": tr.get("label") or None}
        if tr.get("zorder") is not None:
            kw["zorder"] = tr["zorder"]
        kind = tr.get("kind", "line")
        lw = tr.get("lw")
        if kind == "line":
            ax.plot(x, y, ls=tr.get("ls", "-"), lw=lw, **kw)
        elif kind == "scatter":
            if tr.get("errorbars") and err is not None:
                ax.errorbar(x, y, yerr=np.asarray(err) * tr.get("scale", 1.0), fmt=tr.get("marker", "o"),
                            ms=tr.get("ms", 2.0), elinewidth=0.5, capsize=0, **kw)
            else:
                ax.plot(x, y, ls="none", marker=tr.get("marker", "o"), ms=tr.get("ms", 2.0), mew=0, **kw)
        elif kind == "fill":
            ax.fill_between(x, tr.get("offset", 0.0), y, lw=0, **kw)
            ax.plot(x, y, lw=(lw or 0.6), color=color, alpha=min(1.0, kw["alpha"] + 0.4))
        elif kind == "step":
            ax.step(x, y, where="mid", lw=lw, **kw)
        else:
            raise ValueError(f"unknown trace kind {kind!r}")
    if panel.get("zero_line"):
        ax.axhline(0.0, color="0.4", lw=0.5, zorder=0)
    ax.set_xlabel(panel.get("xlabel", ""))
    ax.set_ylabel(panel.get("ylabel", ""))
    ax.set_xscale(panel.get("xscale", "linear"))
    ax.set_yscale(panel.get("yscale", "linear"))
    if panel.get("xlim"):
        ax.set_xlim(*panel["xlim"])
    if panel.get("ylim"):
        ax.set_ylim(*panel["ylim"])
    if panel.get("invert_x") and not ax.xaxis_inverted():
        ax.invert_xaxis()
    if panel.get("minor_ticks"):
        ax.minorticks_on()
    if panel.get("grid"):
        ax.grid(True, lw=0.3, alpha=0.5)
    if panel.get("legend") and any(t.get("label") for t in panel["traces"] if t.get("visible", True)):
        ax.legend(loc=panel["legend"])
    for a in panel.get("annotations", []):
        coords = ax.transAxes if a.get("coords", "axes") == "axes" else ax.transData
        ax.text(a["x"], a["y"], a["text"], transform=coords, ha=a.get("ha", "left"), va=a.get("va", "top"))
    if panel.get("secondary_x"):
        _secondary(ax, panel["secondary_x"])
    pb = panel.get("param_box")
    if pb and pb.get("lines"):
        ax.text(pb.get("x", 0.97), pb.get("y", 0.95), "\n".join(pb["lines"]), transform=ax.transAxes,
                ha=pb.get("ha", "right"), va="top", fontsize=None,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "lw": 0.5})


def render_figure(spec: dict, resolve: Callable):
    """Build a matplotlib Figure (no pyplot state involved)."""
    import matplotlib
    from matplotlib.figure import Figure
    with matplotlib.rc_context(_rc(spec)):
        fig = Figure(figsize=(spec["width_mm"] * MM, spec["height_mm"] * MM), layout="constrained")
        panels = spec["panels"]
        if not panels:
            fig.add_subplot()
            return fig
        axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False,
                            gridspec_kw={"height_ratios": [p.get("height_ratio", 1.0) for p in panels]})[:, 0]
        cycle = color_cycle(spec.get("colors", "petroff10"))
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            _draw_panel(ax, panel, resolve, cycle)
            if i < len(panels) - 1:
                ax.tick_params(labelbottom=False)
            if spec.get("panel_labels"):
                ax.text(0.01, 0.98, f"({chr(97 + i)})", transform=ax.transAxes, ha="left", va="top",
                        fontweight="bold")
        if spec.get("title"):
            fig.suptitle(spec["title"])
        fig._ezspec_rc = _rc(spec)
    return fig


def save_figure(spec: dict, resolve: Callable, path, fmt: str | None = None) -> None:
    """Render and write PDF/SVG/PNG/PGF/EPS (format from the file suffix by default)."""
    import matplotlib
    fig = render_figure(spec, resolve)
    with matplotlib.rc_context(fig._ezspec_rc):
        fig.savefig(path, format=fmt, dpi=spec.get("dpi", 600), metadata=_metadata(path, fmt))


def _metadata(path, fmt):
    ext = (fmt or str(path).rsplit(".", 1)[-1]).lower()
    if ext == "pdf":
        return {"Creator": "EZSpec", "CreationDate": None}
    if ext == "svg":
        return {"Creator": "EZSpec", "Date": None}
    if ext == "png":
        return {"Software": "EZSpec"}
    return None


def param_box_lines(result, names: list, labels: dict | None = None) -> list:
    """Formatted 'name = value ± error' lines for a parameter box."""
    from ..fit.result import fmt_value
    out = []
    table = result.derived_table()
    for n in names:
        if n in result.params:
            p = result.params[n]
            v, e = p.value, p.stderr
        else:
            comp, _, key = n.partition(".")
            v, e = table[comp][key]
        lab = (labels or {}).get(n, n)
        out.append(f"{lab} = {fmt_value(v, e)}")
    return out
