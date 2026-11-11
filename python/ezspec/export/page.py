"""Pages: views of the project arranged freely on a sheet and exported together.

A *page* is a plain dict (stored in ``Project.meta['pages']``) with a size in
mm and a list of *frames*. A frame is a captured view – which dataset is
active, which ones are overlaid, stacking/normalisation, the x range – and a
rectangle on the page. Frames refer to the datasets instead of copying their
curves, so a page follows the analysis: after a new fit the page shows the
new fit.

The rectangle of a frame is its outer box: tick and axis labels are laid out
inside it, so frames that touch do not overlap.
"""

from __future__ import annotations

import uuid
from typing import Callable

import numpy as np

from ..view import NORMALIZE, ViewState
from .figure import MM, _metadata

PAGES_KEY = "pages"
PAGE_SIZES = {
    "a4_landscape": ("A4 landscape", 297.0, 210.0),
    "a4_portrait": ("A4 portrait", 210.0, 297.0),
    "col_1": ("1 column (90 mm wide)", 90.0, 130.0),
    "col_2": ("2 columns (190 mm wide)", 190.0, 130.0),
    "slide": ("Presentation 16:9", 254.0, 142.9),
}
MARGIN = 10.0            # mm kept free at the page border when frames are placed automatically
MIN_SIZE = 20.0          # mm, smallest frame
FIT_COLOR = "#bd1f01"
COMPONENT_COLORS = ["#3f90da", "#ffa90e", "#832db6", "#a96b59", "#94a4a2", "#e76300", "#b9ac70",
                    "#717581", "#92dadd", "#bd1f01"]


def new_page(name: str = "Page 1", size: str = "a4_landscape") -> dict:
    _title, w, h = PAGE_SIZES.get(size, PAGE_SIZES["a4_landscape"])
    return {"format": "ezspec.page", "version": 1, "name": name, "size": size, "width_mm": w, "height_mm": h,
            "font_size": 8.0, "font_family": "sans-serif", "line_width": 1.0, "panel_labels": False, "dpi": 600,
            "frames": []}


def new_frame(active: str, shown=(), rect=(MARGIN, MARGIN, 90.0, 65.0), view: ViewState | None = None,
              **options) -> dict:
    """A view of the dataset ``active`` (overlaid with ``shown``) in the box ``rect`` = [x, y, w, h]
    in mm from the top left corner of the page."""
    view = view or ViewState()
    frame = {"id": uuid.uuid4().hex[:8], "rect": [float(v) for v in rect], "active": active,
             "shown": [i for i in shown if i != active], "step": None,
             "stacked": bool(view.stacked), "gap": float(view.gap), "normalize": view.normalize,
             "x_range": None, "y_range": None, "fit": True, "components": True, "legend": False,
             "labels": True, "title": ""}
    frame.update(options)
    return frame


def check_page(page: dict) -> dict:
    """Fill in what an older or hand-edited page lacks (pages are plain JSON in the project file)."""
    base = new_page()
    for key, value in base.items():
        page.setdefault(key, value)
    frames = []
    for f in page.get("frames") or []:
        if not isinstance(f, dict) or not f.get("active") or len(f.get("rect", [])) != 4:
            continue
        full = new_frame(f["active"])
        full.update(f)
        if full["normalize"] not in NORMALIZE:
            full["normalize"] = "none"
        frames.append(full)
    page["frames"] = frames
    return page


# ============================================================================ placing frames
def _overlaps(rect, others, pad=1.0) -> bool:
    x, y, w, h = rect
    return any(x < ox + ow + pad and ox < x + w + pad and y < oy + oh + pad and oy < y + h + pad
               for ox, oy, ow, oh in others)


def free_rect(page: dict, w: float, h: float) -> list:
    """Where a new box of ``w`` × ``h`` mm goes: the first free place, row by row."""
    W, H = page["width_mm"], page["height_mm"]
    w, h = min(w, W - 2 * MARGIN), min(h, H - 2 * MARGIN)
    taken = [f["rect"] for f in page["frames"]]
    step = 5.0
    y = MARGIN
    while y + h <= H - MARGIN + 1e-9:
        x = MARGIN
        while x + w <= W - MARGIN + 1e-9:
            if not _overlaps([x, y, w, h], taken):
                return [x, y, w, h]
            x += step
        y += step
    k = len(page["frames"]) % 8                      # the page is full: cascade
    return [MARGIN + 6.0 * k, MARGIN + 6.0 * k, w, h]


def capture(page: dict, active: str, shown=(), view: ViewState | None = None, step=None, x_range=None,
            y_range=None, regions=()) -> list:
    """Add the view to the page: one frame, plus one per zoom region placed next to it as in the
    plot (overview on the left, enlarged regions on the right). Returns the new frames."""
    view = view or ViewState()
    regions = [list(r) for r in regions] if step is None else []
    W = page["width_mm"]
    group_w = min(W - 2 * MARGIN, 180.0 if regions else 90.0)
    group_h = min(page["height_mm"] - 2 * MARGIN, 68.0 if len(regions) <= 1 else 100.0)
    x0, y0, group_w, group_h = free_rect(page, group_w, group_h)
    common = {"view": view, "step": step}
    frames = []
    if not regions:
        frames.append(new_frame(active, shown, [x0, y0, group_w, group_h], x_range=x_range, y_range=y_range,
                                **common))
    else:
        cols = 1 if len(regions) <= 3 else 2
        rows = -(-len(regions) // cols)
        main_w = group_w * 3.0 / (4.0 + cols)
        frames.append(new_frame(active, shown, [x0, y0, main_w, group_h], x_range=x_range, y_range=y_range,
                                **common))
        cw, ch = (group_w - main_w) / cols, group_h / rows
        for i, (a, b) in enumerate(regions):
            rect = [x0 + main_w + (i % cols) * cw, y0 + (i // cols) * ch, cw, ch]
            frames.append(new_frame(active, shown, rect, x_range=[float(a), float(b)], labels=False, **common))
    page["frames"].extend(frames)
    return frames


# ============================================================================ alignment
def align(frames, how: str) -> None:
    """Align or distribute frames in place. ``how``: left, right, top, bottom, hcenter, vcenter,
    width, height (same as the first), hspace, vspace (equal gaps)."""
    frames = list(frames)
    if len(frames) < 2:
        return
    rects = [f["rect"] for f in frames]
    if how == "left":
        v = min(r[0] for r in rects)
        for r in rects:
            r[0] = v
    elif how == "right":
        v = max(r[0] + r[2] for r in rects)
        for r in rects:
            r[0] = v - r[2]
    elif how == "top":
        v = min(r[1] for r in rects)
        for r in rects:
            r[1] = v
    elif how == "bottom":
        v = max(r[1] + r[3] for r in rects)
        for r in rects:
            r[1] = v - r[3]
    elif how == "hcenter":
        v = float(np.mean([r[0] + r[2] / 2 for r in rects]))
        for r in rects:
            r[0] = v - r[2] / 2
    elif how == "vcenter":
        v = float(np.mean([r[1] + r[3] / 2 for r in rects]))
        for r in rects:
            r[1] = v - r[3] / 2
    elif how == "width":
        for r in rects[1:]:
            r[2] = rects[0][2]
    elif how == "height":
        for r in rects[1:]:
            r[3] = rects[0][3]
    elif how in ("hspace", "vspace"):
        i, j = (0, 2) if how == "hspace" else (1, 3)
        order = sorted(rects, key=lambda r: r[i])
        span = order[-1][i] + order[-1][j] - order[0][i]
        gap = (span - sum(r[j] for r in order)) / (len(order) - 1)
        pos = order[0][i]
        for r in order:
            r[i] = pos
            pos += r[j] + gap
    else:
        raise ValueError(f"unknown alignment: {how}")


# ============================================================================ rendering
def frame_scene(frame: dict, resolve: Callable):
    """Scene of a frame. ``resolve(ids)`` returns ``(dataset, run, colour, fit_is_current)`` for the
    datasets that still exist, in project order. ``None`` if the active dataset is gone."""
    from ..gui.scene import build_view
    found = resolve([frame["active"]] + list(frame["shown"]))
    active = next((e for e in found if e[0].id == frame["active"]), None)
    if active is None:
        return None
    view = ViewState(stacked=frame["stacked"], gap=frame["gap"], normalize=frame["normalize"])
    return build_view(active[0], active[1], [e[:3] for e in found], view, frame.get("step"), active[3])


def _rc(page: dict) -> dict:
    fs = page["font_size"]
    serif = page.get("font_family") == "serif"
    return {"font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs, "legend.fontsize": fs * 0.9,
            "xtick.labelsize": fs * 0.9, "ytick.labelsize": fs * 0.9, "font.family": page.get("font_family"),
            "mathtext.fontset": "stix" if serif else "dejavusans", "lines.linewidth": page["line_width"],
            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
            "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "svg.hashsalt": "ezspec", "savefig.dpi": page.get("dpi", 600), "figure.dpi": 100}


def draw_scene(ax, scene, frame: dict) -> None:
    """Draw a scene into matplotlib axes the way the interactive plot shows it."""
    drawn = []                                       # (x, y) of what sets the y range

    def line(xy, scatter=False, **kw):
        x, y = np.asarray(xy[0], float), np.asarray(xy[1], float)
        if scatter:
            kw.pop("lw", None)
            ax.plot(x, y, ls="none", marker="o", ms=2.0, **kw)
        else:
            ax.plot(x, y, **kw)
        drawn.append((x, y))

    lw = ax.figure._ezspec_lw
    for o in scene.overlays:
        line((o.x, o.y), o.scatter, color=o.color, lw=lw * 0.85, label=o.name, zorder=2)
    color = scene.color or "#2b2b2b"
    if scene.input is not None:
        line(scene.input, color="#a0a0a0", lw=lw * 0.8, label="Input", zorder=1)
    if scene.slices:
        for i, sl in enumerate(scene.slices):
            c = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
            line((sl["x"], sl["y"]), True, color=c, label=sl["label"], zorder=3)
            if sl.get("xd") is not None and frame["fit"]:
                line((sl["xd"], sl["yd"]), color=c, lw=lw, zorder=4)
    else:
        line(scene.data, scene.scatter, color=color, lw=lw, zorder=3,
             label=scene.name if scene.overlays or scene.step is None else "Data")
    if scene.excluded is not None and len(scene.excluded[0]):
        ax.plot(*scene.excluded, ls="none", marker="o", ms=1.6, color="#c8c8c8", zorder=2.5)
    if scene.baseline is not None:
        line(scene.baseline, color="#e76300", lw=lw * 1.3, ls="--", label="Baseline", zorder=4)
    if scene.smoothed is not None:
        line(scene.smoothed, color="#3f90da", lw=lw * 1.1, label="Smoothed", zorder=4)
    if frame["fit"]:
        if frame["components"]:
            base = scene.y_map[1]
            for i, (name, cx, cy) in enumerate(scene.components):
                c = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
                ax.fill_between(cx, base, cy, color=c, alpha=0.25, lw=0, zorder=1.5)
                ax.plot(cx, cy, color=c, lw=lw * 0.7, zorder=1.6, label=name if not scene.overlays else None)
        if scene.fit is not None:
            line(scene.fit, color=FIT_COLOR, lw=lw * 1.3, label="Fit", zorder=5,
                 ls="-." if scene.fit_stale else "-")
    xr = frame.get("x_range")
    if xr:
        ax.set_xlim(min(xr), max(xr))
    lo, hi = ax.get_xlim()
    lo, hi = min(lo, hi), max(lo, hi)
    yr = frame.get("y_range")
    if yr:
        ax.set_ylim(min(yr), max(yr))
    else:                                            # like a zoom panel: y follows what is inside the x range
        inside = [y[(x >= lo) & (x <= hi) & np.isfinite(y)] for x, y in drawn]
        inside = [v for v in inside if len(v)]
        if inside:
            y0, y1 = min(float(v.min()) for v in inside), max(float(v.max()) for v in inside)
            pad = 0.06 * (y1 - y0) if y1 > y0 else (abs(y1) * 0.1 or 1.0)
            ax.set_ylim(y0 - pad, y1 + pad)
    if frame["labels"]:
        ax.set_xlabel(_mathtext(scene.x_label))
        ax.set_ylabel(_mathtext(scene.y_label))
    if frame.get("title"):
        ax.set_title(frame["title"])
    if frame["legend"]:
        ax.legend(loc="best")


def _mathtext(label: str) -> str:
    """Axis labels of the interactive plot use Unicode superscripts; fonts of a PDF may lack them."""
    return label.replace("⁻¹", "$^{-1}$").replace("²", "$^{2}$")


def _fit_axes(fig, ax, box) -> None:
    """Shrink the axes until ticks and labels are inside ``box`` (x0, y0, x1, y1 in figure units)."""
    from matplotlib.transforms import Bbox
    ax.set_position(Bbox.from_extents(*box))
    renderer = fig.canvas.get_renderer()
    for _ in range(2):
        tight = ax.get_tightbbox(renderer)
        if tight is None:
            return
        t = tight.transformed(fig.transFigure.inverted())
        p = ax.get_position()
        ext = [p.x0 + max(0.0, box[0] - t.x0), p.y0 + max(0.0, box[1] - t.y0),
               p.x1 - max(0.0, t.x1 - box[2]), p.y1 - max(0.0, t.y1 - box[3])]
        if ext[2] - ext[0] < 0.02 or ext[3] - ext[1] < 0.02:
            return                                   # box too small for its labels: leave it
        ax.set_position(Bbox.from_extents(*ext))


def render_page(page: dict, resolve: Callable, only: dict | None = None):
    """Matplotlib figure of the page (``only``: of a single frame, at the size of the frame)."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rc = _rc(page)
    if only is not None:
        W, H = only["rect"][2], only["rect"][3]
        frames = [dict(only, rect=[0.0, 0.0, W, H])]
    else:
        W, H = page["width_mm"], page["height_mm"]
        frames = page["frames"]
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(W * MM, H * MM))
        FigureCanvasAgg(fig)
        fig._ezspec_rc = rc
        fig._ezspec_lw = page["line_width"]
        gap = 0.8                                    # mm between the box of a frame and its content
        for i, frame in enumerate(frames):
            x, y, w, h = frame["rect"]
            box = [(x + gap) / W, 1.0 - (y + h - gap) / H, (x + w - gap) / W, 1.0 - (y + gap) / H]
            ax = fig.add_axes([box[0], box[1], box[2] - box[0], box[3] - box[1]])
            scene = frame_scene(frame, resolve)
            if scene is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "dataset removed", ha="center", va="center", transform=ax.transAxes,
                        color="#a0a0a0")
                continue
            draw_scene(ax, scene, frame)
            if page.get("panel_labels") and only is None:
                ax.text(0.02, 0.97, f"({chr(97 + i % 26)})", transform=ax.transAxes, ha="left", va="top",
                        fontweight="bold")
            _fit_axes(fig, ax, box)
    return fig


def save_page(page: dict, resolve: Callable, path, fmt: str | None = None) -> None:
    """Write the page as PDF/SVG/PNG/EPS (format from the file suffix by default)."""
    import matplotlib
    fig = render_page(page, resolve)
    with matplotlib.rc_context(fig._ezspec_rc):
        fig.savefig(path, format=fmt, dpi=page.get("dpi", 600), metadata=_metadata(path, fmt))


def project_resolver(project, colors: dict | None = None, fit_is_current: Callable | None = None) -> Callable:
    """Resolver for a project outside the GUI (scripts, tests): pipelines are run on demand."""
    from ..view import PALETTE
    runs = {}

    def resolve(ids):
        out = []
        for k, d in enumerate(project.datasets):
            if d.id in ids:
                if d.id not in runs:
                    runs[d.id] = d.run_pipeline()
                color = (colors or {}).get(d.id, PALETTE[k % len(PALETTE)])
                current = fit_is_current(d) if fit_is_current else d.fit_result is not None
                out.append((d, runs[d.id], color, current))
        return out
    return resolve
