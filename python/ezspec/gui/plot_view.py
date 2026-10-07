"""Interactive plot: data, baseline, fit, components and residuals with editing tools."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .theme import COLORS, COMPONENT_COLORS

MODES = {
    "navigate": ("Navigate", "Zoom (mouse wheel / right-drag) and pan (left-drag)"),
    "anchor": ("Anchor", "Click: baseline anchor (snaps to the data; Shift: free) · Drag: move "
                         "· Right-click: delete"),
    "region": ("Range", "Click: add range (masks, baseline/noise/fit range) · Drag edges to resize "
                        "· Right-click: delete"),
    "peak": ("Peak", "Click: add a peak here · Drag markers: position/height or width"),
}


class AnchorItem(pg.TargetItem):
    sigRemove = QtCore.Signal(object)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton and not self.moving:
            ev.accept()
            self.sigRemove.emit(self)
            return
        super().mouseClickEvent(ev)


class RegionItem(pg.LinearRegionItem):
    sigRemove = QtCore.Signal(object)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.sigRemove.emit(self)
            return
        super().mouseClickEvent(ev)


def _curve(pen, name=None, **kw):
    item = pg.PlotDataItem(pen=pen, name=name, **kw)
    item.setDownsampling(auto=True, method="peak")
    item.setClipToView(True)
    return item


def _set_curve_data(item, x, y):
    """setData with view clipping / downsampling only for x-sorted data: both
    pyqtgraph optimisations assume increasing x and would drop or merge points
    of back-and-forth (multi-sweep) data drawn in acquisition order."""
    x = np.asarray(x)
    y = np.asarray(y)
    ordered = len(x) < 2 or not np.any(np.diff(x) < 0)
    item.setClipToView(ordered)
    if ordered:
        item.setDownsampling(auto=True, method="peak")
    else:
        item.setDownsampling(ds=1, auto=False)
    item.setData(x, y)


class PlotView(QtWidgets.QWidget):
    anchorsEdited = QtCore.Signal(list)
    rangesEdited = QtCore.Signal(str, list)
    rangeEdited = QtCore.Signal(str, list)
    peakAdded = QtCore.Signal(float, float)
    peakDragged = QtCore.Signal(str, str, float, float, bool)   # prefix, handle (top|width), x, y, final
    cursorMoved = QtCore.Signal(float, float)
    modeChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "navigate"
        self._step = None
        self._range_param = None
        self._ranges = {}            # param -> list
        self._region_items = []
        self._anchor_items = []
        self._anchors = []
        self._peak_items = []
        self._component_items = []
        self._slice_items = []
        self._x_extent = (0.0, 1.0)
        self._data = (np.array([]), np.array([]))

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        lay.addWidget(self.toolbar)
        group = QtGui.QActionGroup(self)
        self.mode_actions = {}
        for key, (title, tip) in MODES.items():
            a = QtGui.QAction(title, self, checkable=True)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            a.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            group.addAction(a)
            self.toolbar.addAction(a)
            self.mode_actions[key] = a
        self.mode_actions["navigate"].setChecked(True)
        self.toolbar.addSeparator()
        self.act_auto = self.toolbar.addAction("Auto-range", self.autorange)
        self.act_invert = QtGui.QAction("Invert x", self, checkable=True)
        self.act_invert.setToolTip("Reverse the x axis (e.g. IR wavenumbers)")
        self.act_invert.toggled.connect(lambda on: self.p_main.getViewBox().invertX(on))
        self.toolbar.addAction(self.act_invert)
        self.act_components = QtGui.QAction("Components", self, checkable=True, checked=True)
        self.act_components.toggled.connect(lambda on: [c.setVisible(on) for c in self._component_items])
        self.toolbar.addAction(self.act_components)
        self.act_resid = QtGui.QAction("Residuals", self, checkable=True, checked=True)
        self.act_resid.toggled.connect(self._toggle_resid)
        self.toolbar.addAction(self.act_resid)
        self.act_points = QtGui.QAction("Points", self, checkable=True)
        self.act_points.setToolTip("Draw data as points instead of a line")
        self.act_points.toggled.connect(lambda _on: self._apply_data_style())
        self.toolbar.addAction(self.act_points)
        self._scatter = False
        self.hint = QtWidgets.QLabel()
        self.hint.setObjectName("hint")
        self.hint.setContentsMargins(8, 2, 8, 2)
        lay.addWidget(self.hint)

        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw, 1)
        self.p_main = self.glw.addPlot(row=0, col=0)
        self.p_res = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 4)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        self.p_res.setXLink(self.p_main)
        for p in (self.p_main, self.p_res):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setWidth(60)
        self.p_res.setMaximumHeight(180)
        self.p_res.setLabel("left", "Res.")
        self.legend = self.p_main.addLegend(offset=(-10, 10), labelTextSize="9pt")

        self.c_input = _curve(pg.mkPen(COLORS["input"], width=1))
        self.c_data = _curve(pg.mkPen(COLORS["data"], width=1.2))
        self.c_excl = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(COLORS["excluded"]))
        self.c_base = _curve(pg.mkPen(COLORS["baseline"], width=2, style=QtCore.Qt.DashLine))
        self.c_smooth = _curve(pg.mkPen(COLORS["smoothed"], width=1.5))
        self.c_preview = _curve(pg.mkPen(COLORS["preview"], width=1.5, style=QtCore.Qt.DashLine))
        self.c_fit = _curve(pg.mkPen(COLORS["fit"], width=2))
        self._legend_names = [(self.c_input, "Input"), (self.c_data, "Data"), (self.c_excl, "Excluded"),
                              (self.c_base, "Baseline"), (self.c_smooth, "Smoothed (display only)"),
                              (self.c_preview, "Model (start values)"), (self.c_fit, "Fit")]
        for c, _n in self._legend_names:
            self.p_main.addItem(c)
        self.c_res = _curve(pg.mkPen(COLORS["resid"], width=1))
        self.res_zero = pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#808080"))
        self.res_band = [pg.InfiniteLine(v, angle=0, pen=pg.mkPen("#c0c0c0", style=QtCore.Qt.DashLine))
                         for v in (-2, 2)]
        self.p_res.addItem(self.res_zero)
        for b in self.res_band:
            self.p_res.addItem(b)
        self.p_res.addItem(self.c_res)

        self.p_main.scene().sigMouseClicked.connect(self._clicked)
        self._proxy = pg.SignalProxy(self.p_main.scene().sigMouseMoved, rateLimit=30, slot=self._moved)
        self.set_mode("navigate")

    # ================================================================ modes
    def set_mode(self, mode):
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        self.hint.setText(MODES[mode][1])
        self.modeChanged.emit(mode)

    def _toggle_resid(self, on):
        self.p_res.setVisible(on)

    def _apply_data_style(self):
        if self._scatter or self.act_points.isChecked():   # points instead of a connecting line
            self.c_data.setPen(None)
            self.c_data.setSymbol("o")
            self.c_data.setSymbolSize(4)
            self.c_data.setSymbolBrush(pg.mkBrush(COLORS["data"]))
            self.c_data.setSymbolPen(None)
        else:
            self.c_data.setPen(pg.mkPen(COLORS["data"], width=1.2))
            self.c_data.setSymbol(None)

    def autorange(self):
        self.p_main.enableAutoRange()
        self.p_res.enableAutoRange()

    # ================================================================ scene
    def set_scene(self, *, data, input=None, excluded=None, baseline=None, smoothed=None, fit=None,
                  fit_stale=False, components=(), preview=None, residuals=None, x_label="", y_label="",
                  scatter=False, slices=()):
        """slices: list of dicts (label, x, y, xd, yd) – y(x, v) data drawn per value of v."""
        def setc(item, xy):
            if xy is None:
                item.setData([], [])
                item.setVisible(False)
            else:
                _set_curve_data(item, xy[0], xy[1])
                item.setVisible(True)
        x, y = data
        self._data = (np.asarray(x), np.asarray(y))
        if len(x):
            self._x_extent = (float(np.min(x)), float(np.max(x)))
        setc(self.c_data, data)
        self.c_data.setVisible(not slices)        # y(x, v) data: replaced by coloured groups
        self._scatter = scatter
        self._apply_data_style()
        setc(self.c_input, input)
        if excluded is not None and len(excluded[0]):
            self.c_excl.setData(excluded[0], excluded[1])
            self.c_excl.setVisible(True)
        else:
            self.c_excl.setData([], [])
            self.c_excl.setVisible(False)
        setc(self.c_base, baseline)
        setc(self.c_smooth, smoothed)
        setc(self.c_preview, preview)
        setc(self.c_fit, fit)
        self.c_fit.setPen(pg.mkPen("#d9a0a0" if fit_stale else COLORS["fit"], width=2,
                                   style=QtCore.Qt.DashDotLine if fit_stale else QtCore.Qt.SolidLine))
        for c in self._component_items:
            self.p_main.removeItem(c)
        self._component_items = []
        for i, (name, cx, cy) in enumerate(components):
            col = QtGui.QColor(COMPONENT_COLORS[i % len(COMPONENT_COLORS)])
            fill = QtGui.QColor(col)
            fill.setAlpha(60)
            item = _curve(pg.mkPen(col, width=1), None, fillLevel=0, brush=pg.mkBrush(fill))
            item.opts["name"] = name
            item.setZValue(-5)
            self.p_main.addItem(item)        # add before setData (pyqtgraph clipToView needs the view)
            _set_curve_data(item, cx, cy)
            item.setVisible(self.act_components.isChecked())
            self._component_items.append(item)
        if residuals is None:
            self.c_res.setData([], [])
            for b in self.res_band:
                b.setVisible(False)
        else:
            rx, ry, normalized = residuals
            _set_curve_data(self.c_res, rx, ry)
            for b in self.res_band:
                b.setVisible(bool(normalized))
            self.p_res.setLabel("left", "Res./σ" if normalized else "Res.")
        self.p_main.setLabel("bottom", x_label)
        self.p_main.setLabel("left", y_label)
        self.legend.clear()
        for item, name in self._legend_names:
            if item.isVisible():
                self.legend.addItem(item, name if not (item is self.c_fit and fit_stale) else "Fit (out of date)")
        for item in self._component_items:
            self.legend.addItem(item, item.opts.get("name") or "")
        self._set_slices(slices, fit_stale)
        self.p_res.enableAutoRange(axis="y")

    def _set_slices(self, slices, stale=False):
        for it in self._slice_items:
            self.p_main.removeItem(it)
        self._slice_items = []
        if not slices:
            return
        for i, sl in enumerate(slices):
            col = QtGui.QColor(COMPONENT_COLORS[i % len(COMPONENT_COLORS)])
            pts = pg.ScatterPlotItem(sl["x"], sl["y"], size=5, pen=None, brush=pg.mkBrush(col))
            pts.setZValue(2)
            self.p_main.addItem(pts)
            self._slice_items.append(pts)
            if sl.get("xd") is not None:
                line = pg.PlotDataItem(pen=pg.mkPen(col, width=1.6,
                                                    style=QtCore.Qt.DashDotLine if stale else QtCore.Qt.SolidLine))
                self.p_main.addItem(line)
                _set_curve_data(line, sl["xd"], sl["yd"])
                line.setZValue(3)
                self._slice_items.append(line)
                self.legend.addItem(line, sl["label"])
            else:
                self.legend.addItem(pts, sl["label"])

    # ================================================================ interactive items
    def set_tools(self, step=None, anchors_eff=None, peaks=(), fit_range=None):
        """step: selected pipeline Step (or None); anchors_eff: (x, y) effective anchor positions;
        peaks: list of dicts {prefix, top: (x, y), width: (x, y) | None}; fit_range: [a, b] | None."""
        self._step = step
        self._clear_items()
        params = step.params if step is not None else {}
        if step is not None and step.op == "baseline_anchors":
            self._anchors = [list(a) for a in params.get("anchors", [])]
            if anchors_eff is not None:
                for i, (ax, ay) in enumerate(zip(*anchors_eff)):
                    manual = self._anchors[i][1] is not None if i < len(self._anchors) else False
                    it = AnchorItem(pos=(ax, ay), size=12, symbol="o" if not manual else "s",
                                    pen=pg.mkPen(COLORS["anchor"], width=2),
                                    brush=pg.mkBrush(255, 255, 255, 200), hoverBrush=pg.mkBrush(COLORS["anchor"]))
                    it.setZValue(20)
                    it.idx = i
                    it.sigPositionChangeFinished.connect(self._anchor_moved)
                    it.sigRemove.connect(self._anchor_removed)
                    self.p_main.addItem(it)
                    self._anchor_items.append(it)
        # ranges / range parameters
        self._ranges = {}
        if step is not None:
            from ..ops import get_op
            spec = get_op(step.op)
            for p in spec.params:
                if p.kind == "ranges":
                    self._ranges[p.name] = [list(r) for r in params.get(p.name, [])]
                    color = COLORS["region_exclude"] if "exclude" in p.name else \
                        COLORS["region_force"] if "force" in p.name else COLORS["region_generic"]
                    for j, (a, b) in enumerate(self._ranges[p.name]):
                        self._add_region_item(p.name, j, a, b, color)
                elif p.kind == "range":
                    a, b = params.get(p.name) or [None, None]
                    if a is not None and b is not None:
                        self._add_region_item(p.name, None, a, b, COLORS["region_generic"])
            if step.op == "crop":
                a = params.get("xmin")
                b = params.get("xmax")
                lo, hi = self._x_extent
                self._add_region_item("crop", None, lo if a is None else a, hi if b is None else b,
                                      (63, 144, 218, 25))
        elif fit_range:
            self._add_region_item("fit_range", None, fit_range[0], fit_range[1], (90, 170, 90, 35))
        for pk in peaks:
            top = AnchorItem(pos=pk["top"], size=13, symbol="t1", pen=pg.mkPen(COLORS["peak"], width=2),
                             brush=pg.mkBrush(255, 255, 255, 220), hoverBrush=pg.mkBrush(COLORS["peak"]),
                             label=pk["prefix"].rstrip("_"),
                             labelOpts={"color": COLORS["peak"], "offset": (0, -18), "anchor": (0.5, 1)})
            top.prefix, top.handle = pk["prefix"], "top"
            top.setZValue(25)
            top.sigPositionChanged.connect(lambda it: self._peak_moved(it, False))
            top.sigPositionChangeFinished.connect(lambda it: self._peak_moved(it, True))
            self.p_main.addItem(top)
            self._peak_items.append(top)
            if pk.get("width") is not None:
                wid = AnchorItem(pos=pk["width"], size=10, symbol="arrow_right",
                                 pen=pg.mkPen(COLORS["peak"], width=1.5), brush=pg.mkBrush(255, 255, 255, 220),
                                 hoverBrush=pg.mkBrush(COLORS["peak"]))
                wid.prefix, wid.handle = pk["prefix"], "width"
                wid.setZValue(25)
                wid.sigPositionChanged.connect(lambda it: self._peak_moved(it, False))
                wid.sigPositionChangeFinished.connect(lambda it: self._peak_moved(it, True))
                self.p_main.addItem(wid)
                self._peak_items.append(wid)

    def set_range_target(self, param):
        self._range_param = param

    def _clear_items(self):
        for it in self._anchor_items + self._region_items + self._peak_items:
            self.p_main.removeItem(it)
        self._anchor_items, self._region_items, self._peak_items = [], [], []

    def _add_region_item(self, param, idx, a, b, color):
        it = RegionItem(values=(a, b), brush=pg.mkBrush(*color) if isinstance(color, tuple) else pg.mkBrush(color),
                        pen=pg.mkPen((90, 90, 90, 160)), hoverBrush=pg.mkBrush(120, 120, 120, 70))
        it.param, it.idx = param, idx
        it.setZValue(-10)
        it.sigRegionChangeFinished.connect(self._region_moved)
        it.sigRemove.connect(self._region_removed)
        self.p_main.addItem(it)
        self._region_items.append(it)

    # ================================================================ events
    def _view_xy(self, ev):
        vb = self.p_main.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return None
        pt = vb.mapSceneToView(ev.scenePos())
        return pt.x(), pt.y()

    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or ev.isAccepted() or self.mode == "navigate":
            return
        xy = self._view_xy(ev)
        if xy is None:
            return
        x, y = xy
        if self.mode == "anchor":
            if self._step is None or self._step.op != "baseline_anchors":
                self.hint.setText("To place anchors, first select a 'Baseline: anchor points' step.")
                return
            manual = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            self._anchors.append([x, y if manual else None])
            self._anchors.sort(key=lambda a: a[0])
            self.anchorsEdited.emit([list(a) for a in self._anchors])
        elif self.mode == "region":
            vr = self.p_main.getViewBox().viewRange()[0]
            half = 0.02 * abs(vr[1] - vr[0])
            a, b = x - half, x + half
            if self._step is not None:
                from ..ops import get_op
                spec = get_op(self._step.op)
                kinds = {p.name: p.kind for p in spec.params}
                target = self._range_param if self._range_param in kinds else None
                if target is None:
                    target = next((n for n, k in kinds.items() if k in ("ranges", "range")), None)
                if target is None:
                    self.hint.setText("This step has no range parameters.")
                    return
                if kinds[target] == "ranges":
                    lst = self._ranges.get(target, []) + [[a, b]]
                    self.rangesEdited.emit(target, lst)
                else:
                    self.rangeEdited.emit(target, [a, b])
            else:
                self.rangeEdited.emit("fit_range", [a, b])
        elif self.mode == "peak":
            self.peakAdded.emit(x, y)

    def _moved(self, args):
        pos = args[0]
        vb = self.p_main.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            pt = vb.mapSceneToView(pos)
            self.cursorMoved.emit(pt.x(), pt.y())

    def _anchor_moved(self, item):
        p = item.pos()
        i = item.idx
        if i < len(self._anchors):
            shift = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
            manual = shift or self._anchors[i][1] is not None
            self._anchors[i] = [p.x(), p.y() if manual else None]
        self._anchors.sort(key=lambda a: a[0])
        self.anchorsEdited.emit([list(a) for a in self._anchors])

    def _anchor_removed(self, item):
        if item.idx < len(self._anchors):
            del self._anchors[item.idx]
            self.anchorsEdited.emit([list(a) for a in self._anchors])

    def _region_moved(self, item):
        a, b = item.getRegion()
        if item.param == "crop":
            self.rangeEdited.emit("crop", [a, b])
        elif item.idx is None:
            self.rangeEdited.emit(item.param, [a, b])
        else:
            lst = [list(r) for r in self._ranges.get(item.param, [])]
            if item.idx < len(lst):
                lst[item.idx] = [a, b]
            self.rangesEdited.emit(item.param, lst)

    def _region_removed(self, item):
        if item.param == "crop":
            self.rangeEdited.emit("crop", [None, None])
        elif item.idx is None:
            self.rangeEdited.emit(item.param, [None, None])
        else:
            lst = [list(r) for r in self._ranges.get(item.param, [])]
            if item.idx < len(lst):
                del lst[item.idx]
            self.rangesEdited.emit(item.param, lst)

    def _peak_moved(self, item, final):
        p = item.pos()
        self.peakDragged.emit(item.prefix, item.handle, p.x(), p.y(), final)

    def update_preview_curve(self, xy):
        if xy is None:
            self.c_preview.setVisible(False)
        else:
            _set_curve_data(self.c_preview, *xy)
            self.c_preview.setVisible(True)
