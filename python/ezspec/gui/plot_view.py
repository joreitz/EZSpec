"""Interactive plot: data, baseline, fit, components and residuals with editing tools."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..view import NORMALIZE
from .theme import COLORS, COMPONENT_COLORS

MAX_LEGEND_OVERLAYS = 12     # more overlaid spectra than this are identified by the Data list, not the legend

MODES = {
    "navigate": ("Navigate", "Zoom (mouse wheel / right-drag) and pan (left-drag)"),
    "anchor": ("Anchor", "Click: baseline anchor (snaps to the data; Shift: free) · Drag: move "
                         "· Right-click: delete"),
    "region": ("Range", "Click: add range (masks, baseline/noise/fit range) · Drag edges to resize "
                        "· Right-click: delete"),
    "peak": ("Peak", "Click: add a peak here · Drag markers: position/height or width"),
    "zoom": ("Zoom", "Drag over a part of the spectrum to show it enlarged next to it · Drag a band or its edges "
                     "· Remove: right-click the band or its panel, or the × of the panel"),
}
ZOOM_COLORS = ["#3f90da", "#e76300", "#2e9c6a", "#832db6", "#a96b59", "#17a7b5"]


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


class ZoomBand(RegionItem):
    """Part of the overview that is shown enlarged in a panel of its own. Only the zoom tool
    moves it, so that it is not in the way of the other tools."""


class CloseButton(pg.LabelItem):
    """Small × in the corner of a zoom panel."""

    def __init__(self, callback, color):
        super().__init__("×", color=color, size="13pt", bold=True)
        self.callback = callback
        self.setZValue(200)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Remove this zoom region")

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.callback()


class BandViewBox(pg.ViewBox):
    """View box in which a horizontal drag can select an x range instead of panning, and whose
    right-clicks go to the owner first.

    Bands lie behind the curves (negative z), and pyqtgraph offers a click to the view box before
    it offers it to anything behind it – the view box would open its context menu and a band
    would never see its right-click. ``right_click(ev) -> bool`` therefore decides first."""
    sigBand = QtCore.Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.band_mode = False
        self.right_click = None
        self._band = None

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton and self.right_click is not None and self.right_click(ev):
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if not self.band_mode or axis is not None or ev.button() != QtCore.Qt.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
        x1 = self.mapSceneToView(ev.scenePos()).x()
        lo, hi = min(x0, x1), max(x0, x1)
        if self._band is None:
            self._band = pg.LinearRegionItem(values=(lo, hi), movable=False, brush=pg.mkBrush(63, 144, 218, 50),
                                             pen=pg.mkPen("#3f90da"))
            self._band.setZValue(50)
            self.addItem(self._band, ignoreBounds=True)
        self._band.setRegion((lo, hi))
        if ev.isFinish():
            self.removeItem(self._band)
            self._band = None
            if hi > lo:
                self.sigBand.emit(lo, hi)


class _Detail:
    """A zoom region: the band in the overview and the panel that shows it enlarged."""

    def __init__(self, plot, layer, band, color):
        self.plot, self.layer, self.band, self.color = plot, layer, band, color
        self.region = [0.0, 1.0]


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


class SceneLayer:
    """The curves of one :class:`~ezspec.gui.scene.Scene` inside one plot panel.

    Owns the data, baseline, fit, component and slice items of a ``PlotItem``
    (and its legend, if any), so every panel showing the scene draws it the
    same way."""

    def __init__(self, plot, legend=None):
        self.plot = plot
        self.legend = legend
        self.points = False              # draw data as points (toolbar switch)
        self.components_visible = True
        self._scatter = False            # the data ask for points (e.g. calibration points)
        self._color = COLORS["data"]
        self._emphasis = False           # several spectra in the plot: the active one is drawn stronger
        self.component_items = []
        self.slice_items = []
        self.overlay_items = {}          # dataset id -> curve of an overlaid spectrum
        self.overlay_clicked = None      # callback(dataset id)
        self._clickable = True
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
            plot.addItem(c)

    def set_points(self, on):
        self.points = bool(on)
        self._apply_data_style()
        for item in self.overlay_items.values():
            self._style_overlay(item)

    def set_clickable(self, on):
        """Overlaid spectra can be clicked to make them the active one (only while navigating,
        so that a tool can still be used on top of a curve)."""
        self._clickable = bool(on)
        for item in self.overlay_items.values():
            item.setCurveClickable(self._clickable, width=8)

    def set_components_visible(self, on):
        self.components_visible = bool(on)
        for c in self.component_items:
            c.setVisible(self.components_visible)

    def _apply_data_style(self):
        if self._scatter or self.points:   # points instead of a connecting line
            self.c_data.setPen(None)
            self.c_data.setSymbol("o")
            self.c_data.setSymbolSize(4)
            self.c_data.setSymbolBrush(pg.mkBrush(self._color))
            self.c_data.setSymbolPen(None)
        else:
            self.c_data.setPen(pg.mkPen(self._color, width=1.8 if self._emphasis else 1.2))
            self.c_data.setSymbol(None)

    def _style_overlay(self, item):
        if item.ov_scatter or self.points:
            item.setPen(None)
            item.setSymbol("o")
            item.setSymbolSize(4)
            item.setSymbolBrush(pg.mkBrush(item.ov_color))
            item.setSymbolPen(None)
        else:
            item.setPen(pg.mkPen(item.ov_color, width=1.1))
            item.setSymbol(None)

    def _set_overlays(self, overlays):
        keep = {o.id for o in overlays}
        for key in [k for k in self.overlay_items if k not in keep]:
            self.plot.removeItem(self.overlay_items.pop(key))
        for o in overlays:
            item = self.overlay_items.get(o.id)
            if item is None:
                item = _curve(None)
                item.setZValue(-3)
                item.setCurveClickable(self._clickable, width=8)
                # (pyqtgraph passes the inner curve item or the data item, depending on its version)
                item.sigClicked.connect(lambda *_args, ds_id=o.id: self._overlay_clicked(ds_id))
                self.plot.addItem(item)              # add before setData (clipToView needs the view)
                self.overlay_items[o.id] = item
            item.ov_color, item.ov_scatter = o.color, o.scatter
            self._style_overlay(item)
            _set_curve_data(item, o.x, o.y)

    def _overlay_clicked(self, ds_id):
        if self.overlay_clicked is not None:
            self.overlay_clicked(ds_id)

    def set_preview(self, xy):
        """Replace only the start-value model curve (live while a peak handle is dragged)."""
        if xy is None:
            self.c_preview.setVisible(False)
        else:
            _set_curve_data(self.c_preview, *xy)
            self.c_preview.setVisible(True)

    def set_scene(self, scene):
        def setc(item, xy):
            if xy is None:
                item.setData([], [])
                item.setVisible(False)
            else:
                _set_curve_data(item, xy[0], xy[1])
                item.setVisible(True)
        self._set_overlays(scene.overlays)
        setc(self.c_data, scene.data)
        self.c_data.setVisible(not scene.slices)        # y(x, v) data: replaced by coloured groups
        self._scatter = scene.scatter
        self._color = scene.color or COLORS["data"]
        self._emphasis = bool(scene.overlays)
        self._apply_data_style()
        setc(self.c_input, scene.input)
        if scene.excluded is not None and len(scene.excluded[0]):
            self.c_excl.setData(scene.excluded[0], scene.excluded[1])
            self.c_excl.setVisible(True)
        else:
            self.c_excl.setData([], [])
            self.c_excl.setVisible(False)
        setc(self.c_base, scene.baseline)
        setc(self.c_smooth, scene.smoothed)
        setc(self.c_preview, scene.preview)
        setc(self.c_fit, scene.fit)
        stale = scene.fit_stale
        self.c_fit.setPen(pg.mkPen("#d9a0a0" if stale else COLORS["fit"], width=2,
                                   style=QtCore.Qt.DashDotLine if stale else QtCore.Qt.SolidLine))
        for c in self.component_items:
            self.plot.removeItem(c)
        self.component_items = []
        for i, (name, cx, cy) in enumerate(scene.components):
            col = QtGui.QColor(COMPONENT_COLORS[i % len(COMPONENT_COLORS)])
            fill = QtGui.QColor(col)
            fill.setAlpha(60)
            item = _curve(pg.mkPen(col, width=1), None, fillLevel=scene.y_map[1], brush=pg.mkBrush(fill))
            item.opts["name"] = name
            item.setZValue(-5)
            self.plot.addItem(item)          # add before setData (pyqtgraph clipToView needs the view)
            _set_curve_data(item, cx, cy)
            item.setVisible(self.components_visible)
            self.component_items.append(item)
        if self.legend is not None:
            self.legend.clear()
            for item, name in self._legend_names:
                if item.isVisible():
                    if item is self.c_fit and stale:
                        name = "Fit (out of date)"
                    elif item is self.c_data and scene.overlays:
                        name = scene.name or name       # among other spectra: say which one this is
                    self.legend.addItem(item, name)
            if not scene.overlays:               # among several spectra the peak labels have to do
                for item in self.component_items:
                    self.legend.addItem(item, item.opts.get("name") or "")
            if len(scene.overlays) <= MAX_LEGEND_OVERLAYS:
                for o in scene.overlays:
                    self.legend.addItem(self.overlay_items[o.id], o.name)
        self._set_slices(scene.slices, stale)

    def _set_slices(self, slices, stale=False):
        for it in self.slice_items:
            self.plot.removeItem(it)
        self.slice_items = []
        for i, sl in enumerate(slices):
            col = QtGui.QColor(COMPONENT_COLORS[i % len(COMPONENT_COLORS)])
            pts = pg.ScatterPlotItem(sl["x"], sl["y"], size=5, pen=None, brush=pg.mkBrush(col))
            pts.setZValue(2)
            self.plot.addItem(pts)
            self.slice_items.append(pts)
            entry = pts
            if sl.get("xd") is not None:
                line = pg.PlotDataItem(pen=pg.mkPen(col, width=1.6,
                                                    style=QtCore.Qt.DashDotLine if stale else QtCore.Qt.SolidLine))
                self.plot.addItem(line)
                _set_curve_data(line, sl["xd"], sl["yd"])
                line.setZValue(3)
                self.slice_items.append(line)
                entry = line
            if self.legend is not None:
                self.legend.addItem(entry, sl["label"])


class PlotView(QtWidgets.QWidget):
    anchorsEdited = QtCore.Signal(list)
    rangesEdited = QtCore.Signal(str, list)
    rangeEdited = QtCore.Signal(str, list)
    peakAdded = QtCore.Signal(float, float)
    peakDragged = QtCore.Signal(str, str, float, float, bool)   # prefix, handle (top|width), x, y, final
    cursorMoved = QtCore.Signal(float, float)
    modeChanged = QtCore.Signal(str)
    figureRequested = QtCore.Signal()
    captureRequested = QtCore.Signal()           # add the view to the page
    pagesRequested = QtCore.Signal()             # open the page composer
    datasetClicked = QtCore.Signal(str)          # an overlaid spectrum was clicked
    regionsEdited = QtCore.Signal(list)          # zoom regions [[x0, x1], …] changed in the plot
    viewOptionChanged = QtCore.Signal(str, object)   # stacked / gap / normalize set in the toolbar

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
        self._details = []           # zoom regions shown next to the overview
        self._detail_layout = None
        self._scene = None
        self._x_extent = (0.0, 1.0)
        self._y_map = (1.0, 0.0)     # display transform of the scene: y' = a·y + b

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
        self.act_auto = self.toolbar.addAction("Auto", self.autorange)
        self.act_auto.setToolTip("Auto-range: show everything again")
        self.toolbar.addSeparator()
        # several spectra in the plot: display only, data and fits are never changed
        self.act_stack = QtGui.QAction("Stack", self, checkable=True)
        self.act_stack.setToolTip("Draw the spectra in the plot above each other instead of on top of each other "
                                  "(display only – data and fits are not changed)")
        self.act_stack.toggled.connect(lambda on: self.viewOptionChanged.emit("stacked", bool(on)))
        self.toolbar.addAction(self.act_stack)
        self.norm_button = QtWidgets.QToolButton()
        self.norm_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.norm_button.setToolTip("Normalise: scale every spectrum in the plot for comparison (display only – data and "
                                    "fits are not changed)")
        norm_menu = QtWidgets.QMenu(self.norm_button)
        norm_group = QtGui.QActionGroup(self)
        self.norm_actions = {}
        for key, title in NORMALIZE.items():
            a = QtGui.QAction(title, self, checkable=True)
            a.triggered.connect(lambda _=False, k=key: self.viewOptionChanged.emit("normalize", k))
            norm_group.addAction(a)
            norm_menu.addAction(a)
            self.norm_actions[key] = a
        self.norm_button.setMenu(norm_menu)
        self.toolbar.addWidget(self.norm_button)
        # what is drawn
        self.act_invert = QtGui.QAction("Invert x", self, checkable=True)
        self.act_invert.setToolTip("Reverse the x axis (e.g. IR wavenumbers)")
        self.act_invert.toggled.connect(self._invert_x)
        self.act_components = QtGui.QAction("Components", self, checkable=True, checked=True)
        self.act_components.toggled.connect(
            lambda on: [layer.set_components_visible(on) for layer in self._layers()])
        self.act_resid = QtGui.QAction("Residuals", self, checkable=True, checked=True)
        self.act_resid.toggled.connect(self._toggle_resid)
        self.act_points = QtGui.QAction("Points", self, checkable=True)
        self.act_points.setToolTip("Draw data as points instead of a line")
        self.act_points.toggled.connect(lambda on: [layer.set_points(on) for layer in self._layers()])
        self.act_legend = QtGui.QAction("Legend", self, checkable=True, checked=True)
        self.act_legend.toggled.connect(lambda on: self.legend.setVisible(on))
        self.display_button = QtWidgets.QToolButton()
        self.display_button.setText("Display")
        self.display_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.display_button.setToolTip("What the plot draws: components, residuals, points, legend, axis direction, "
                                       "distance of stacked spectra")
        display_menu = QtWidgets.QMenu(self.display_button)
        display_menu.setToolTipsVisible(True)
        for a in (self.act_components, self.act_resid, self.act_points, self.act_legend, self.act_invert):
            display_menu.addAction(a)
        display_menu.addSeparator()
        gap_row = QtWidgets.QWidget()
        gl = QtWidgets.QHBoxLayout(gap_row)
        gl.setContentsMargins(8, 3, 8, 3)
        gl.addWidget(QtWidgets.QLabel("Stack distance"))
        self.gap_spin = QtWidgets.QSpinBox()
        self.gap_spin.setRange(0, 500)
        self.gap_spin.setSingleStep(10)
        self.gap_spin.setValue(100)
        self.gap_spin.setSuffix(" %")
        self.gap_spin.setKeyboardTracking(False)
        self.gap_spin.setToolTip("Distance between stacked spectra, in % of the tallest one")
        self.gap_spin.valueChanged.connect(lambda v: self.viewOptionChanged.emit("gap", v / 100.0))
        gl.addWidget(self.gap_spin)
        gap_action = QtWidgets.QWidgetAction(self)
        gap_action.setDefaultWidget(gap_row)
        display_menu.addAction(gap_action)
        display_menu.addSeparator()
        self.act_clear_zoom = display_menu.addAction("Remove all zoom regions", lambda: self.regionsEdited.emit([]))
        self.act_clear_zoom.setEnabled(False)
        self.display_button.setMenu(display_menu)
        self.toolbar.addWidget(self.display_button)
        self.toolbar.addSeparator()
        self.figure_button = QtWidgets.QToolButton()
        self.figure_button.setText("Figure")
        self.figure_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.figure_button.setToolTip("Publication figures: of this dataset, or a page on which several views are "
                                      "arranged freely")
        fig_menu = QtWidgets.QMenu(self.figure_button)
        fig_menu.setToolTipsVisible(True)
        a = fig_menu.addAction("Figure of this dataset…", self.figureRequested.emit)
        a.setToolTip("Figure editor with journal templates (result, raw data or the selected step)")
        fig_menu.addSeparator()
        a = fig_menu.addAction("Add this view to the page", self.captureRequested.emit)
        a.setToolTip("Capture what the plot shows – overlays, stacking, zoom regions – as frames on a page")
        fig_menu.addAction("Open pages…", self.pagesRequested.emit)
        self.figure_button.setMenu(fig_menu)
        self.toolbar.addWidget(self.figure_button)
        self.hint = QtWidgets.QLabel()
        self.hint.setObjectName("hint")
        self.hint.setContentsMargins(8, 2, 8, 2)
        self.hint.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)   # never widens the plot
        lay.addWidget(self.hint)

        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw, 1)
        self.p_main = self.glw.addPlot(row=0, col=0, viewBox=BandViewBox())
        self.p_main.getViewBox().sigBand.connect(self._band_selected)
        self.p_main.getViewBox().right_click = lambda ev: self._right_clicked(self.p_main, ev)
        self.p_res = self.glw.addPlot(row=1, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 4)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        self.p_res.setXLink(self.p_main)
        for p in (self.p_main, self.p_res):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setWidth(60)
        self.p_res.setMaximumHeight(180)
        self.p_res.setLabel("left", "Res.")
        self.legend = self.p_main.addLegend(offset=(-10, 10), labelTextSize="9pt",
                                            brush=pg.mkBrush(255, 255, 255, 215))

        self.layer = SceneLayer(self.p_main, self.legend)
        self.layer.overlay_clicked = self.datasetClicked.emit
        self.c_res = _curve(pg.mkPen(COLORS["resid"], width=1))
        self.res_zero = pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#808080"))
        self.res_band = [pg.InfiniteLine(v, angle=0, pen=pg.mkPen("#c0c0c0", style=QtCore.Qt.DashLine))
                         for v in (-2, 2)]
        self.p_res.addItem(self.res_zero)
        for b in self.res_band:
            self.p_res.addItem(b)
        self.p_res.addItem(self.c_res)

        self._region_timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self._region_timer.timeout.connect(self._emit_regions)
        self.p_main.scene().sigMouseClicked.connect(self._clicked)
        self._proxy = pg.SignalProxy(self.p_main.scene().sigMouseMoved, rateLimit=30, slot=self._moved)
        self.set_mode("navigate")
        self.norm_actions["none"].setChecked(True)
        self.norm_button.setText("Norm")

    # ================================================================ modes
    def set_mode(self, mode):
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        self.hint.setText(MODES[mode][1])
        self.hint.setToolTip(MODES[mode][1])
        for layer in self._layers():
            layer.set_clickable(mode == "navigate")
        self.p_main.getViewBox().band_mode = mode == "zoom"
        for d in self._details:
            d.band.setMovable(mode == "zoom")
        self.modeChanged.emit(mode)

    def set_view_options(self, view):
        """Show the display options of a :class:`~ezspec.view.ViewState` in the toolbar."""
        for w in (self.act_stack, self.gap_spin):
            w.blockSignals(True)
        self.act_stack.setChecked(view.stacked)
        self.gap_spin.setValue(int(round(view.gap * 100)))
        for w in (self.act_stack, self.gap_spin):
            w.blockSignals(False)
        self.norm_actions[view.normalize].setChecked(True)
        self.norm_button.setText("Norm" if view.normalize == "none" else
                                 {"max": "Norm: max", "range": "Norm: 0–1", "area": "Norm: area"}[view.normalize])

    def _data_y(self, y):
        """y of the data at a position on the (possibly stacked / normalised) y axis."""
        a, b = self._y_map
        return (y - b) / a

    def _toggle_resid(self, on):
        self.p_res.setVisible(on)

    def autorange(self):
        self.p_main.enableAutoRange()
        # only y: the residual panel follows the x range of the main panel through its link, and
        # auto-ranging its x axis would switch the automatic x range of the main panel off again
        self.p_res.enableAutoRange(axis="y")

    def visible_ranges(self) -> tuple:
        """``(x range, y range)`` of the overview as far as the user has set them (``None``: automatic)."""
        vb = self.p_main.getViewBox()
        auto = vb.state["autoRange"]
        (x0, x1), (y0, y1) = vb.viewRange()
        return (None if auto[0] else [float(x0), float(x1)]), (None if auto[1] else [float(y0), float(y1)])

    def _invert_x(self, on):
        for plot in self._plots():
            plot.getViewBox().invertX(on)

    # ================================================================ zoom regions
    def _plots(self) -> list:
        """Panels that show the scene: the overview and one per zoom region."""
        return [self.p_main] + [d.plot for d in self._details]

    def _layers(self) -> list:
        return [self.layer] + [d.layer for d in self._details]

    @property
    def regions(self) -> list:
        return [list(d.region) for d in self._details]

    def set_regions(self, regions):
        """Show the x ranges ``[[x0, x1], …]`` enlarged next to the overview."""
        regions = [[float(min(a, b)), float(max(a, b))] for a, b in regions]
        if len(regions) != len(self._details):
            self._build_details(len(regions))
        for d, (a, b) in zip(self._details, regions):
            if d.region != [a, b]:
                self._apply_region(d, a, b)
        self.act_clear_zoom.setEnabled(bool(regions))

    def _apply_region(self, d, a, b, band=True):
        d.region = [a, b]
        if band:
            d.band.blockSignals(True)
            d.band.setRegion((a, b))
            d.band.blockSignals(False)
        d.plot.setXRange(a, b, padding=0)
        d.plot.setTitle(f"{a:.5g} – {b:.5g}", color=d.color, size="9pt")

    def _build_details(self, n):
        self._clear_items()                  # handles live in the panels that are about to go
        for d in self._details:
            self.p_main.removeItem(d.band)
        self._details = []
        if self._detail_layout is not None:
            self.glw.ci.removeItem(self._detail_layout)
            self._detail_layout = None
        grid = self.glw.ci.layout
        if n == 0:
            grid.setColumnStretchFactor(0, 1)
            grid.setColumnStretchFactor(1, 0)
            return
        cols = 1 if n <= 3 else 2
        self._detail_layout = self.glw.ci.addLayout(row=0, col=1, rowspan=2)
        self._detail_layout.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretchFactor(0, 3)
        grid.setColumnStretchFactor(1, 1 + cols)
        for i in range(n):
            color = ZOOM_COLORS[i % len(ZOOM_COLORS)]
            plot = self._detail_layout.addPlot(row=i // cols, col=i % cols, viewBox=BandViewBox())
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis("left").setWidth(48)
            vb = plot.getViewBox()
            vb.setMouseEnabled(x=True, y=False)          # y follows the data inside the range
            vb.enableAutoRange(y=True)
            vb.setAutoVisible(y=True)
            vb.invertX(self.act_invert.isChecked())
            for axis in ("left", "bottom", "top", "right"):
                plot.getAxis(axis).setPen(pg.mkPen(color))
            plot.showAxis("top")
            plot.showAxis("right")
            plot.getAxis("top").setStyle(showValues=False, tickLength=0)
            plot.getAxis("right").setStyle(showValues=False, tickLength=0)
            layer = SceneLayer(plot)
            layer.points, layer.components_visible = self.layer.points, self.layer.components_visible
            layer.overlay_clicked = self.datasetClicked.emit
            layer.set_clickable(self.mode == "navigate")
            fill = QtGui.QColor(color)
            fill.setAlpha(38)
            hover = QtGui.QColor(color)
            hover.setAlpha(70)
            band = ZoomBand(values=(0, 1), brush=pg.mkBrush(fill), pen=pg.mkPen(color), hoverBrush=pg.mkBrush(hover))
            band.setZValue(-8)
            band.setMovable(self.mode == "zoom")
            self.p_main.addItem(band, ignoreBounds=True)
            d = _Detail(plot, layer, band, color)
            d.region = None
            vb.right_click = lambda ev, d=d: self._right_clicked(d.plot, ev, d)
            d.close = CloseButton(lambda d=d: self._remove_later(d), color)
            d.close.setParentItem(vb)
            d.close.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-3, -2))
            band.sigRegionChanged.connect(lambda _b=None, d=d: self._band_moved(d))
            band.sigRegionChangeFinished.connect(lambda _b=None: self._emit_regions())
            band.sigRemove.connect(lambda _b=None, d=d: self._band_removed(d))
            vb.sigRangeChangedManually.connect(lambda *_a, d=d: self._detail_ranged(d))
            self._details.append(d)
        if self._scene is not None:
            self.show_scene(self._scene)

    def _band_selected(self, a, b):
        """The zoom tool was dragged over [a, b] in the overview."""
        self.regionsEdited.emit(self.regions + [[a, b]])

    def _band_moved(self, d):
        a, b = d.band.getRegion()
        self._apply_region(d, min(a, b), max(a, b), band=False)

    def _band_removed(self, d):
        if d in self._details:
            self.regionsEdited.emit([list(o.region) for o in self._details if o is not d])

    def _remove_later(self, d=None):
        """Remove a zoom region (``None``: all) once the click that asked for it has been handled –
        its panel is deleted, and the click may come from that panel."""
        if d is None:
            QtCore.QTimer.singleShot(0, lambda: self.regionsEdited.emit([]))
        else:
            QtCore.QTimer.singleShot(0, lambda: self._band_removed(d))

    def zoom_menu(self, d) -> QtWidgets.QMenu:
        """Context menu of a zoom region (of its band in the overview and of its panel)."""
        menu = QtWidgets.QMenu(self)
        a, b = d.region
        menu.addAction(f"Remove zoom region {a:.5g} – {b:.5g}", lambda: self._remove_later(d))
        if len(self._details) > 1:
            menu.addAction("Remove all zoom regions", lambda: self._remove_later(None))
        return menu

    def _right_clicked(self, plot, ev, detail=None) -> bool:
        """Right-click in a panel. A tool removes its own items directly: the range tool the range
        under the cursor, the zoom tool the zoom region of the band or panel. With another tool a
        zoom band or panel offers its removal in a menu. False: not ours, the usual plot menu opens."""
        vb = plot.getViewBox()
        x = vb.mapSceneToView(ev.scenePos()).x()
        tol = 4.0 * abs(vb.viewPixelSize()[0])

        def under(lo, hi):
            return min(lo, hi) - tol <= x <= max(lo, hi) + tol

        def width(region):
            return abs(region[1] - region[0])
        ranges = [it for it in self._region_items if it.panel is plot and under(*it.getRegion())]
        if ranges and self.mode == "region":
            it = min(ranges, key=lambda it: width(it.getRegion()))
            QtCore.QTimer.singleShot(0, lambda: self._region_removed(it))
            return True
        if detail is None:
            bands = [d for d in self._details if under(*d.region)]
            if not bands:
                return False
            detail = min(bands, key=lambda d: width(d.region))
        if self.mode == "zoom":
            self._remove_later(detail)
        else:
            self.zoom_menu(detail).exec(ev.screenPos().toPoint())
        return True

    def _detail_ranged(self, d):
        """The enlarged panel was panned or zoomed with the mouse: its band follows."""
        a, b = d.plot.getViewBox().viewRange()[0]
        d.region = [float(a), float(b)]
        d.band.blockSignals(True)
        d.band.setRegion((a, b))
        d.band.blockSignals(False)
        d.plot.setTitle(f"{a:.5g} – {b:.5g}", color=d.color, size="9pt")
        self._region_timer.start()

    def _emit_regions(self):
        self.regionsEdited.emit(self.regions)

    # ================================================================ scene
    def show_scene(self, scene):
        """Draw a :class:`~ezspec.gui.scene.Scene`: curves, residual panel, labels and editing tools."""
        x = np.asarray(scene.data[0])
        if len(x):
            self._x_extent = (float(np.min(x)), float(np.max(x)))
        self._y_map = scene.y_map
        self._scene = scene
        for layer in self._layers():
            layer.set_scene(scene)
        if scene.residuals is None:
            self.c_res.setData([], [])
            for b in self.res_band:
                b.setVisible(False)
            self.p_res.setLabel("left", scene.residual_label or "Res.")
        else:
            rx, ry, normalized = scene.residuals
            _set_curve_data(self.c_res, rx, ry)
            for b in self.res_band:
                b.setVisible(bool(normalized))
            self.p_res.setLabel("left", scene.residual_label or ("Res./σ" if normalized else "Res."))
        self.p_main.setLabel("bottom", scene.x_label)
        self.p_main.setLabel("left", scene.y_label)
        self.p_res.enableAutoRange(axis="y")
        self.set_tools(step=scene.step, anchors_eff=scene.anchors_eff, peaks=scene.peaks,
                       fit_range=scene.fit_range if scene.step is None else None)

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
                    for plot in self._plots():
                        it = AnchorItem(pos=(ax, ay), size=12, symbol="o" if not manual else "s",
                                        pen=pg.mkPen(COLORS["anchor"], width=2), brush=pg.mkBrush(255, 255, 255, 200),
                                        hoverBrush=pg.mkBrush(COLORS["anchor"]))
                        it.setZValue(20)
                        it.idx = i
                        it.sigPositionChangeFinished.connect(self._anchor_moved)
                        it.sigRemove.connect(self._anchor_removed)
                        self._place(plot, it, self._anchor_items)
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
            for plot in self._plots():
                top = AnchorItem(pos=pk["top"], size=13, symbol="t1", pen=pg.mkPen(COLORS["peak"], width=2),
                                 brush=pg.mkBrush(255, 255, 255, 220), hoverBrush=pg.mkBrush(COLORS["peak"]),
                                 label=pk["prefix"].rstrip("_"),
                                 labelOpts={"color": COLORS["peak"], "offset": (0, -18), "anchor": (0.5, 1)})
                top.prefix, top.handle = pk["prefix"], "top"
                top.setZValue(25)
                top.sigPositionChanged.connect(lambda it: self._peak_moved(it, False))
                top.sigPositionChangeFinished.connect(lambda it: self._peak_moved(it, True))
                self._place(plot, top, self._peak_items)
                if pk.get("width") is not None:
                    wid = AnchorItem(pos=pk["width"], size=10, symbol="arrow_right",
                                     pen=pg.mkPen(COLORS["peak"], width=1.5), brush=pg.mkBrush(255, 255, 255, 220),
                                     hoverBrush=pg.mkBrush(COLORS["peak"]))
                    wid.prefix, wid.handle = pk["prefix"], "width"
                    wid.setZValue(25)
                    wid.sigPositionChanged.connect(lambda it: self._peak_moved(it, False))
                    wid.sigPositionChangeFinished.connect(lambda it: self._peak_moved(it, True))
                    self._place(plot, wid, self._peak_items)

    def set_range_target(self, param):
        self._range_param = param

    def _place(self, plot, item, store):
        """Add a tool item to a panel (every panel that shows the scene has its own set of handles)."""
        item.panel = plot
        if plot is self.p_main:
            plot.addItem(item)
        else:
            plot.addItem(item, ignoreBounds=True)        # handles do not set the range of a zoom panel
        store.append(item)

    def _clear_items(self):
        for it in self._anchor_items + self._region_items + self._peak_items:
            it.panel.removeItem(it)
        self._anchor_items, self._region_items, self._peak_items = [], [], []

    def _add_region_item(self, param, idx, a, b, color):
        for plot in self._plots():
            it = RegionItem(values=(a, b),
                            brush=pg.mkBrush(*color) if isinstance(color, tuple) else pg.mkBrush(color),
                            pen=pg.mkPen((90, 90, 90, 160)), hoverBrush=pg.mkBrush(120, 120, 120, 70))
            it.param, it.idx = param, idx
            it.setZValue(-10)
            it.sigRegionChangeFinished.connect(self._region_moved)
            it.sigRemove.connect(self._region_removed)
            self._place(plot, it, self._region_items)

    # ================================================================ events
    def _panel_at(self, scene_pos):
        """``(view box, x, y)`` of the panel under a scene position, or ``None``."""
        for plot in self._plots():
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(scene_pos):
                pt = vb.mapSceneToView(scene_pos)
                return vb, pt.x(), pt.y()
        return None

    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or ev.isAccepted() or self.mode == "navigate":
            return
        hit = self._panel_at(ev.scenePos())
        if hit is None:
            return
        vb, x, y = hit[0], hit[1], self._data_y(hit[2])
        if self.mode == "zoom":
            if vb is self.p_main.getViewBox():           # a click without dragging: a region around it
                vr = vb.viewRange()[0]
                half = 0.05 * abs(vr[1] - vr[0])
                self.regionsEdited.emit(self.regions + [[x - half, x + half]])
            return
        if self.mode == "anchor":
            if self._step is None or self._step.op != "baseline_anchors":
                self.hint.setText("To place anchors, first select a 'Baseline: anchor points' step.")
                return
            manual = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            self._anchors.append([x, y if manual else None])
            self._anchors.sort(key=lambda a: a[0])
            self.anchorsEdited.emit([list(a) for a in self._anchors])
        elif self.mode == "region":
            vr = vb.viewRange()[0]
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
        hit = self._panel_at(args[0])
        if hit is not None:
            self.cursorMoved.emit(hit[1], self._data_y(hit[2]))

    def _anchor_moved(self, item):
        p = item.pos()
        i = item.idx
        if i < len(self._anchors):
            shift = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
            manual = shift or self._anchors[i][1] is not None
            self._anchors[i] = [p.x(), self._data_y(p.y()) if manual else None]
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
        self.peakDragged.emit(item.prefix, item.handle, p.x(), self._data_y(p.y()), final)

    def update_preview_curve(self, xy):
        """``xy`` in data coordinates (the display transform is applied here)."""
        if xy is not None:
            a, b = self._y_map
            xy = (xy[0], a * np.asarray(xy[1]) + b)
        for layer in self._layers():
            layer.set_preview(xy)
