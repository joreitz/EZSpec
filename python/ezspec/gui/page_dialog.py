"""Page composer: captured views are placed, resized and aligned freely on a sheet.

The sheet is drawn in millimetres. Every frame shows what the export will
contain (rendered by matplotlib at screen resolution); moving a frame does not
render anything, resizing or changing its options renders that frame again.
"""

from __future__ import annotations

import copy
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from ..export.page import MARGIN, MIN_SIZE, PAGE_SIZES, align, check_page, new_page, render_page, save_page
from ..view import NORMALIZE

HANDLE_PX = 9            # size of the resize handles on screen
SNAP_PX = 7              # distance on screen within which an edge snaps to another edge
ACCENT = "#2f6fd0"


class FrameItem(QtWidgets.QGraphicsObject):
    """A frame on the sheet: position and size in mm, movable and resizable with the mouse."""
    edited = QtCore.Signal(object, bool)             # (item, size changed) after a drag

    def __init__(self, frame: dict, composer):
        super().__init__()
        self.frame = frame
        self.composer = composer
        self.pixmap = None
        self._drag = None                            # (mode, press scene pos, start rects of the moved items)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.sync()

    # ------------------------------------------------------------------ geometry
    def sync(self):
        """Take position and size from the frame."""
        self.prepareGeometryChange()
        x, y, self.w, self.h = self.frame["rect"]
        self.setPos(x, y)
        self.update()

    def _px(self) -> float:
        """One screen pixel in mm."""
        views = self.scene().views() if self.scene() else []
        return 1.0 / views[0].transform().m11() if views else 0.3

    def boundingRect(self):
        m = HANDLE_PX * self._px()
        return QtCore.QRectF(-m, -m, self.w + 2 * m, self.h + 2 * m)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addRect(self.boundingRect() if self.isSelected() else QtCore.QRectF(0, 0, self.w, self.h))
        return path

    def _handles(self) -> dict:
        s = HANDLE_PX * self._px()
        pts = {"tl": (0, 0), "tr": (self.w, 0), "bl": (0, self.h), "br": (self.w, self.h),
               "t": (self.w / 2, 0), "b": (self.w / 2, self.h), "l": (0, self.h / 2), "r": (self.w, self.h / 2)}
        return {k: QtCore.QRectF(x - s / 2, y - s / 2, s, s) for k, (x, y) in pts.items()}

    def _handle_at(self, pos):
        if not self.isSelected():
            return None
        return next((k for k, r in self._handles().items() if r.contains(pos)), None)

    def paint(self, painter, _option, _widget=None):
        rect = QtCore.QRectF(0, 0, self.w, self.h)
        if self.pixmap is not None:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            painter.drawPixmap(rect, self.pixmap, QtCore.QRectF(self.pixmap.rect()))
        else:
            painter.fillRect(rect, QtGui.QColor("#f4f5f7"))
        pen = QtGui.QPen(QtGui.QColor(ACCENT if self.isSelected() else "#c4c8d0"), 0,
                         QtCore.Qt.SolidLine if self.isSelected() else QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rect)
        if self.isSelected():
            painter.setBrush(QtGui.QColor("white"))
            painter.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 0))
            for r in self._handles().values():
                painter.drawRect(r)

    # ------------------------------------------------------------------ mouse
    _CURSORS = {"tl": QtCore.Qt.SizeFDiagCursor, "br": QtCore.Qt.SizeFDiagCursor, "tr": QtCore.Qt.SizeBDiagCursor,
                "bl": QtCore.Qt.SizeBDiagCursor, "t": QtCore.Qt.SizeVerCursor, "b": QtCore.Qt.SizeVerCursor,
                "l": QtCore.Qt.SizeHorCursor, "r": QtCore.Qt.SizeHorCursor}

    def hoverMoveEvent(self, ev):
        self.setCursor(self._CURSORS.get(self._handle_at(ev.pos()), QtCore.Qt.SizeAllCursor))

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(ev)
            return
        handle = self._handle_at(ev.pos())
        if not self.isSelected():
            if not ev.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
                self.scene().clearSelection()
            self.setSelected(True)
        elif ev.modifiers() & QtCore.Qt.ControlModifier and handle is None:
            self.setSelected(False)
            ev.accept()
            return
        moved = [self] if handle else [it for it in self.scene().selectedItems() if isinstance(it, FrameItem)]
        self._drag = (handle or "move", ev.scenePos(), {it: list(it.frame["rect"]) for it in moved})
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        mode, start, rects = self._drag
        d = ev.scenePos() - start
        snap = not ev.modifiers() & QtCore.Qt.AltModifier
        x, y, w, h = rects[self]
        if mode == "move":
            nx, ny = self.composer.snap_move(self, x + d.x(), y + d.y(), w, h, snap)
            for it, (ox, oy, _w, _h) in rects.items():
                it.frame["rect"][0], it.frame["rect"][1] = ox + nx - x, oy + ny - y
                it.sync()
        else:
            x0, y0, x1, y1 = x, y, x + w, y + h
            if "l" in mode:
                x0 = min(self.composer.snap_edge(self, x0 + d.x(), "x", snap), x1 - MIN_SIZE)
            if "r" in mode:
                x1 = max(self.composer.snap_edge(self, x1 + d.x(), "x", snap), x0 + MIN_SIZE)
            if "t" in mode:
                y0 = min(self.composer.snap_edge(self, y0 + d.y(), "y", snap), y1 - MIN_SIZE)
            if "b" in mode:
                y1 = max(self.composer.snap_edge(self, y1 + d.y(), "y", snap), y0 + MIN_SIZE)
            self.frame["rect"][:] = [x0, y0, x1 - x0, y1 - y0]
            self.sync()
        self.composer.geometry_changing(self)

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            mode, _start, rects = self._drag
            self._drag = None
            for it, old in rects.items():
                if it.frame["rect"] != old:
                    it.edited.emit(it, mode != "move")
        ev.accept()


class _SheetView(QtWidgets.QGraphicsView):
    resized = QtCore.Signal()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.resized.emit()


class PageComposer(QtWidgets.QDialog):
    """Window that edits the pages of a project (``pages``: the list stored in the project)."""
    captureRequested = QtCore.Signal()               # "add the current view of the main window"
    changed = QtCore.Signal()                        # a page was edited

    def __init__(self, pages: list, resolve, names=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pages")
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.resize(1250, 780)
        self.resolve = resolve
        self.names = names or (lambda ids: ", ".join(ids))
        self.pages = pages
        self.items: list = []
        self._dirty: set = set()
        self._loading = False

        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.page_box = QtWidgets.QComboBox()
        self.page_box.setMinimumWidth(130)
        self.page_box.currentIndexChanged.connect(self._page_selected)
        bar.addWidget(self.page_box)
        for text, slot, tip in (("New page", self.add_page, "Add an empty page"),
                                ("Delete page", self.delete_page, "Delete this page")):
            bar.addWidget(self._button(text, slot, tip))
        bar.addSpacing(14)
        add = self._button("+ Add current view", self.captureRequested.emit,
                           "Capture what the main window shows now as a frame (with its zoom regions)")
        add.setObjectName("primary")
        bar.addWidget(add)
        bar.addWidget(self._button("Duplicate", self.duplicate_selected, "Copy the selected frames (Ctrl+D)"))
        bar.addWidget(self._button("Delete", self.delete_selected, "Delete the selected frames (Del)"))
        self.align_btn = QtWidgets.QToolButton()
        self.align_btn.setText("Align")
        self.align_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.align_btn.setToolTip("Align or distribute the selected frames (select several with Ctrl+click or by "
                                  "dragging a box)")
        menu = QtWidgets.QMenu(self.align_btn)
        for how, title in (("left", "Left edges"), ("right", "Right edges"), ("top", "Top edges"),
                           ("bottom", "Bottom edges"), ("hcenter", "Centres horizontally"),
                           ("vcenter", "Centres vertically"), (None, None), ("width", "Same width (as the first)"),
                           ("height", "Same height (as the first)"), (None, None),
                           ("hspace", "Distribute horizontally"), ("vspace", "Distribute vertically")):
            if how is None:
                menu.addSeparator()
            else:
                menu.addAction(title, lambda h=how: self.align_selected(h))
        self.align_btn.setMenu(menu)
        bar.addWidget(self.align_btn)
        bar.addStretch()
        export = self._button("Export…", self.export, "Write the page as PDF, SVG or PNG")
        bar.addWidget(export)
        lay.addLayout(bar)

        split = QtWidgets.QSplitter()
        lay.addWidget(split, 1)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.view = _SheetView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.view.setBackgroundBrush(QtGui.QColor("#d9dce2"))
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.view.resized.connect(self._fit)
        split.addWidget(self.view)
        self.sheet = self.scene.addRect(0, 0, 1, 1, QtGui.QPen(QtGui.QColor("#9aa0aa"), 0), QtGui.QBrush("white"))
        self.sheet.setZValue(-10)
        self.margin = self.scene.addRect(0, 0, 1, 1, QtGui.QPen(QtGui.QColor("#e3e6eb"), 0, QtCore.Qt.DotLine))
        self.margin.setZValue(-9)

        side = QtWidgets.QWidget()
        side.setMinimumWidth(270)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 0, 0)
        split.addWidget(side)
        split.setStretchFactor(0, 1)

        pg = QtWidgets.QGroupBox("Page")
        pf = QtWidgets.QFormLayout(pg)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(lambda: self._set_page("name", self.name_edit.text().strip() or "Page"))
        pf.addRow("Name", self.name_edit)
        self.size_box = QtWidgets.QComboBox()
        for key, (title, _w, _h) in PAGE_SIZES.items():
            self.size_box.addItem(title, key)
        self.size_box.addItem("Custom", "custom")
        self.size_box.activated.connect(self._size_chosen)
        pf.addRow("Size", self.size_box)
        self.page_w, self.page_h = self._spin(30, 1200, " mm"), self._spin(30, 1200, " mm")
        self.page_w.valueChanged.connect(lambda v: self._set_page("width_mm", v, size="custom"))
        self.page_h.valueChanged.connect(lambda v: self._set_page("height_mm", v, size="custom"))
        wh = QtWidgets.QHBoxLayout()
        wh.addWidget(self.page_w)
        wh.addWidget(QtWidgets.QLabel("×"))
        wh.addWidget(self.page_h)
        pf.addRow("W × H", wh)
        self.font_spin = self._spin(4, 40, " pt", 0.5)
        self.font_spin.valueChanged.connect(lambda v: self._set_page("font_size", v))
        pf.addRow("Font size", self.font_spin)
        self.lw_spin = self._spin(0.2, 6, " pt", 0.1)
        self.lw_spin.valueChanged.connect(lambda v: self._set_page("line_width", v))
        pf.addRow("Line width", self.lw_spin)
        self.labels_check = QtWidgets.QCheckBox("Label the frames (a), (b), …")
        self.labels_check.toggled.connect(lambda on: self._set_page("panel_labels", bool(on)))
        pf.addRow(self.labels_check)
        sl.addWidget(pg)

        self.frame_group = QtWidgets.QGroupBox("Selected frame")
        ff = QtWidgets.QFormLayout(self.frame_group)
        self.data_label = QtWidgets.QLabel()
        self.data_label.setWordWrap(True)
        self.data_label.setObjectName("hint")
        ff.addRow(self.data_label)
        self.x_from, self.x_to = self._range_edit("full range"), self._range_edit("full range")
        self.y_from, self.y_to = self._range_edit("automatic"), self._range_edit("automatic")
        for title, a, b, key in (("x from / to", self.x_from, self.x_to, "x_range"),
                                 ("y from / to", self.y_from, self.y_to, "y_range")):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(a)
            row.addWidget(b)
            ff.addRow(title, row)
            for e in (a, b):
                e.editingFinished.connect(lambda k=key, a=a, b=b: self._range_edited(k, a, b))
        self.checks = {}
        for key, title in (("fit", "Fit"), ("components", "Components"), ("legend", "Legend"),
                           ("labels", "Axis labels"), ("stacked", "Stacked")):
            c = QtWidgets.QCheckBox(title)
            c.toggled.connect(lambda on, k=key: self._set_frames(k, bool(on)))
            self.checks[key] = c
        grid = QtWidgets.QGridLayout()
        for i, c in enumerate(self.checks.values()):
            grid.addWidget(c, i // 2, i % 2)
        ff.addRow(grid)
        self.norm_box = QtWidgets.QComboBox()
        for key, title in NORMALIZE.items():
            self.norm_box.addItem(title, key)
        self.norm_box.activated.connect(lambda _i: self._set_frames("normalize", self.norm_box.currentData()))
        ff.addRow("Normalise", self.norm_box)
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.editingFinished.connect(lambda: self._set_frames("title", self.title_edit.text()))
        ff.addRow("Title", self.title_edit)
        self.geo = [self._spin(-2000, 2000, " mm") for _ in range(4)]
        for spin in self.geo:
            spin.valueChanged.connect(self._geometry_edited)
        g1, g2 = QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout()
        g1.addWidget(self.geo[0])
        g1.addWidget(self.geo[1])
        g2.addWidget(self.geo[2])
        g2.addWidget(self.geo[3])
        ff.addRow("x, y", g1)
        ff.addRow("W × H", g2)
        sl.addWidget(self.frame_group)
        hint = QtWidgets.QLabel("Drag a frame to move it, its handles to resize it. Edges snap to other frames, to "
                                "the margin and to full millimetres (Alt: free). The frames follow the project: "
                                "after a new fit they show the new fit.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        sl.addWidget(hint)
        sl.addStretch()

        QtGui.QShortcut(QtGui.QKeySequence.Delete, self, self.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self.duplicate_selected)
        QtGui.QShortcut(QtGui.QKeySequence.SelectAll, self, self.select_all)
        self._render_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._render_timer.timeout.connect(self._render_dirty)
        self.set_pages(pages)

    # ------------------------------------------------------------------ small helpers
    def _button(self, text, slot, tip):
        b = QtWidgets.QPushButton(text)
        b.setToolTip(tip)
        b.setAutoDefault(False)
        b.clicked.connect(lambda _=False: slot())
        return b

    @staticmethod
    def _spin(lo, hi, suffix, step=1.0):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(1)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        return s

    @staticmethod
    def _range_edit(placeholder):
        e = QtWidgets.QLineEdit()
        e.setPlaceholderText(placeholder)
        return e

    # ------------------------------------------------------------------ pages
    @property
    def page(self) -> dict:
        return self.pages[self.page_box.currentIndex()]

    def set_pages(self, pages: list, index: int | None = None):
        """Show the pages of a project (always at least one)."""
        self.pages = pages
        if not pages:
            pages.append(new_page())
        for p in pages:
            check_page(p)
        index = self.page_box.currentIndex() if index is None else index
        self.page_box.blockSignals(True)
        self.page_box.clear()
        self.page_box.addItems([p["name"] for p in pages])
        self.page_box.setCurrentIndex(min(max(index, 0), len(pages) - 1))
        self.page_box.blockSignals(False)
        self._load_page()

    def _page_selected(self, _i):
        if self.page_box.currentIndex() >= 0:
            self._load_page()

    def add_page(self):
        names = {p["name"] for p in self.pages}
        n = len(self.pages) + 1
        while f"Page {n}" in names:
            n += 1
        page = new_page(f"Page {n}", self.page.get("size", "a4_landscape"))
        for key in ("width_mm", "height_mm", "font_size", "font_family", "line_width"):
            page[key] = self.page[key]
        self.pages.append(page)
        self.set_pages(self.pages, len(self.pages) - 1)
        self.changed.emit()

    def delete_page(self):
        if self.page["frames"]:
            r = QtWidgets.QMessageBox.question(self, "Delete page", f"Delete “{self.page['name']}” with its "
                                                                    f"{len(self.page['frames'])} frame(s)?")
            if r != QtWidgets.QMessageBox.Yes:
                return
        del self.pages[self.page_box.currentIndex()]
        self.set_pages(self.pages, self.page_box.currentIndex() - 1)
        self.changed.emit()

    def _load_page(self):
        page = self.page
        for it in self.items:
            self.scene.removeItem(it)
        self.items = []
        self._dirty.clear()
        self._loading = True
        self.name_edit.setText(page["name"])
        self.size_box.setCurrentIndex(max(0, self.size_box.findData(page.get("size", "custom"))))
        self.page_w.setValue(page["width_mm"])
        self.page_h.setValue(page["height_mm"])
        self.font_spin.setValue(page["font_size"])
        self.lw_spin.setValue(page["line_width"])
        self.labels_check.setChecked(bool(page["panel_labels"]))
        self._loading = False
        for frame in page["frames"]:
            self._add_item(frame)
        self._fit()
        self._selection_changed()

    def _add_item(self, frame) -> FrameItem:
        it = FrameItem(frame, self)
        it.edited.connect(self._item_edited)
        self.scene.addItem(it)
        self.items.append(it)
        self._mark(it)
        return it

    def _fit(self):
        page = self.page
        W, H = page["width_mm"], page["height_mm"]
        self.sheet.setRect(0, 0, W, H)
        self.margin.setRect(MARGIN, MARGIN, max(1.0, W - 2 * MARGIN), max(1.0, H - 2 * MARGIN))
        pad = 0.04 * max(W, H)
        self.scene.setSceneRect(-pad, -pad, W + 2 * pad, H + 2 * pad)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        for it in self.items:                        # the scale changed: handles and previews change with it
            it.prepareGeometryChange()
            self._mark(it)

    def _set_page(self, key, value, size=None):
        if self._loading or self.page.get(key) == value:
            return
        self.page[key] = value
        if size is not None:
            self.page["size"] = size
            self.size_box.setCurrentIndex(self.size_box.findData(size))
        if key == "name":
            self.page_box.setItemText(self.page_box.currentIndex(), value)
        self._fit()
        self.changed.emit()

    def _size_chosen(self, _i):
        key = self.size_box.currentData()
        if key in PAGE_SIZES:
            _title, w, h = PAGE_SIZES[key]
            self.page.update(size=key, width_mm=w, height_mm=h)
            self._loading = True
            self.page_w.setValue(w)
            self.page_h.setValue(h)
            self._loading = False
            self._fit()
            self.changed.emit()

    # ------------------------------------------------------------------ frames
    def selected(self) -> list:
        return [it for it in self.items if it.isSelected()]

    def select_all(self):
        for it in self.items:
            it.setSelected(True)

    def add_frames(self, frames):
        """Frames that were appended to the current page (by a capture) appear and get selected."""
        self.scene.clearSelection()
        for frame in frames:
            self._add_item(frame).setSelected(True)
        self.changed.emit()

    def duplicate_selected(self):
        new = []
        for it in self.selected():
            frame = copy.deepcopy(it.frame)
            frame["id"] = uuid.uuid4().hex[:8]
            frame["rect"][0] += 6.0
            frame["rect"][1] += 6.0
            self.page["frames"].append(frame)
            new.append(frame)
        if new:
            self.add_frames(new)

    def delete_selected(self):
        gone = self.selected()
        for it in gone:
            self.page["frames"].remove(it.frame)
            self.items.remove(it)
            self.scene.removeItem(it)
            self._dirty.discard(it)
        if gone:
            self._relabel()
            self.changed.emit()

    def align_selected(self, how):
        sel = self.selected()
        if len(sel) < 2:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Select at least two frames (Ctrl+click).")
            return
        align([it.frame for it in sel], how)
        for it in sel:
            it.sync()
            if how in ("width", "height"):
                self._mark(it)
        self._selection_changed()
        self.changed.emit()

    def _relabel(self):
        if self.page.get("panel_labels"):
            for it in self.items:
                self._mark(it)

    # snapping: edges of the other frames, their centres, the sheet and its margin
    def _guides(self, item, axis) -> list:
        i, j = (0, 2) if axis == "x" else (1, 3)
        size = self.page["width_mm" if axis == "x" else "height_mm"]
        out = [0.0, MARGIN, size - MARGIN, size, size / 2]
        moving = {it for it in self.items if it.isSelected()} | {item}
        for it in self.items:
            if it not in moving:
                r = it.frame["rect"]
                out += [r[i], r[i] + r[j], r[i] + r[j] / 2]
        return out

    def snap_edge(self, item, value, axis, snap=True) -> float:
        if not snap:
            return value
        tol = SNAP_PX * item._px()
        best = min(self._guides(item, axis), key=lambda g: abs(g - value))
        return best if abs(best - value) <= tol else float(round(value))

    def snap_move(self, item, x, y, w, h, snap=True) -> tuple:
        if not snap:
            return x, y
        out = []
        for pos, size, axis in ((x, w, "x"), (y, h, "y")):
            tol = SNAP_PX * item._px()
            guides = self._guides(item, axis)
            best = None
            for offset in (0.0, size, size / 2):     # leading edge, trailing edge, centre
                g = min(guides, key=lambda v: abs(v - (pos + offset)))
                d = g - (pos + offset)
                if abs(d) <= tol and (best is None or abs(d) < abs(best)):
                    best = d
            out.append(pos + best if best is not None else float(round(pos)))
        return out[0], out[1]

    def geometry_changing(self, item):
        if len(self.selected()) == 1:
            self._show_geometry(item.frame)

    def _item_edited(self, item, resized):
        if resized:
            self._mark(item)
        self.changed.emit()

    # ------------------------------------------------------------------ properties of the selection
    def _selection_changed(self):
        sel = self.selected()
        self.frame_group.setEnabled(bool(sel))
        self.frame_group.setTitle("Selected frame" if len(sel) <= 1 else f"{len(sel)} selected frames")
        if not sel:
            self.data_label.setText("Select a frame on the page.")
            return
        frame = sel[0].frame
        self._loading = True
        ids = [frame["active"]] + list(frame["shown"])
        self.data_label.setText(self.names(ids) + (" – processing step" if frame.get("step") else ""))
        for (a, b), key in (((self.x_from, self.x_to), "x_range"), ((self.y_from, self.y_to), "y_range")):
            r = frame.get(key)
            a.setText("" if not r else f"{r[0]:.6g}")
            b.setText("" if not r else f"{r[1]:.6g}")
        for key, c in self.checks.items():
            c.setChecked(bool(frame[key]))
        self.norm_box.setCurrentIndex(max(0, self.norm_box.findData(frame["normalize"])))
        self.title_edit.setText(frame.get("title", ""))
        self._loading = False
        self._show_geometry(frame)
        for spin in self.geo:
            spin.setEnabled(len(sel) == 1)

    def _show_geometry(self, frame):
        self._loading = True
        for spin, v in zip(self.geo, frame["rect"]):
            spin.setValue(v)
        self._loading = False

    def _set_frames(self, key, value):
        if self._loading:
            return
        touched = False
        for it in self.selected():
            if it.frame.get(key) != value:
                it.frame[key] = value
                self._mark(it)
                touched = True
        if touched:
            self.changed.emit()

    def _range_edited(self, key, a, b):
        def num(e):
            try:
                return float(e.text().replace(",", "."))
            except ValueError:
                return None
        lo, hi = num(a), num(b)
        self._set_frames(key, None if lo is None or hi is None or lo == hi else [min(lo, hi), max(lo, hi)])

    def _geometry_edited(self, _v):
        sel = self.selected()
        if self._loading or len(sel) != 1:
            return
        x, y, w, h = (s.value() for s in self.geo)
        rect = [x, y, max(w, MIN_SIZE), max(h, MIN_SIZE)]
        if rect != sel[0].frame["rect"]:
            resized = rect[2:] != sel[0].frame["rect"][2:]
            sel[0].frame["rect"][:] = rect
            sel[0].sync()
            self._item_edited(sel[0], resized)

    # ------------------------------------------------------------------ rendering
    def _mark(self, item):
        self._dirty.add(item)
        self._render_timer.start()

    def refresh(self):
        """The project changed (new fit, other processing, colours): draw every frame again."""
        for it in self.items:
            self._mark(it)

    def _render_dirty(self):
        if not self.isVisible():
            return                                   # rendered when the window is shown
        dirty, self._dirty = [it for it in self.items if it in self._dirty], set()
        scale = self.view.transform().m11() * self.devicePixelRatioF()       # px per mm
        page = self.page
        for it in dirty:
            try:
                fig = render_page(page, self.resolve, only=it.frame)
                if page.get("panel_labels"):
                    k = page["frames"].index(it.frame)
                    fig.axes[0].text(0.02, 0.97, f"({chr(97 + k % 26)})", transform=fig.axes[0].transAxes,
                                     ha="left", va="top", fontweight="bold", fontsize=page["font_size"])
                fig.set_dpi(max(30.0, scale * 25.4))
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                img = QtGui.QImage(fig.canvas.buffer_rgba(), w, h, QtGui.QImage.Format_RGBA8888).copy()
                it.pixmap = QtGui.QPixmap.fromImage(img)
            except Exception as exc:  # noqa: BLE001 - a frame that cannot be drawn must not block the others
                it.pixmap = None
                it.setToolTip(f"Cannot be drawn: {exc}")
            it.update()

    def showEvent(self, ev):
        super().showEvent(ev)
        QtCore.QTimer.singleShot(0, self._fit)

    # ------------------------------------------------------------------ export
    def export(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export page", f"{self.page['name']}.pdf", "PDF (*.pdf);;SVG (*.svg);;PNG (*.png);;EPS (*.eps)")
        if not path:
            return False
        try:
            save_page(self.page, self.resolve, path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export", str(e))
            return False
        self.setWindowTitle(f"Pages – saved {path}")
        return True
