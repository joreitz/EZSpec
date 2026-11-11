"""Window shell in the manner of a code editor: activity bar and welcome page."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

ICON_COLOR = "#4a5160"
ACCENT = "#2f6fd0"


def _icon(kind: str, size: int = 24) -> QtGui.QIcon:
    """Line icon drawn with the painter (no icon files needed, sharp at every scale)."""
    icon = QtGui.QIcon()
    for mode, color in ((QtGui.QIcon.Normal, ICON_COLOR), (QtGui.QIcon.Active, ACCENT),
                        (QtGui.QIcon.Selected, ACCENT)):
        for state, col in ((QtGui.QIcon.Off, color), (QtGui.QIcon.On, ACCENT)):
            pm = QtGui.QPixmap(size * 2, size * 2)
            pm.setDevicePixelRatio(2.0)
            pm.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(pm)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(QtGui.QPen(QtGui.QColor(col), 1.6, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            p.setBrush(QtCore.Qt.NoBrush)
            _draw(p, kind, float(size))
            p.end()
            icon.addPixmap(pm, mode, state)
    return icon


def _draw(p, kind, s):
    u = s / 24.0
    F = QtCore.QPointF
    R = QtCore.QRectF
    if kind == "explorer":                      # two sheets
        p.drawRoundedRect(R(8 * u, 3 * u, 11 * u, 14 * u), 1.5 * u, 1.5 * u)
        p.drawPolyline([F(8 * u, 7 * u), F(5 * u, 7 * u), F(5 * u, 21 * u), F(16 * u, 21 * u), F(16 * u, 17 * u)])
    elif kind == "data":                        # stacked spectra
        for y0 in (8, 13, 18):
            path = QtGui.QPainterPath(F(4 * u, y0 * u))
            path.lineTo(9 * u, y0 * u)
            path.lineTo(11 * u, (y0 - 4) * u)
            path.lineTo(13 * u, y0 * u)
            path.lineTo(20 * u, y0 * u)
            p.drawPath(path)
    elif kind == "processing":                  # sliders
        for y0, x0 in ((7, 9), (12, 15), (17, 11)):
            p.drawLine(F(4 * u, y0 * u), F(20 * u, y0 * u))
            p.save()
            p.setBrush(QtGui.QColor("white"))
            p.drawEllipse(F(x0 * u, y0 * u), 2.1 * u, 2.1 * u)
            p.restore()
    elif kind == "model":                       # a peak on a baseline
        path = QtGui.QPainterPath(F(3 * u, 19 * u))
        path.cubicTo(F(9 * u, 19 * u), F(10 * u, 5 * u), F(12 * u, 5 * u))
        path.cubicTo(F(14 * u, 5 * u), F(15 * u, 19 * u), F(21 * u, 19 * u))
        p.drawPath(path)
        p.drawLine(F(3 * u, 21.5 * u), F(21 * u, 21.5 * u))
    elif kind == "results":                     # table
        p.drawRoundedRect(R(4 * u, 5 * u, 16 * u, 14 * u), 1.5 * u, 1.5 * u)
        p.drawLine(F(4 * u, 9.5 * u), F(20 * u, 9.5 * u))
        p.drawLine(F(4 * u, 14 * u), F(20 * u, 14 * u))
        p.drawLine(F(10 * u, 9.5 * u), F(10 * u, 19 * u))
    elif kind == "history":                     # clock
        p.drawEllipse(F(12 * u, 12 * u), 8 * u, 8 * u)
        p.drawPolyline([F(12 * u, 7 * u), F(12 * u, 12 * u), F(15.5 * u, 14 * u)])
    elif kind == "pages":                       # sheet with frames
        p.drawRoundedRect(R(5 * u, 3 * u, 14 * u, 18 * u), 1.5 * u, 1.5 * u)
        p.drawRect(R(7.5 * u, 6 * u, 4 * u, 4 * u))
        p.drawRect(R(13 * u, 6 * u, 3.5 * u, 4 * u))
        p.drawRect(R(7.5 * u, 12.5 * u, 9 * u, 5.5 * u))
    elif kind == "search":                      # magnifier
        p.drawEllipse(F(10.5 * u, 10.5 * u), 5.5 * u, 5.5 * u)
        p.drawLine(F(14.8 * u, 14.8 * u), F(20 * u, 20 * u))


class ActivityBar(QtWidgets.QToolBar):
    """Narrow bar at the left edge: one button per panel (click shows or hides it) and shortcuts
    to things that are always needed."""

    def __init__(self, parent=None):
        super().__init__("Activity bar", parent)
        self.setObjectName("ActivityBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setOrientation(QtCore.Qt.Vertical)
        self.setIconSize(QtCore.QSize(24, 24))
        self.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.setStyleSheet("""
            QToolBar#ActivityBar { background: #e4e7ec; border: none; spacing: 2px; padding: 4px 2px; }
            QToolBar#ActivityBar QToolButton { padding: 7px; border: none; border-left: 2px solid transparent; }
            QToolBar#ActivityBar QToolButton:hover { background: #d7dbe3; }
            QToolBar#ActivityBar QToolButton:checked { border-left: 2px solid #2f6fd0; background: #f4f5f7; }
        """)

    def add_dock(self, dock, kind: str, tip: str):
        """Button that shows / hides a dock (stays in sync when the dock is closed elsewhere)."""
        action = dock.toggleViewAction()
        action.setIcon(_icon(kind))
        action.setToolTip(tip)
        self.addAction(action)
        return action

    def add_command(self, kind: str, tip: str, slot):
        action = QtGui.QAction(_icon(kind), tip, self)
        action.setToolTip(tip)
        action.triggered.connect(lambda _=False: slot())
        self.addAction(action)
        return action

    def add_stretch(self):
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.addWidget(spacer)


class WelcomePage(QtWidgets.QWidget):
    """Shown instead of an empty plot: what to do first, and what was open last."""
    openFolderRequested = QtCore.Signal()
    openProjectRequested = QtCore.Signal()
    importRequested = QtCore.Signal()
    exampleRequested = QtCore.Signal()
    recentRequested = QtCore.Signal(str)
    helpRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#ffffff"))
        self.setPalette(pal)
        outer = QtWidgets.QHBoxLayout(self)
        outer.addStretch(1)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(6)
        outer.addLayout(col, 0)
        outer.addStretch(1)
        col.addStretch(2)
        title = QtWidgets.QLabel("EZSpec")
        title.setStyleSheet("font-size: 30px; font-weight: 300; color: #2b2b2b;")
        col.addWidget(title)
        sub = QtWidgets.QLabel("Spectral analysis and curve fitting with honest statistics")
        sub.setStyleSheet("font-size: 14px; color: #5d6470;")
        col.addWidget(sub)
        col.addSpacing(18)
        col.addWidget(self._heading("Start"))
        for text, signal, key in (("Open Folder…", self.openFolderRequested, "Ctrl+Shift+O"),
                                  ("Open Project…", self.openProjectRequested, "Ctrl+O"),
                                  ("Import a Data File…", self.importRequested, "Ctrl+I"),
                                  ("Load Example Data", self.exampleRequested, ""),
                                  ("Quick Start", self.helpRequested, "F1")):
            col.addWidget(self._link(text, signal.emit, key))
        col.addSpacing(18)
        col.addWidget(self._heading("Recent"))
        self.recent_box = QtWidgets.QVBoxLayout()
        self.recent_box.setSpacing(4)
        col.addLayout(self.recent_box)
        col.addSpacing(18)
        tips = QtWidgets.QLabel("Ctrl+P  go to a file or dataset   ·   Ctrl+Shift+P  all commands   ·   "
                                "Ctrl+B  hide the side bar\nFiles and folders can be dropped onto the window.")
        tips.setStyleSheet("color: #5d6470;")
        col.addWidget(tips)
        col.addStretch(3)
        self.set_recent([])

    @staticmethod
    def _heading(text):
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet("font-size: 16px; color: #2b2b2b; margin-bottom: 2px;")
        return lab

    def _link(self, text, slot, key=""):
        b = QtWidgets.QPushButton(text if not key else f"{text}    {key}")
        b.setFlat(True)
        b.setCursor(QtCore.Qt.PointingHandCursor)
        b.setStyleSheet("QPushButton { text-align: left; color: #2f6fd0; border: none; padding: 2px 0; }"
                        "QPushButton:hover { text-decoration: underline; }")
        b.clicked.connect(lambda _=False: slot())
        return b

    def set_recent(self, paths):
        """Recently opened folders and projects (those that still exist)."""
        while self.recent_box.count():
            w = self.recent_box.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self.recent_buttons = []
        for p in paths:
            path = Path(p)
            where = str(path.parent)
            if len(where) > 48:                         # long paths must not widen the window
                where = "…" + where[-47:]
            b = self._link(f"{path.name}    {where}", lambda p=str(path): self.recentRequested.emit(p))
            b.setToolTip(str(path))
            self.recent_box.addWidget(b)
            self.recent_buttons.append(b)
        if not self.recent_buttons:
            lab = QtWidgets.QLabel("Nothing opened yet.")
            lab.setStyleSheet("color: #5d6470;")
            self.recent_box.addWidget(lab)
