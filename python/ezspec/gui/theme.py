"""Colours, palette and pyqtgraph defaults."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

COLORS = {
    "data": "#2b2b2b",
    "input": "#a0a0a0",
    "excluded": "#c8c8c8",
    "baseline": "#e76300",
    "smoothed": "#3f90da",
    "fit": "#bd1f01",
    "preview": "#bd1f01",
    "anchor": "#e76300",
    "peak": "#832db6",
    "region_exclude": (189, 31, 1, 40),
    "region_force": (63, 144, 218, 45),
    "region_generic": (120, 120, 120, 35),
    "resid": "#3c3c3c",
}
COMPONENT_COLORS = ["#3f90da", "#ffa90e", "#832db6", "#a96b59", "#94a4a2", "#e76300", "#b9ac70",
                    "#717581", "#92dadd", "#bd1f01"]

SEVERITY_ICON = {"info": "ℹ", "warning": "⚠", "error": "✖"}
SEVERITY_COLOR = {"info": "#3f6fb5", "warning": "#b36b00", "error": "#b3261e"}


def configure_pyqtgraph():
    pg.setConfigOptions(antialias=True, background="w", foreground="#303030", useNumba=False)


def apply_palette(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    base = QtGui.QColor("#ffffff")
    window = QtGui.QColor("#f4f5f7")
    text = QtGui.QColor("#1f1f1f")
    accent = QtGui.QColor("#2f6fd0")
    pal.setColor(QtGui.QPalette.Window, window)
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, base)
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f0f2f5"))
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#fbfbfc"))
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.Highlight, accent)
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#ffffe8"))
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    app.setPalette(pal)
    app.setStyleSheet("""
        QDockWidget::title { padding: 4px 6px; background: #e9ecf1; font-weight: 600; }
        QGroupBox { font-weight: 600; margin-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }
        QToolBar { spacing: 3px; }
        QPushButton#primary { background: #2f6fd0; color: white; font-weight: 600; padding: 5px 14px;
                              border-radius: 4px; }
        QPushButton#primary:disabled { background: #9db6e0; }
        QLabel#hint { color: #5d6470; }
        QLabel#stale { color: #b36b00; font-weight: 600; }
        QTableWidget { gridline-color: #e1e4e8; }
    """)
