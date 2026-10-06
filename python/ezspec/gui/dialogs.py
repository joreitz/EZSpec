"""Dialogs: import, figure editor, model comparison, bootstrap settings."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QUndoCommand, QUndoStack

from .. import units as U
from ..export.figure import (PRESET_NOTE, PRESETS, curves_for, default_spec, get_path, make_resolver,
                             param_box_lines, render_figure, save_figure, set_path)
from ..fit import compare
from ..fit.compare import ComparisonError
from ..io import read_spectra, sniff
from ..io.text import parse_table
from ..project import Dataset
from .param_form import SciEdit

# ============================================================================ import


class ImportDialog(QtWidgets.QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle(f"Importieren – {self.path.name}")
        self.info = sniff(self.path)
        lay = QtWidgets.QVBoxLayout(self)
        self.datasets = []
        if self.info["format"] == "jcamp":
            specs = read_spectra(self.path)
            s = specs[0]
            j = s.meta.get("jcamp", {})
            text = (f"<b>JCAMP-DX</b> · {j.get('TITLE', '')}<br>Datentyp: {j.get('DATATYPE', '?')} · "
                    f"{s.n} Punkte · x: {j.get('XUNITS', '?')} · y: {j.get('YUNITS', '?')}")
            for w in s.meta.get("import_warnings", []):
                text += f"<br><span style='color:#b36b00'>⚠ {w}</span>"
            lay.addWidget(QtWidgets.QLabel(text))
            self._jcamp = specs
        else:
            self._jcamp = None
            top = QtWidgets.QFormLayout()
            self.delim = QtWidgets.QComboBox()
            for label, val in (("automatisch", "auto"), ("Tab", "\t"), ("Semikolon ;", ";"), ("Komma ,", ","),
                               ("Leerraum", None)):
                self.delim.addItem(label, val)
            self.decimal = QtWidgets.QComboBox()
            for label, val in (("automatisch", "auto"), ("Punkt .", "."), ("Komma ,", ",")):
                self.decimal.addItem(label, val)
            top.addRow("Trennzeichen", self.delim)
            top.addRow("Dezimalzeichen", self.decimal)
            lay.addLayout(top)
            self.table = QtWidgets.QTableWidget()
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            lay.addWidget(self.table, 1)
            cols = QtWidgets.QFormLayout()
            self.xcol = QtWidgets.QComboBox()
            self.ycols = QtWidgets.QListWidget()
            self.ycols.setMaximumHeight(110)
            self.scol = QtWidgets.QComboBox()
            self.vars = QtWidgets.QListWidget()
            self.vars.setMaximumHeight(80)
            self.vars.setToolTip("weitere unabhängige Variablen für Formeln mit mehreren Prädiktoren")
            self.unit = QtWidgets.QComboBox()
            self.unit.addItem("(keine/unbekannt)", "")
            for k, v in U.UNITS.items():
                self.unit.addItem(v, k)
            cols.addRow("x-Spalte", self.xcol)
            cols.addRow("y-Spalte(n)", self.ycols)
            cols.addRow("σ-Spalte (optional)", self.scol)
            cols.addRow("zusätzl. Variablen", self.vars)
            cols.addRow("x-Einheit", self.unit)
            lay.addLayout(cols)
            self.status = QtWidgets.QLabel()
            self.status.setObjectName("hint")
            lay.addWidget(self.status)
            self.delim.currentIndexChanged.connect(self._reparse)
            self.decimal.currentIndexChanged.connect(self._reparse)
            self._reparse()
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(640, 600)

    def _reparse(self):
        from ..io.text import decode_bytes, guess_unit
        text, _ = decode_bytes(self.path.read_bytes())
        try:
            t = parse_table(text, delimiter=self.delim.currentData(), decimal=self.decimal.currentData())
        except ValueError as e:
            self.status.setText(f"✖ {e}")
            self.table.setRowCount(0)
            return
        self.tinfo = t
        n, m = t.data.shape
        self.table.setColumnCount(m)
        self.table.setRowCount(min(n, 30))
        self.table.setHorizontalHeaderLabels(t.header)
        for i in range(min(n, 30)):
            for j in range(m):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{t.data[i, j]:.8g}"))
        self.xcol.clear()
        self.scol.clear()
        self.ycols.clear()
        self.vars.clear()
        self.scol.addItem("—", None)
        for j, h in enumerate(t.header):
            self.xcol.addItem(h, j)
            self.scol.addItem(h, j)
            it = QtWidgets.QListWidgetItem(h)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if j == 1 or (m > 2 and j > 0) else QtCore.Qt.Unchecked)
            self.ycols.addItem(it)
            v = QtWidgets.QListWidgetItem(h)
            v.setFlags(v.flags() | QtCore.Qt.ItemIsUserCheckable)
            v.setCheckState(QtCore.Qt.Unchecked)
            self.vars.addItem(v)
        if m == 1:
            self.ycols.item(0).setCheckState(QtCore.Qt.Checked)
        guess = guess_unit(t.header[0]) if t.header else ""
        self.unit.setCurrentIndex(max(self.unit.findData(guess), 0))
        self.status.setText(f"{n} Datenzeilen, {m} Spalten, {t.n_header_lines} Kopfzeilen, "
                            f"{t.skipped} übersprungen · Trennzeichen "
                            f"{'Leerraum' if t.delimiter is None else repr(t.delimiter)} · Dezimal {t.decimal!r}")

    def _accept(self):
        try:
            if self._jcamp is not None:
                specs = self._jcamp
            else:
                x = self.xcol.currentData()
                s = self.scol.currentData()
                extra = {self.vars.item(j).text().replace(" ", "_"): j for j in range(self.vars.count())
                         if self.vars.item(j).checkState() == QtCore.Qt.Checked and j != x}
                ys = [j for j in range(self.ycols.count()) if self.ycols.item(j).checkState() == QtCore.Qt.Checked
                      and j not in (x, s) and j not in extra.values()]
                if not ys and self.tinfo.data.shape[1] > 1:
                    raise ValueError("keine y-Spalte gewählt")
                specs = read_spectra(self.path, x_col=x, y_cols=ys or None, sigma_col=s,
                                     delimiter=self.tinfo.delimiter, decimal=self.tinfo.decimal,
                                     x_unit=self.unit.currentData(), extra_cols=extra or None)
            raw = self.path.read_bytes()
            self.datasets = [Dataset(name=sp.meta.get("name", self.path.stem), raw=sp, raw_bytes=raw,
                                     raw_ext=self.path.suffix.lower()) for sp in specs]
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import", str(e))
            return
        self.accept()


# ============================================================================ figure editor
class SetPathCommand(QUndoCommand):
    def __init__(self, dlg, path, old, new):
        super().__init__(f"Figur: {path}")
        self.dlg, self.path, self.old, self.new = dlg, path, old, new

    def redo(self):
        set_path(self.dlg.spec, self.path, copy.deepcopy(self.new))
        self.dlg.spec_changed()

    def undo(self):
        set_path(self.dlg.spec, self.path, copy.deepcopy(self.old))
        self.dlg.spec_changed()


class FigureDialog(QtWidgets.QDialog):
    """Publication figure editor: every change is an undoable Set(path, value) on the JSON spec."""

    LEGEND = [None, "best", "upper right", "upper left", "lower left", "lower right", "center right",
              "upper center", "lower center"]

    def __init__(self, ds, spec=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Abbildung – {ds.name}")
        self.ds = ds
        run = ds.run_pipeline()
        self.processed = run.final
        res = ds.fit_result
        self.curves = {ds.id: curves_for(self.processed, ds.raw, res, ds.model if res is not None else None)}
        xl = U.AXIS_LABELS.get(self.processed.x_unit, self.processed.x_label)
        self.spec = copy.deepcopy(spec) if spec else default_spec(ds.id, self.curves[ds.id], xl,
                                                                  self.processed.y_label or "Intensity")
        self.undo = QUndoStack(self)
        self._timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self._render)
        self._loading = False

        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget()
        left.setMaximumWidth(380)
        lv = QtWidgets.QVBoxLayout(left)
        form = QtWidgets.QFormLayout()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("(benutzerdefiniert)", "custom")
        for k, p in PRESETS.items():
            self.preset.addItem(p["title"], k)
        self.preset.setToolTip(PRESET_NOTE)
        self.width = SciEdit()
        self.height = SciEdit()
        self.font = QtWidgets.QDoubleSpinBox()
        self.font.setRange(4, 30)
        self.font.setSingleStep(0.5)
        self.family = QtWidgets.QComboBox()
        self.family.addItems(["sans-serif", "serif"])
        self.lw = QtWidgets.QDoubleSpinBox()
        self.lw.setRange(0.2, 5)
        self.lw.setSingleStep(0.1)
        self.panel_labels = QtWidgets.QCheckBox("(a), (b) …")
        form.addRow("Vorlage", self.preset)
        form.addRow("Breite / mm", self.width)
        form.addRow("Höhe / mm", self.height)
        form.addRow("Schrift / pt", self.font)
        form.addRow("Schriftart", self.family)
        form.addRow("Linienbreite / pt", self.lw)
        form.addRow("Panel-Label", self.panel_labels)
        lv.addLayout(form)
        note = QtWidgets.QLabel(PRESET_NOTE)
        note.setObjectName("hint")
        note.setWordWrap(True)
        lv.addWidget(note)

        self.panel_sel = QtWidgets.QComboBox()
        lv.addWidget(self.panel_sel)
        pf = QtWidgets.QFormLayout()
        self.xlabel = QtWidgets.QLineEdit()
        self.ylabel = QtWidgets.QLineEdit()
        self.legend = QtWidgets.QComboBox()
        for v in self.LEGEND:
            self.legend.addItem("keine" if v is None else v, v)
        self.invert = QtWidgets.QCheckBox("x invertieren")
        self.ylog = QtWidgets.QCheckBox("y logarithmisch")
        self.grid = QtWidgets.QCheckBox("Gitter")
        self.sec = QtWidgets.QComboBox()
        self.sec.addItem("—", None)
        for k, v in U.UNITS.items():
            self.sec.addItem(v, k)
        self.laser = SciEdit(optional=True)
        self.laser.setPlaceholderText("Laser / nm")
        pf.addRow("x-Titel", self.xlabel)
        pf.addRow("y-Titel", self.ylabel)
        pf.addRow("Legende", self.legend)
        checks = QtWidgets.QHBoxLayout()
        for c in (self.invert, self.ylog, self.grid):
            checks.addWidget(c)
        pf.addRow(checks)
        secrow = QtWidgets.QHBoxLayout()
        secrow.addWidget(self.sec, 1)
        secrow.addWidget(self.laser)
        pf.addRow("2. x-Achse", secrow)
        lv.addLayout(pf)
        self.traces = QtWidgets.QTableWidget(0, 4)
        self.traces.setHorizontalHeaderLabels(["an", "Kurve", "Legende", "Farbe"])
        self.traces.verticalHeader().setVisible(False)
        self.traces.horizontalHeader().setStretchLastSection(True)
        self.traces.setColumnWidth(0, 28)
        self.traces.cellChanged.connect(self._trace_edited)
        self.traces.cellDoubleClicked.connect(self._pick_color)
        lv.addWidget(self.traces, 1)
        self.params = QtWidgets.QListWidget()
        self.params.setMaximumHeight(100)
        self.params.setToolTip("Parameter für das Textfeld im Hauptpanel")
        if res is not None:
            names = list(res.params) + [f"{d.component}.{d.name}" for d in res.derived]
            for n in names:
                it = QtWidgets.QListWidgetItem(n)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Unchecked)
                self.params.addItem(it)
        lv.addWidget(QtWidgets.QLabel("Parameterbox:"))
        lv.addWidget(self.params)
        btns = QtWidgets.QHBoxLayout()
        ub = QtWidgets.QPushButton("↶")
        ub.setToolTip("Rückgängig")
        ub.clicked.connect(self.undo.undo)
        rb = QtWidgets.QPushButton("↷")
        rb.setToolTip("Wiederholen")
        rb.clicked.connect(self.undo.redo)
        ex = QtWidgets.QPushButton("Export…")
        ex.setObjectName("primary")
        ex.setMinimumWidth(90)
        ex.clicked.connect(self._export)
        ok = QtWidgets.QPushButton("Übernehmen")
        ok.clicked.connect(self.accept)
        for b in (ub, rb, ex, ok):
            btns.addWidget(b)
        lv.addLayout(btns)
        lay.addWidget(left)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._canvas_cls = FigureCanvasQTAgg
        self.preview = QtWidgets.QScrollArea()
        self.preview.setWidgetResizable(False)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("QScrollArea { background: #8a8f98; }")
        right = QtWidgets.QVBoxLayout()
        self.zoom = QtWidgets.QComboBox()
        for z in ("100 % (physische Größe)", "150 %", "200 %"):
            self.zoom.addItem(z)
        self.zoom.currentIndexChanged.connect(lambda: self._timer.start())
        right.addWidget(self.zoom)
        right.addWidget(self.preview, 1)
        lay.addLayout(right, 1)

        self._connect()
        self._load_controls()
        self._render()
        self.resize(1200, 760)

    # ------------------------------------------------------------------ wiring
    def _set(self, path, value):
        if self._loading:
            return
        old = copy.deepcopy(get_path(self.spec, path))
        if old == value:
            return
        self.undo.push(SetPathCommand(self, path, old, value))

    def _panel(self):
        return max(self.panel_sel.currentIndex(), 0)

    def _connect(self):
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.width.valueEdited.connect(lambda v: self._set("width_mm", v))
        self.height.valueEdited.connect(lambda v: self._set("height_mm", v))
        self.font.valueChanged.connect(lambda v: self._set("font_size", v))
        self.family.currentTextChanged.connect(lambda v: self._set("font_family", v))
        self.lw.valueChanged.connect(lambda v: self._set("line_width", v))
        self.panel_labels.toggled.connect(lambda v: self._set("panel_labels", v))
        self.panel_sel.currentIndexChanged.connect(lambda _i: self._load_panel())
        self.xlabel.editingFinished.connect(lambda: self._set(f"panels.{self._panel()}.xlabel", self.xlabel.text()))
        self.ylabel.editingFinished.connect(lambda: self._set(f"panels.{self._panel()}.ylabel", self.ylabel.text()))
        self.legend.currentIndexChanged.connect(
            lambda: self._set(f"panels.{self._panel()}.legend", self.legend.currentData()))
        self.invert.toggled.connect(lambda v: self._set(f"panels.{self._panel()}.invert_x", v))
        self.ylog.toggled.connect(lambda v: self._set(f"panels.{self._panel()}.yscale", "log" if v else "linear"))
        self.grid.toggled.connect(lambda v: self._set(f"panels.{self._panel()}.grid", v))
        self.sec.currentIndexChanged.connect(self._secondary_changed)
        self.laser.valueEdited.connect(lambda _v: self._secondary_changed())
        self.params.itemChanged.connect(self._params_changed)

    def _apply_preset(self):
        key = self.preset.currentData()
        if self._loading or key == "custom":
            return
        p = PRESETS[key]
        self.undo.beginMacro(f"Vorlage {p['title']}")
        self._set("preset", key)
        self._set("width_mm", p["width_mm"])
        self._set("height_mm", p["height_mm"])
        self._set("font_size", p["font_size"])
        self.undo.endMacro()

    def _secondary_changed(self):
        unit = self.sec.currentData()
        src = self.processed.x_unit
        val = None
        if unit and src in U.UNITS and unit != src:
            val = {"from": src, "to": unit, "laser_nm": self.laser.value()}
        self._set(f"panels.{self._panel()}.secondary_x", val)

    def _params_changed(self):
        res = self.ds.fit_result
        names = [self.params.item(i).text() for i in range(self.params.count())
                 if self.params.item(i).checkState() == QtCore.Qt.Checked]
        val = {"lines": param_box_lines(res, names)} if names else None
        self._set("panels.0.param_box", val)

    def _trace_edited(self, row, col):
        if self._loading:
            return
        base = f"panels.{self._panel()}.traces.{row}"
        it = self.traces.item(row, col)
        if col == 0:
            self._set(base + ".visible", it.checkState() == QtCore.Qt.Checked)
        elif col == 2:
            self._set(base + ".label", it.text())

    def _pick_color(self, row, col):
        if col != 3:
            return
        base = f"panels.{self._panel()}.traces.{row}"
        cur = get_path(self.spec, base + ".color") or "#000000"
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(cur), self, "Farbe")
        if c.isValid():
            self._set(base + ".color", c.name())

    # ------------------------------------------------------------------ state <-> controls
    def spec_changed(self):
        self._load_controls()
        self._timer.start()

    def _load_controls(self):
        self._loading = True
        try:
            s = self.spec
            self.preset.setCurrentIndex(max(self.preset.findData(s.get("preset", "custom")), 0))
            self.width.setValue(s["width_mm"])
            self.height.setValue(s["height_mm"])
            self.font.setValue(s["font_size"])
            self.family.setCurrentText(s.get("font_family", "sans-serif"))
            self.lw.setValue(s.get("line_width", 1.0))
            self.panel_labels.setChecked(bool(s.get("panel_labels")))
            idx = self._panel()
            self.panel_sel.clear()
            for i, p in enumerate(s["panels"]):
                self.panel_sel.addItem(f"Panel {i + 1}" + (" (Haupt)" if i == 0 else " (Residuen)" if i == 1 else ""))
            self.panel_sel.setCurrentIndex(min(idx, len(s["panels"]) - 1))
        finally:
            self._loading = False
        self._load_panel()

    def _load_panel(self):
        if not self.spec["panels"]:
            return
        self._loading = True
        try:
            p = self.spec["panels"][self._panel()]
            self.xlabel.setText(p.get("xlabel", ""))
            self.ylabel.setText(p.get("ylabel", ""))
            self.legend.setCurrentIndex(max(self.legend.findData(p.get("legend")), 0))
            self.invert.setChecked(bool(p.get("invert_x")))
            self.ylog.setChecked(p.get("yscale") == "log")
            self.grid.setChecked(bool(p.get("grid")))
            sec = p.get("secondary_x")
            self.sec.setCurrentIndex(max(self.sec.findData(sec["to"] if sec else None), 0))
            self.traces.setRowCount(len(p["traces"]))
            for i, t in enumerate(p["traces"]):
                on = QtWidgets.QTableWidgetItem()
                on.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                on.setCheckState(QtCore.Qt.Checked if t.get("visible", True) else QtCore.Qt.Unchecked)
                self.traces.setItem(i, 0, on)
                name = QtWidgets.QTableWidgetItem(t["curve"])
                name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
                self.traces.setItem(i, 1, name)
                self.traces.setItem(i, 2, QtWidgets.QTableWidgetItem(t.get("label", "")))
                col = QtWidgets.QTableWidgetItem(t.get("color") or "auto")
                col.setFlags(col.flags() & ~QtCore.Qt.ItemIsEditable)
                if t.get("color"):
                    qc = QtGui.QColor(t["color"])
                    col.setBackground(qc)
                    col.setForeground(QtGui.QColor("white" if qc.lightness() < 128 else "black"))
                col.setToolTip("Doppelklick: Farbe wählen")
                self.traces.setItem(i, 3, col)
        finally:
            self._loading = False

    # ------------------------------------------------------------------ render / export
    def _render(self):
        try:
            fig = render_figure(self.spec, make_resolver(self.curves))
        except Exception as e:  # noqa: BLE001
            lab = QtWidgets.QLabel(f"Fehler beim Rendern: {e}")
            lab.setWordWrap(True)
            self.preview.setWidget(lab)
            return
        zoom = [1.0, 1.5, 2.0][self.zoom.currentIndex()]
        dpi = self.logicalDpiX() * zoom
        fig.set_dpi(dpi)
        canvas = self._canvas_cls(fig)
        w, h = fig.get_size_inches()
        canvas.setFixedSize(int(w * dpi), int(h * dpi))
        self.preview.setWidget(canvas)

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Abbildung exportieren", f"{self.ds.name}.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png);;PGF/LaTeX (*.pgf);;EPS (*.eps)")
        if not path:
            return
        try:
            save_figure(self.spec, make_resolver(self.curves), path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export", str(e))
            return
        QtWidgets.QMessageBox.information(self, "Export", f"Gespeichert: {path}")


# ============================================================================ comparison
class CompareDialog(QtWidgets.QDialog):
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modellvergleich")
        lay = QtWidgets.QVBoxLayout(self)
        try:
            c = compare(results)
        except ComparisonError as e:
            lay.addWidget(QtWidgets.QLabel(f"<span style='color:#b3261e'>✖ {e}</span>"))
            c = None
        if c is not None:
            t = QtWidgets.QTableWidget(len(c["rows"]), 6)
            t.setHorizontalHeaderLabels(["Modell", "K", f"Δ{c['rows'][0]['criterion']}", "Akaike-Gewicht",
                                         "ΔBIC", "χ²_ν / s"])
            for i, r in enumerate(c["rows"]):
                vals = [r["model"], str(r["k"]), f"{r['delta_aic']:.2f}", f"{r['akaike_weight']:.3f}",
                        f"{r['delta_bic']:.2f}",
                        f"{r['redchi']:.4g}" if r["redchi"] is not None else f"s = {r['s_res']:.4g}"]
                for j, v in enumerate(vals):
                    t.setItem(i, j, QtWidgets.QTableWidgetItem(v))
            t.resizeColumnsToContents()
            lay.addWidget(t)
            note = QtWidgets.QLabel(c["note"] + "\nForm: " + c["form"])
            note.setWordWrap(True)
            note.setObjectName("hint")
            lay.addWidget(note)
        self.results = results
        self._task = None
        box = QtWidgets.QGroupBox("Verschachtelter Test „eine Komponente mehr“ (Simulation)")
        form = QtWidgets.QFormLayout(box)
        self.null = QtWidgets.QComboBox()
        self.alt = QtWidgets.QComboBox()
        order = sorted(results, key=lambda k: results[k].stats.n_varys)
        for k in order:
            self.null.addItem(f"{k} (p = {results[k].stats.n_varys})", k)
            self.alt.addItem(f"{k} (p = {results[k].stats.n_varys})", k)
        self.alt.setCurrentIndex(self.alt.count() - 1)
        self.n_sim = QtWidgets.QSpinBox()
        self.n_sim.setRange(20, 5000)
        self.n_sim.setValue(200)
        run = QtWidgets.QPushButton("Simulation starten")
        run.clicked.connect(self._run_test)
        self.bar = QtWidgets.QProgressBar()
        self.out = QtWidgets.QLabel("Der nominelle LRT/F-Test ist am Parameterrand (Amplitude = 0) ungültig; "
                                    "die p-Wert-Verteilung wird deshalb durch Simulation unter dem Nullmodell "
                                    "bestimmt.")
        self.out.setWordWrap(True)
        form.addRow("Nullmodell", self.null)
        form.addRow("Alternative", self.alt)
        form.addRow("Replikate", self.n_sim)
        form.addRow(run, self.bar)
        form.addRow(self.out)
        lay.addWidget(box)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(680, 520)

    def _run_test(self):
        from ..fit import simulate_nested_test
        from .state import start_task
        if self._task is not None:
            return
        r0 = self.results[self.null.currentData()]
        r1 = self.results[self.alt.currentData()]
        self.bar.setRange(0, self.n_sim.value())

        def done(res):
            self._task = None
            self.out.setText("<br>".join(res["summary"]))

        def failed(msg):
            self._task = None
            self.out.setText(f"<span style='color:#b3261e'>✖ {msg.splitlines()[0]}</span>")
        self._task = start_task(simulate_nested_test, done, failed, self.bar.setValue, r0, r1, self.n_sim.value())


class BootstrapDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bootstrap")
        form = QtWidgets.QFormLayout(self)
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(50, 20000)
        self.n.setValue(500)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItem("Residuen (homoskedastisch)", "residual")
        self.kind.addItem("Wild/Rademacher (heteroskedastisch)", "wild")
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        form.addRow("Wiederholungen", self.n)
        form.addRow("Verfahren", self.kind)
        form.addRow("Seed", self.seed)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


# ============================================================================ series / global
class SeriesDialog(QtWidgets.QDialog):
    """Series fit (sequential, start values propagated) or global fit (shared parameters)."""

    def __init__(self, state, parent=None):
        import pyqtgraph as pg
        super().__init__(parent)
        self.setWindowTitle("Serie / globaler Fit")
        self.state = state
        self.series = None
        self.global_result = None
        self._task = None
        cur = state.current()
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Datensätze (Reihenfolge = Index):"))
        self.ds_list = QtWidgets.QListWidget()
        for ds in state.project.datasets:
            it = QtWidgets.QListWidgetItem(ds.name)
            it.setData(QtCore.Qt.UserRole, ds.id)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.ds_list.addItem(it)
        left.addWidget(self.ds_list, 1)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("Serie: nacheinander, Startwerte weitergeben", "series")
        self.mode.addItem("Serie: nacheinander, gleiche Startwerte", "series_fixed")
        self.mode.addItem("Global: gleichzeitig mit geteilten Parametern", "global")
        left.addWidget(self.mode)
        self.apply_template = QtWidgets.QCheckBox(f"Pipeline + Modell von „{cur.name if cur else ''}“ auf alle "
                                                  "anwenden")
        self.apply_template.setChecked(True)
        left.addWidget(self.apply_template)
        self.index_from_name = QtWidgets.QCheckBox("Index = erste Zahl im Namen (z. B. Temperatur)")
        left.addWidget(self.index_from_name)
        left.addWidget(QtWidgets.QLabel("Geteilte Parameter (nur global):"))
        self.shared = QtWidgets.QListWidget()
        self.shared.setMaximumHeight(130)
        if cur is not None:
            for n in cur.model.param_names():
                it = QtWidgets.QListWidgetItem(n)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Unchecked)
                self.shared.addItem(it)
        left.addWidget(self.shared)
        run = QtWidgets.QPushButton("Starten")
        run.setObjectName("primary")
        run.clicked.connect(self.run)
        left.addWidget(run)
        self.progress = QtWidgets.QProgressBar()
        left.addWidget(self.progress)
        lay.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Parameter:"))
        self.param = QtWidgets.QComboBox()
        self.param.currentIndexChanged.connect(self._plot)
        row.addWidget(self.param, 1)
        exp = QtWidgets.QPushButton("CSV exportieren…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        right.addLayout(row)
        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True, alpha=0.2)
        right.addWidget(self.pw, 2)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right.addWidget(self.table, 2)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setVisible(False)
        right.addWidget(self.report, 2)
        lay.addLayout(right, 2)
        self.resize(1150, 720)

    def _selected(self):
        out = []
        for i in range(self.ds_list.count()):
            it = self.ds_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                out.append(self.state.project.get(it.data(QtCore.Qt.UserRole)))
        return out

    def _index(self, datasets):
        import re as _re
        if self.index_from_name.isChecked():
            vals = []
            for i, ds in enumerate(datasets):
                m = _re.search(r"[-+]?\d+(?:[.,]\d+)?", ds.name)
                vals.append(float(m.group(0).replace(",", ".")) if m else float(i))
            return np.array(vals)
        return np.arange(len(datasets), dtype=float)

    def run(self):
        from ..fit import fit_global, fit_series
        from .state import start_task
        cur = self.state.current()
        datasets = self._selected()
        if cur is None or len(datasets) < 2:
            QtWidgets.QMessageBox.information(self, "Serie", "Mindestens zwei Datensätze wählen.")
            return
        if self.apply_template.isChecked():
            pipe = cur.pipeline.to_dict()
            model = cur.model.to_dict()
            opts = cur.fit_options.to_dict()
            self.state.undo.beginMacro("Vorlage auf Serie anwenden")
            for ds in datasets:
                if ds is cur:
                    continue
                self.state.edit("pipeline", lambda p, d=pipe: p.steps.__setitem__(slice(None), type(p).from_dict(d).steps),
                                "Pipeline übernehmen", ds=ds)
                self.state.edit("model", lambda m, d=model: m.components.__setitem__(
                    slice(None), type(m).from_dict(d).components), "Modell übernehmen", ds=ds)
                self.state.edit("options", lambda o, d=opts: o.__dict__.update(d), "Optionen übernehmen", ds=ds)
            self.state.undo.endMacro()
        spectra = [self.state.run(ds).final for ds in datasets]
        names = [ds.name for ds in datasets]
        index = self._index(datasets)
        mode = self.mode.currentData()
        self.progress.setRange(0, len(datasets) if mode != "global" else 0)
        if mode == "global":
            shared = [self.shared.item(i).text() for i in range(self.shared.count())
                      if self.shared.item(i).checkState() == QtCore.Qt.Checked]
            models = [ds.model.copy() for ds in datasets]
            task_fn = lambda progress=None: fit_global(spectra, models, shared, cur.fit_options)  # noqa: E731
        else:
            def task_fn(progress=None):
                return fit_series(spectra, cur.model, cur.fit_options, propagate=(mode == "series"),
                                  names=names, index=index, progress=progress)

        def done(res):
            self._task = None
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            if mode == "global":
                self.global_result = res
                self.series = None
                self.report.setVisible(True)
                self.report.setPlainText(res.report())
                self._fill_global(res, datasets)
            else:
                self.series = res
                self.report.setVisible(False)
                for ds, r in zip(datasets, res.results):
                    if r is not None:
                        ds.fit_result = r
                        ds.fit_record = r.to_dict(include_curves=False)
                        self.state.fitChanged.emit(ds.id)
                self._fill_series(res)

        def failed(msg):
            self._task = None
            self.progress.setRange(0, 1)
            QtWidgets.QMessageBox.warning(self, "Serie", msg.split("\n\n")[0])

        self._task = start_task(task_fn, done, failed, lambda i, n: self.progress.setValue(i))

    def _fill_series(self, res):
        rows = res.rows()
        names = res.parameter_names()
        self.param.blockSignals(True)
        self.param.clear()
        for n in names:
            self.param.addItem(n)
        self.param.blockSignals(False)
        cols = ["name", "index", "success", "redchi", "s_res"] + [n for n in names if "." not in n]
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(cols)
        from ..fit.result import fmt_value
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if c in names:
                    txt = fmt_value(r.get(c), r.get(c + "_stderr"))
                else:
                    v = r.get(c, "")
                    txt = f"{v:.5g}" if isinstance(v, float) else str(v)
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self._plot()

    def _fill_global(self, res, datasets):
        self.param.blockSignals(True)
        self.param.clear()
        bases = sorted({n.rsplit("_d", 1)[0] for n in res.params if "_d" in n})
        for b in bases:
            self.param.addItem(b)
        self.param.blockSignals(False)
        names = list(res.params)
        self.table.setColumnCount(2)
        self.table.setRowCount(len(names))
        self.table.setHorizontalHeaderLabels(["Parameter", "Wert ± SE"])
        from ..fit.result import fmt_value
        for i, n in enumerate(names):
            p = res.params[n]
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(n + ("  (geteilt)" if "_d" not in n else "")))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(fmt_value(p.value, p.stderr)))
        self.table.resizeColumnsToContents()
        self._global_n = len(datasets)
        self._plot()

    def _plot(self):
        import pyqtgraph as pg
        self.pw.clear()
        name = self.param.currentText()
        if not name:
            return
        if self.series is not None:
            v, e = self.series.parameter(name)
            x = self.series.index
        elif self.global_result is not None:
            r = self.global_result
            x = np.arange(self._global_n, dtype=float)
            v = np.array([r.params[f"{name}_d{i}"].value for i in range(self._global_n)])
            e = np.array([r.params[f"{name}_d{i}"].stderr or np.nan for i in range(self._global_n)])
        else:
            return
        ok = np.isfinite(v)
        err = pg.ErrorBarItem(x=x[ok], y=v[ok], height=2 * np.nan_to_num(e[ok]), beam=0.0,
                              pen=pg.mkPen("#3f90da"))
        self.pw.addItem(err)
        self.pw.plot(x[ok], v[ok], pen=pg.mkPen("#3f90da"), symbol="o", symbolSize=6, symbolBrush="#3f90da")
        self.pw.setLabel("left", name)
        self.pw.setLabel("bottom", "Index")

    def _export(self):
        if self.series is None and self.global_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", "serie.csv", "CSV (*.csv)")
        if not path:
            return
        if self.series is not None:
            self.series.to_csv(path)
        else:
            from ..export.results import write_parameters_csv
            write_parameters_csv(self.global_result, path)


# ============================================================================ combine datasets
class CombineDialog(QtWidgets.QDialog):
    """Compute a new dataset from several datasets (ratio, difference, parametric x/y)."""

    PRESETS = [("a / b", "x", "a/b"), ("a − b", "x", "a - b"), ("a + b", "x", "a + b"), ("a · b", "x", "a*b"),
               ("(a/b) als x, c als y", "a/b", "c"), ("b gegen a (parametrisch)", "a", "b")]

    def __init__(self, state, parent=None):
        import pyqtgraph as pg

        from ..combine import default_aliases
        super().__init__(parent)
        self.setWindowTitle("Daten verrechnen")
        self.state = state
        self.result = None
        self.report = None
        ds_list = state.project.datasets
        aliases = default_aliases(len(ds_list))
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget(len(ds_list), 5)
        self.table.setHorizontalHeaderLabels(["nutzen", "Kurzname", "Datensatz", "N / x-Bereich", "σ"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        current = state.current_id
        for i, ds in enumerate(ds_list):
            s = ds.processed
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            use.setCheckState(QtCore.Qt.Checked if len(ds_list) <= 3 or ds.id == current else QtCore.Qt.Unchecked)
            use.setData(QtCore.Qt.UserRole, ds.id)
            self.table.setItem(i, 0, use)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(aliases[i]))
            for j, txt in ((2, ds.name), (3, f"{s.n} / {s.x.min():.5g} … {s.x.max():.5g}" if s.n else "leer"),
                           (4, s.sigma_source.label)):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        left.addWidget(self.table, 2)
        form = QtWidgets.QFormLayout()
        self.stage = QtWidgets.QComboBox()
        self.stage.addItem("verarbeitet (nach Pipeline)", "processed")
        self.stage.addItem("Rohdaten", "raw")
        self.mode = QtWidgets.QComboBox()
        for k, v in (("auto", "automatisch"), ("exact", "identisches x-Raster"),
                     ("index", "punktweise (gleichzeitig aufgenommen)"),
                     ("interpolate", "Interpolation auf Referenz")):
            self.mode.addItem(v, k)
        self.ref = QtWidgets.QLineEdit("")
        self.ref.setPlaceholderText("Kurzname der Referenz (leer = erster)")
        presets = QtWidgets.QHBoxLayout()
        for title, xe, ye in self.PRESETS:
            b = QtWidgets.QToolButton()
            b.setText(title)
            b.clicked.connect(lambda _=False, xe=xe, ye=ye: (self.xexpr.setText(xe), self.yexpr.setText(ye)))
            presets.addWidget(b)
        self.xexpr = QtWidgets.QLineEdit("x")
        self.yexpr = QtWidgets.QLineEdit("a/b" if len(ds_list) > 1 else "a")
        for e in (self.xexpr, self.yexpr):
            e.setToolTip("Namen: Kurznamen (y-Werte), x (gemeinsames x), x_<Kurzname> (x des Datensatzes); "
                         "Funktionen wie in Formeln (exp, log, sqrt, …)")
        self.xlabel = QtWidgets.QLineEdit()
        self.ylabel = QtWidgets.QLineEdit()
        self.name = QtWidgets.QLineEdit()
        form.addRow("Datenstand", self.stage)
        form.addRow("Ausrichtung", self.mode)
        form.addRow("Referenz", self.ref)
        form.addRow("Vorlagen", presets)
        form.addRow("neue x =", self.xexpr)
        form.addRow("neue y =", self.yexpr)
        form.addRow("x-Titel", self.xlabel)
        form.addRow("y-Titel", self.ylabel)
        form.addRow("Name", self.name)
        left.addLayout(form)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        left.addWidget(self.info, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Als neuen Datensatz anlegen")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        self.ok = bb.button(QtWidgets.QDialogButtonBox.Ok)
        left.addWidget(bb)
        lay.addLayout(left, 3)
        self.pw = pg.PlotWidget()
        self.pw.showGrid(x=True, y=True, alpha=0.2)
        lay.addWidget(self.pw, 2)
        self._timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self.update_preview)
        for w in (self.xexpr, self.yexpr, self.ref):
            w.textChanged.connect(lambda _t: self._timer.start())
        for w in (self.stage, self.mode):
            w.currentIndexChanged.connect(lambda _i: self._timer.start())
        self.table.itemChanged.connect(lambda _it: self._timer.start())
        self.resize(1150, 640)
        self.update_preview()

    def selection(self) -> dict:
        out = {}
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() != QtCore.Qt.Checked:
                continue
            ds = self.state.project.get(self.table.item(i, 0).data(QtCore.Qt.UserRole))
            alias = self.table.item(i, 1).text().strip()
            out[alias] = ds.processed if self.stage.currentData() == "processed" else ds.raw
        return out

    def update_preview(self):
        from ..combine import combine, diagnose_alignment
        import pyqtgraph as pg
        self.pw.clear()
        self.result = None
        try:
            sel = self.selection()
            if not sel:
                raise ValueError("mindestens einen Datensatz auswählen")
            diag = diagnose_alignment(list(sel.values()))
            out, rep = combine(sel, self.xexpr.text(), self.yexpr.text(), self.mode.currentData(),
                               self.ref.text().strip() or None, self.xlabel.text(), self.ylabel.text(),
                               self.name.text())
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.info.setText(f"<span style='color:#b3261e'>✖ {e}</span>")
            self.ok.setEnabled(False)
            return
        self.result, self.report = out, rep
        lines = [f"<b>Ausrichtung (automatisch erkannt):</b> {diag['text']}" if self.mode.currentData() == "auto"
                 else f"<b>Ausrichtung:</b> {rep['alignment']}",
                 f"<b>Ergebnis:</b> {rep['n_output']} von {rep['n_input']} Punkten · σ: "
                 f"{'fortgepflanzt (' + out.sigma_source.label + ')' if out.sigma is not None else 'keins'}"
                 f"{' · σ_x vorhanden' if 'sigma_x' in out.aux else ''}"]
        lines += [f"<span style='color:#b36b00'>⚠ {w}</span>" for w in rep["warnings"]]
        self.info.setText("<br>".join(lines))
        self.ok.setEnabled(out.n > 1)
        scatter = out.meta.get("plot_style") == "scatter"
        if out.sigma is not None and out.n < 3000:
            self.pw.addItem(pg.ErrorBarItem(x=out.x, y=out.y, height=2 * out.sigma, pen=pg.mkPen("#9db6e0")))
        self.pw.plot(out.x, out.y, pen=None if scatter else pg.mkPen("#2b2b2b"),
                     symbol="o" if scatter else None, symbolSize=4, symbolBrush="#2b2b2b", symbolPen=None)
        self.pw.setLabel("bottom", out.x_label)
        self.pw.setLabel("left", out.y_label)

    def _accept(self):
        import hashlib

        from ..project import Dataset
        if self.result is None:
            return
        out = self.result
        name = out.meta.get("name") or "verrechnet"
        cols = ["x", "y"] + (["sigma"] if out.sigma is not None else [])
        arrays = [out.x, out.y] + ([out.sigma] if out.sigma is not None else [])
        text = ",".join(cols) + "\n" + "\n".join(",".join(repr(float(v)) for v in row) for row in zip(*arrays)) + "\n"
        data = text.encode("utf-8")
        fname = re_safe(name) + ".csv"
        meta = dict(out.meta)
        meta["source"] = {"filename": fname, "path": fname, "sha256": hashlib.sha256(data).hexdigest(),
                          "reader": "text", "options": {"x_col": 0, "y_col": 1,
                                                        "sigma_col": 2 if out.sigma is not None else None,
                                                        "delimiter": ",", "decimal": "."}}
        ds = Dataset(name=name, raw=out.replace(meta=meta), raw_bytes=data, raw_ext=".csv")
        ds.notes = "abgeleitet: x = {x_expr}, y = {y_expr}".format(**out.meta["derived"])
        self.state.add_dataset(ds)
        self.accept()


def re_safe(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "verrechnet"
