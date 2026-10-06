"""Model builder, parameter table and fit options."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from ..fit import METHODS
from ..fit.result import fmt_value
from ..models import COMPONENT_TYPES, TEMPLATES, Formula, FormulaError, add_template
from .param_form import fmt_num, parse_num
from .theme import SEVERITY_COLOR

PEAK_KINDS = [k for k, t in COMPONENT_TYPES.items() if t.category == "peak"]
WEIGHTING = [("auto", "automatisch (σ falls vorhanden)"), ("sigma", "σ der Daten"),
             ("none", "keine (σ unbekannt)"), ("poisson_model", "Poisson: σ² = Modell (Zählraten)")]
COVARIANCE = [("auto", "automatisch"), ("absolute", "absolut (σ bekannt)"), ("scaled", "skaliert mit √χ²_ν")]


class FormulaDialog(QtWidgets.QDialog):
    """Enter a user formula; parameters are detected live."""

    def __init__(self, expression="", independent="x", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eigene Funktion")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Ausdruck in x (NumPy-Syntax, ^ = Potenz). Alle übrigen Namen sind Fitparameter.\n"
            "Funktionen: exp, log/ln, sqrt, sin, …, erf, gamma, gauss(x,A,c,w), lorentz, voigt(x,A,c,wG,wL), "
            "pvoigt, pearson7, emg"))
        self.edit = QtWidgets.QLineEdit(expression)
        self.edit.setPlaceholderText("z. B.  y0 + A*exp(-x/tau)")
        lay.addWidget(self.edit)
        form = QtWidgets.QFormLayout()
        self.indep = QtWidgets.QLineEdit(independent)
        self.indep.setToolTip("Unabhängige Variablen, durch Komma getrennt; weitere Variablen müssen beim Import "
                              "als zusätzliche Spalten gewählt worden sein")
        form.addRow("Unabhängige Variablen", self.indep)
        self.prefix = QtWidgets.QLineEdit("")
        self.prefix.setToolTip("optionaler Präfix für Parameternamen (nötig, wenn Namen kollidieren)")
        form.addRow("Präfix", self.prefix)
        lay.addLayout(form)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay.addWidget(self.bb)
        self.edit.textChanged.connect(self._check)
        self.indep.textChanged.connect(self._check)
        self._check()
        self.resize(560, 220)

    def independent(self):
        return [v.strip() for v in self.indep.text().split(",") if v.strip()] or ["x"]

    def _check(self):
        try:
            f = Formula(self.edit.text(), self.independent())
            self.info.setText(f"<span style='color:#2e7d32'>✓ Parameter: {', '.join(f.parameters)}</span>")
            self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        except FormulaError as e:
            self.info.setText(f"<span style='color:{SEVERITY_COLOR['error']}'>✖ {e}</span>")
            self.bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)


class ModelPanel(QtWidgets.QWidget):
    fitRequested = QtCore.Signal()
    autoPeaksRequested = QtCore.Signal()
    componentSelected = QtCore.Signal(str)

    COLS = ["Parameter", "Start", "Ergebnis ± SE", "frei", "min", "max", "Ausdruck", "Hinweis"]
    C_NAME, C_START, C_RES, C_VARY, C_MIN, C_MAX, C_EXPR, C_HINT = range(8)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._updating = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        bar = QtWidgets.QHBoxLayout()
        self.peak_kind = QtWidgets.QComboBox()
        for k in PEAK_KINDS:
            self.peak_kind.addItem(COMPONENT_TYPES[k].title, k)
        self.peak_kind.setToolTip("Profiltyp für 'Peak'-Klicks im Plot und automatische Peaksuche")
        bar.addWidget(QtWidgets.QLabel("Peaktyp:"))
        bar.addWidget(self.peak_kind)
        add = QtWidgets.QToolButton()
        add.setText("+ Komponente")
        add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        add.setMenu(self._menu())
        bar.addWidget(add)
        auto = QtWidgets.QToolButton()
        auto.setText("Peaks finden")
        auto.setToolTip("Startwerte aus lokalen Maxima (Prominenz > 5·Rauschen)")
        auto.clicked.connect(self.autoPeaksRequested.emit)
        bar.addWidget(auto)
        rem = QtWidgets.QToolButton()
        rem.setText("Entfernen")
        rem.clicked.connect(self._remove)
        bar.addWidget(rem)
        bar.addStretch()
        lay.addLayout(bar)

        self.components = QtWidgets.QListWidget()
        self.components.setMaximumHeight(90)
        self.components.currentItemChanged.connect(
            lambda cur, _p: cur is not None and self.componentSelected.emit(cur.data(QtCore.Qt.UserRole)))
        lay.addWidget(self.components)

        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        for i, w in enumerate([100, 72, 130, 34, 58, 58, 80]):
            self.table.setColumnWidth(i, w)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._cell_changed)
        lay.addWidget(self.table, 1)

        opt = QtWidgets.QGroupBox("Fit")
        form = QtWidgets.QFormLayout(opt)
        self.method = QtWidgets.QComboBox()
        for k, v in METHODS.items():
            self.method.addItem(v, k)
        self.weighting = QtWidgets.QComboBox()
        for k, v in WEIGHTING:
            self.weighting.addItem(v, k)
        self.covariance = QtWidgets.QComboBox()
        for k, v in COVARIANCE:
            self.covariance.addItem(v, k)
        self.covariance.setToolTip("Bei bekanntem σ ist 'absolut' korrekt; Skalieren verdeckt Fehlanpassung.")
        self.range_label = QtWidgets.QLabel()
        rng_row = QtWidgets.QHBoxLayout()
        rng_row.addWidget(self.range_label, 1)
        clr = QtWidgets.QToolButton()
        clr.setText("ganzer Bereich")
        clr.clicked.connect(lambda: self._set_option("x_range", None, "Fitbereich zurücksetzen"))
        rng_row.addWidget(clr)
        form.addRow("Methode", self.method)
        form.addRow("Gewichtung", self.weighting)
        form.addRow("Kovarianz", self.covariance)
        form.addRow("Fitbereich", rng_row)
        self.method.currentIndexChanged.connect(lambda: self._set_option("method", self.method.currentData()))
        self.weighting.currentIndexChanged.connect(
            lambda: self._set_option("weighting", self.weighting.currentData()))
        self.covariance.currentIndexChanged.connect(
            lambda: self._set_option("covariance", self.covariance.currentData()))
        btns = QtWidgets.QHBoxLayout()
        self.fit_btn = QtWidgets.QPushButton("Fit ausführen")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.fit_btn.setToolTip("Fit starten (Strg+R)")
        self.fit_btn.clicked.connect(self.fitRequested.emit)
        self.adopt_btn = QtWidgets.QPushButton("Ergebnis → Startwerte")
        self.adopt_btn.clicked.connect(self._adopt)
        self.live = QtWidgets.QCheckBox("Live-Fit")
        self.live.setToolTip("Nach jeder Änderung (z. B. Peak ziehen) automatisch neu fitten")
        btns.addWidget(self.fit_btn)
        btns.addWidget(self.adopt_btn)
        btns.addWidget(self.live)
        form.addRow(btns)
        lay.addWidget(opt)

        state.currentChanged.connect(self.refresh)
        state.modelChanged.connect(lambda _id: self.refresh())
        state.fitChanged.connect(lambda _id: self.refresh())
        state.fitOptionsChanged.connect(lambda _id: self.refresh())

    # ------------------------------------------------------------------ menu
    def _menu(self):
        m = QtWidgets.QMenu(self)
        pk = m.addMenu("Peak")
        for k in PEAK_KINDS:
            a = pk.addAction(COMPONENT_TYPES[k].title)
            a.setToolTip(COMPONENT_TYPES[k].formula_text)
            a.triggered.connect(lambda _=False, k=k: self.add_component(k))
        bg = m.addMenu("Untergrund (linear, mitfittbar)")
        bg.addAction("Konstante").triggered.connect(lambda: self.add_component("constant"))
        bg.addAction("Gerade").triggered.connect(lambda: self.add_component("linear"))
        for order in (2, 3, 4, 5):
            bg.addAction(f"Polynom Grad {order}").triggered.connect(
                lambda _=False, o=order: self.add_component("polynomial", {"order": o}))
        cats = {}
        for t in TEMPLATES:
            cats.setdefault(t.category, []).append(t)
        fm = m.addMenu("Klassische Funktion")
        for cat, items in cats.items():
            sub = fm.addMenu(cat)
            for t in items:
                a = sub.addAction(f"{t.name}:  {t.expression}")
                a.setToolTip(t.description)
                a.triggered.connect(lambda _=False, n=t.name: self.add_template(n))
        m.addSeparator()
        m.addAction("Eigene Formel…").triggered.connect(self.add_formula)
        return m

    def _ds(self):
        return self.state.current()

    def _x_center(self):
        ds = self._ds()
        run = self.state.run(ds)
        s = run.final
        return float(0.5 * (s.x.min() + s.x.max())) if s.n else 0.0

    def add_component(self, kind, options=None):
        options = dict(options or {})
        if kind in ("polynomial", "linear"):
            options.setdefault("x0", self._x_center())
        if kind in COMPONENT_TYPES and COMPONENT_TYPES[kind].category == "peak":
            ds = self._ds()
            s = self.state.run(ds).final
            xc = self._x_center()
            span = float(s.x.max() - s.x.min()) if s.n else 1.0
            from ..models.library import add_peak
            h = float(s.y.max() - s.y.min()) if s.n else 1.0
            self.state.edit("model", lambda m: add_peak(m, kind, xc, h, 0.05 * span), f"Peak hinzufügen ({kind})")
            return
        self.state.edit("model", lambda m: m.add(kind, options=options), f"Komponente hinzufügen ({kind})")

    def add_template(self, name):
        def mut(m):
            prefix = ""
            names = set(m.param_names())
            from ..models.library import TEMPLATE_BY_NAME
            t = TEMPLATE_BY_NAME[name]
            if names & set(Formula(t.expression).parameters):
                i = 2
                while any(n.startswith(f"f{i}_") for n in names):
                    i += 1
                prefix = f"f{i}_"
            add_template(m, name, prefix=prefix)
        try:
            self.state.edit("model", mut, f"Funktion: {name}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Modell", str(e))

    def add_formula(self):
        dlg = FormulaDialog(parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        expr, indep, prefix = dlg.edit.text(), dlg.independent(), dlg.prefix.text().strip()
        try:
            self.state.edit("model", lambda m: m.add("formula", prefix=prefix,
                                                      options={"expression": expr, "independent": indep}),
                            "Formel hinzufügen")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Modell", str(e))

    def _remove(self):
        it = self.components.currentItem()
        if it is None:
            return
        idx = self.components.row(it)
        self.state.edit("model", lambda m: m.remove(idx), "Komponente entfernen")

    def _adopt(self):
        ds = self._ds()
        if ds is None or ds.fit_result is None:
            return
        vals = ds.fit_result.values
        self.state.edit("model", lambda m: m.apply_values(vals), "Ergebnis als Startwerte")

    def _set_option(self, name, value, text=None):
        if self._updating or self._ds() is None:
            return
        self.state.edit("options", lambda o: setattr(o, name, value), text or f"Fit-Option: {name}")

    # ------------------------------------------------------------------ table
    def refresh(self):
        ds = self._ds()
        self._updating = True
        try:
            self.components.clear()
            self.table.setRowCount(0)
            if ds is None:
                return
            for c in ds.model.components:
                text = c.display_name
                if c.kind == "formula":
                    text += f":  {c.options.get('expression')}"
                it = QtWidgets.QListWidgetItem(text)
                it.setData(QtCore.Qt.UserRole, c.prefix)
                it.setToolTip(c.type.formula_text if c.kind != "formula" else c.options.get("expression", ""))
                self.components.addItem(it)
            res = ds.fit_result if self.state.fit_current(ds) else None
            for ci, c in enumerate(ds.model.components):
                for local in c.local_names():
                    self._add_row(ci, c, local, res)
            o = ds.fit_options
            self.method.setCurrentIndex(max(self.method.findData(o.method), 0))
            self.weighting.setCurrentIndex(max(self.weighting.findData(o.weighting), 0))
            self.covariance.setCurrentIndex(max(self.covariance.findData(o.covariance), 0))
            self.range_label.setText("ganzer Bereich (nur Maske)" if not o.x_range else
                                     f"{fmt_num(o.x_range[0])} … {fmt_num(o.x_range[1])}")
            self.adopt_btn.setEnabled(res is not None)
        finally:
            self._updating = False

    def _add_row(self, ci, comp, local, res):
        r = self.table.rowCount()
        self.table.insertRow(r)
        st = comp.settings[local]
        name = comp.full_name(local)

        def item(text, editable=True, tip=None):
            it = QtWidgets.QTableWidgetItem(text)
            if not editable:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            if tip:
                it.setToolTip(tip)
            it.setData(QtCore.Qt.UserRole, (ci, local))
            return it

        pdef = next((p for p in comp.type.params if p.name == local), None)
        self.table.setItem(r, self.C_NAME, item(name, False, pdef.label if pdef else None))
        self.table.setItem(r, self.C_START, item(fmt_num(st.value)))
        self.table.setItem(r, self.C_MIN, item("" if math.isinf(st.min) else fmt_num(st.min)))
        self.table.setItem(r, self.C_MAX, item("" if math.isinf(st.max) else fmt_num(st.max)))
        chk = item("", True)
        chk.setFlags((chk.flags() | QtCore.Qt.ItemIsUserCheckable) & ~QtCore.Qt.ItemIsEditable)
        chk.setCheckState(QtCore.Qt.Checked if st.vary and not st.expr else QtCore.Qt.Unchecked)
        self.table.setItem(r, self.C_VARY, chk)
        self.table.setItem(r, self.C_EXPR, item(st.expr or "", True, "Constraint, z. B. p1_fwhm (gleiche Breite) "
                                                                     "oder p1_center + 12.5"))
        txt, hint, color = "", "", None
        if res is not None and name in res.params:
            p = res.params[name]
            txt = fmt_value(p.value, p.stderr)
            if p.at_bound:
                hint, color = f"am Bound ({p.at_bound})", SEVERITY_COLOR["warning"]
            elif p.near_bound:
                hint, color = "nahe Bound", SEVERITY_COLOR["info"]
            elif p.stderr is not None and p.value != 0 and abs(p.stderr / p.value) > 0.5:
                hint, color = "schlecht bestimmt", SEVERITY_COLOR["warning"]
        rit = item(txt, False)
        if color:
            rit.setForeground(QtGui.QColor(color))
        self.table.setItem(r, self.C_RES, rit)
        hit = item(hint, False)
        if color:
            hit.setForeground(QtGui.QColor(color))
        self.table.setItem(r, self.C_HINT, hit)

    def _cell_changed(self, it):
        if self._updating:
            return
        data = it.data(QtCore.Qt.UserRole)
        if not data:
            return
        ci, local = data
        col = it.column()

        def mut(m):
            st = m.components[ci].settings[local]
            if col == self.C_START:
                st.value = parse_num(it.text())
            elif col == self.C_MIN:
                v = parse_num(it.text())
                st.min = -math.inf if v is None else v
            elif col == self.C_MAX:
                v = parse_num(it.text())
                st.max = math.inf if v is None else v
            elif col == self.C_VARY:
                st.vary = it.checkState() == QtCore.Qt.Checked
            elif col == self.C_EXPR:
                st.expr = it.text().strip() or None
            else:
                return
            if st.min > st.max:
                raise ValueError("min > max")
            if not st.expr and st.value is not None and not (st.min <= st.value <= st.max):
                st.value = min(max(st.value, st.min), st.max)
        try:
            self.state.edit("model", mut, f"Parameter {local}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Parameter", f"Ungültige Eingabe: {e}")
            self.refresh()
