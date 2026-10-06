"""Automatic parameter forms for processing operations (from their ParamSpecs)."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from ..ops.registry import OpSpec


def fmt_num(v) -> str:
    if v is None:
        return ""
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e6:
        return f"{v:.6g}"
    return f"{v:.4e}"


def parse_num(text: str):
    t = text.strip().replace(",", ".")
    if t == "" or t.lower() in ("none", "-"):
        return None
    return float(t)


class SciEdit(QtWidgets.QLineEdit):
    """Number entry accepting scientific notation and decimal commas."""
    valueEdited = QtCore.Signal(object)

    def __init__(self, value=None, optional=False, parent=None):
        super().__init__(parent)
        self.optional = optional
        self.setValue(value)
        self.editingFinished.connect(self._commit)
        self.setMinimumWidth(80)

    def setValue(self, v):
        self._value = v
        self.setText(fmt_num(v))
        self.setStyleSheet("")

    def value(self):
        return self._value

    def _commit(self):
        try:
            v = parse_num(self.text())
        except ValueError:
            self.setStyleSheet("background:#ffe0e0;")
            return
        if v is None and not self.optional:
            self.setValue(self._value)
            return
        self.setStyleSheet("")
        if v != self._value:
            self._value = v
            self.valueEdited.emit(v)


class LogSlider(QtWidgets.QWidget):
    """Slider on a log10 scale plus a numeric field (e.g. Whittaker lambda)."""
    valueChanged = QtCore.Signal(float, bool)     # value, final

    def __init__(self, value, lo=-2.0, hi=12.0, parent=None):
        super().__init__(parent)
        self.lo, self.hi = lo, hi
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) * 20)))
        self.edit = SciEdit(value)
        self.edit.setMaximumWidth(90)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.edit)
        self.setValue(value)
        self.slider.valueChanged.connect(self._slid)
        self.slider.sliderReleased.connect(lambda: self.valueChanged.emit(self.edit.value(), True))
        self.edit.valueEdited.connect(self._edited)

    def setValue(self, v):
        self.edit.setValue(v)
        if v and v > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int(round((math.log10(v) - self.lo) * 20)))
            self.slider.blockSignals(False)

    def _slid(self, pos):
        v = 10 ** (self.lo + pos / 20.0)
        v = float(f"{v:.3g}")
        self.edit.setValue(v)
        self.valueChanged.emit(v, not self.slider.isSliderDown())

    def _edited(self, v):
        self.setValue(v)
        self.valueChanged.emit(v, True)


class ParamForm(QtWidgets.QWidget):
    changed = QtCore.Signal(str, object, bool)       # name, value, final
    rangeTarget = QtCore.Signal(str)                 # 'ranges' parameter edited by the plot tool
    editRanges = QtCore.Signal(str)                  # open the table editor for a ranges parameter

    def __init__(self, spec: OpSpec, params: dict, hint_fn=None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.params = dict(params)
        self.hint_fn = hint_fn
        self.widgets = {}
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self._range_group = QtWidgets.QButtonGroup(self)
        first_ranges = True
        for p in spec.params:
            w = self._make(p, params.get(p.name, p.default), first_ranges)
            if p.kind == "ranges":
                first_ranges = False
            if p.help:
                w.setToolTip(p.help)
            form.addRow(p.label or p.name, w)
            self.widgets[p.name] = w
        self.hint = QtWidgets.QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        form.addRow(self.hint)
        if spec.description:
            desc = QtWidgets.QLabel(spec.description)
            desc.setObjectName("hint")
            desc.setWordWrap(True)
            form.addRow(desc)
        self._update_hint()

    # ------------------------------------------------------------------ widgets
    def _make(self, p, value, first_ranges):
        name = p.name
        if p.kind == "float":
            w = SciEdit(value, optional=p.optional)
            w.valueEdited.connect(lambda v, n=name: self._emit(n, v, True))
            return w
        if p.kind == "log_float":
            w = LogSlider(value)
            w.valueChanged.connect(lambda v, final, n=name: self._emit(n, v, final))
            return w
        if p.kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(p.min) if p.min is not None else -10**9, int(p.max) if p.max is not None else 10**9)
            w.setSpecialValueText("") if p.optional else None
            if value is not None:
                w.setValue(int(value))
            w.valueChanged.connect(lambda v, n=name: self._emit(n, int(v), True))
            return w
        if p.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, n=name: self._emit(n, bool(v), True))
            return w
        if p.kind == "choice":
            w = QtWidgets.QComboBox()
            for c in p.choices:
                w.addItem("—" if c in ("", None) else str(c), c)
            if p.optional and None not in p.choices:
                w.insertItem(0, "(automatisch)", None)
            idx = w.findData(value)
            w.setCurrentIndex(max(idx, 0))
            w.currentIndexChanged.connect(lambda i, n=name, cb=w: self._emit(n, cb.itemData(i), True))
            return w
        if p.kind == "str":
            w = QtWidgets.QLineEdit(value or "")
            w.editingFinished.connect(lambda n=name, e=w: self._emit(n, e.text(), True))
            return w
        if p.kind == "range":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            a = SciEdit(value[0] if value else None, optional=True)
            b = SciEdit(value[1] if value else None, optional=True)
            lay.addWidget(a)
            lay.addWidget(QtWidgets.QLabel("…"))
            lay.addWidget(b)
            a.valueEdited.connect(lambda _v, n=name, a=a, b=b: self._emit(n, [a.value(), b.value()], True))
            b.valueEdited.connect(lambda _v, n=name, a=a, b=b: self._emit(n, [a.value(), b.value()], True))
            w.edits = (a, b)
            return w
        if p.kind == "ranges":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lab = QtWidgets.QLabel()
            radio = QtWidgets.QRadioButton("Plot")
            radio.setToolTip("Das Werkzeug 'Bereich' im Plot fügt Bereiche zu diesem Parameter hinzu")
            self._range_group.addButton(radio)
            radio.toggled.connect(lambda on, n=name: on and self.rangeTarget.emit(n))
            btn = QtWidgets.QToolButton()
            btn.setText("…")
            btn.setToolTip("Bereiche als Tabelle bearbeiten")
            btn.clicked.connect(lambda _=False, n=name: self.editRanges.emit(n))
            clr = QtWidgets.QToolButton()
            clr.setText("✕")
            clr.setToolTip("alle Bereiche löschen")
            clr.clicked.connect(lambda _=False, n=name: self._emit(n, [], True))
            lay.addWidget(lab, 1)
            lay.addWidget(radio)
            lay.addWidget(btn)
            lay.addWidget(clr)
            w.label = lab
            w.radio = radio
            lab.setText(f"{len(value or [])}×")
            if first_ranges:
                radio.setChecked(True)
            return w
        if p.kind == "anchors":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lab = QtWidgets.QLabel()
            clr = QtWidgets.QToolButton()
            clr.setText("✕")
            clr.setToolTip("alle Anker löschen")
            clr.clicked.connect(lambda _=False, n=name: self._emit(n, [], True))
            lay.addWidget(lab, 1)
            lay.addWidget(clr)
            w.label = lab
            lab.setText(f"{len(value or [])} – Werkzeug 'Anker'")
            return w
        raise ValueError(p.kind)

    def _emit(self, name, value, final):
        self.params[name] = value
        self._update_hint()
        self.changed.emit(name, value, final)

    def _update_hint(self):
        if self.hint_fn is None:
            self.hint.hide()
            return
        text = self.hint_fn(self.params)
        self.hint.setText(text or "")
        self.hint.setVisible(bool(text))

    def set_params(self, params: dict):
        """Refresh the widgets from outside (undo/redo, plot interaction) without emitting."""
        self.params = dict(params)
        for p in self.spec.params:
            w = self.widgets[p.name]
            v = params.get(p.name, p.default)
            w.blockSignals(True)
            try:
                if p.kind == "float":
                    w.setValue(v)
                elif p.kind == "log_float":
                    w.setValue(v)
                elif p.kind == "int" and v is not None:
                    w.setValue(int(v))
                elif p.kind == "bool":
                    w.setChecked(bool(v))
                elif p.kind == "choice":
                    w.setCurrentIndex(max(w.findData(v), 0))
                elif p.kind == "str":
                    w.setText(v or "")
                elif p.kind == "range":
                    w.edits[0].setValue(v[0] if v else None)
                    w.edits[1].setValue(v[1] if v else None)
                elif p.kind == "ranges":
                    w.label.setText(f"{len(v or [])}×")
                elif p.kind == "anchors":
                    w.label.setText(f"{len(v or [])} – Werkzeug 'Anker'")
            finally:
                w.blockSignals(False)
        self._update_hint()

    def active_ranges_param(self):
        for p in self.spec.params:
            if p.kind == "ranges" and self.widgets[p.name].radio.isChecked():
                return p.name
        return None


class RangesDialog(QtWidgets.QDialog):
    """Table editor for a list of x ranges."""

    def __init__(self, ranges, title="Bereiche", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["von", "bis"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.table)
        for a, b in ranges:
            self._add(a, b)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("+ Zeile")
        add.clicked.connect(lambda: self._add(0.0, 1.0))
        rem = QtWidgets.QPushButton("− Zeile")
        rem.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch()
        lay.addLayout(row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(320, 300)

    def _add(self, a, b):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(fmt_num(a)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(fmt_num(b)))

    def ranges(self):
        out = []
        for r in range(self.table.rowCount()):
            try:
                a = parse_num(self.table.item(r, 0).text())
                b = parse_num(self.table.item(r, 1).text())
            except (ValueError, AttributeError):
                continue
            if a is not None and b is not None:
                out.append([min(a, b), max(a, b)])
        return out


def italic_hint(text):
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    f = lab.font()
    f.setItalic(True)
    lab.setFont(f)
    return lab


_ = QtGui  # keep import for type users
