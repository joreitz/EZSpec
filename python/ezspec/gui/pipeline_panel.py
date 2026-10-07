"""Dataset list and processing pipeline editor."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..ops import list_ops
from ..ops.baseline import whittaker_cutoff_points
from .param_form import ParamForm, RangesDialog
from .theme import SEVERITY_COLOR

CATEGORY_ORDER = ["Range", "Correction", "Baseline", "Smoothing", "Uncertainty", "Scaling", "Units"]


class DatasetsPanel(QtWidgets.QWidget):
    importRequested = QtCore.Signal()
    exampleRequested = QtCore.Signal()
    combineRequested = QtCore.Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = QtWidgets.QListWidget()
        self.list.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.list.currentRowChanged.connect(self._selected)
        self.list.itemChanged.connect(self._renamed)
        lay.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Import…", self.importRequested.emit, "Load text/CSV or JCAMP-DX"),
                                ("Example", self.exampleRequested.emit, "Load a synthetic Raman spectrum"),
                                ("Combine…", self.combineRequested.emit,
                                 "Ratio/difference/formulas from several datasets (Ctrl+K)"),
                                ("Duplicate", self._duplicate, "Copy with the same pipeline and model"),
                                ("Remove", self._remove, "Remove dataset (can be undone)")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        state.projectChanged.connect(self.refresh)
        state.currentChanged.connect(self.refresh)

    def refresh(self):
        self.list.blockSignals(True)
        self.list.clear()
        for ds in self.state.project.datasets:
            it = QtWidgets.QListWidgetItem(ds.name)
            it.setData(QtCore.Qt.UserRole, ds.id)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            src = ds.source
            it.setToolTip(f"{src.get('filename', '')}\nN = {ds.raw.n}, σ: {ds.raw.sigma_source.label}\n"
                          f"SHA-256 {src.get('sha256', '')[:16]}…")
            self.list.addItem(it)
            if ds.id == self.state.current_id:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)

    def _selected(self, row):
        it = self.list.item(row)
        if it is not None:
            self.state.set_current(it.data(QtCore.Qt.UserRole))

    def _renamed(self, it):
        ds = self.state.project.get(it.data(QtCore.Qt.UserRole))
        self.state.rename_dataset(ds, it.text().strip())

    def _remove(self):
        ds = self.state.current()
        if ds is not None:
            self.state.remove_dataset(ds)

    def _duplicate(self):
        from ..project import Dataset
        ds = self.state.current()
        if ds is None:
            return
        copy = Dataset(name=ds.name + " (copy)", raw=ds.raw, pipeline=ds.pipeline.copy(), model=ds.model.copy(),
                       fit_options=type(ds.fit_options).from_dict(ds.fit_options.to_dict()),
                       raw_bytes=ds.raw_bytes, raw_ext=ds.raw_ext)
        self.state.add_dataset(copy)


class PipelinePanel(QtWidgets.QWidget):
    rangeTargetChanged = QtCore.Signal(str)
    calibrationRequested = QtCore.Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.form = None
        self._form_step = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QToolButton()
        self.add_btn.setText("+ Step")
        self.add_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.add_btn.setMenu(self._build_menu())
        bar.addWidget(self.add_btn)
        for text, slot, tip in (("↑", lambda: self._move(-1), "Move step up"),
                                ("↓", lambda: self._move(+1), "Move step down"),
                                ("Remove", self._remove, "Remove step")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(170)
        self.list.currentItemChanged.connect(self._selected)
        self.list.itemChanged.connect(self._toggled)
        lay.addWidget(self.list)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.scroll, 1)
        state.currentChanged.connect(self.refresh)
        state.pipelineChanged.connect(lambda _id: self.refresh())
        state.stepSelected.connect(self._sync_selection)

    def _sync_selection(self, sid):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(QtCore.Qt.UserRole) == sid:
                if self.list.currentItem() is not it:
                    self.list.blockSignals(True)
                    self.list.setCurrentItem(it)
                    self.list.blockSignals(False)
                    ds = self.state.current()
                    self._show_step(ds, self.state.run(ds) if ds else None)
                return

    def _build_menu(self):
        menu = QtWidgets.QMenu(self)
        cats = {}
        for spec in list_ops():
            cats.setdefault(spec.category, []).append(spec)
        for cat in CATEGORY_ORDER + sorted(set(cats) - set(CATEGORY_ORDER)):
            if cat not in cats:
                continue
            sub = menu.addMenu(cat)
            for spec in cats[cat]:
                if spec.name == "calibrate_x":     # needs a calibration from a fit: dedicated dialog
                    a = sub.addAction(spec.title + "…")
                    a.setToolTip(spec.description)
                    a.triggered.connect(self.calibrationRequested.emit)
                    continue
                a = sub.addAction(spec.title)
                a.setToolTip(spec.description)
                a.triggered.connect(lambda _=False, n=spec.name: self.add_step(n))
        return menu

    # ------------------------------------------------------------------ list
    def refresh(self):
        ds = self.state.current()
        run = self.state.run(ds) if ds else None
        self.list.blockSignals(True)
        self.list.clear()
        raw = QtWidgets.QListWidgetItem("● Raw data / result")
        raw.setData(QtCore.Qt.UserRole, None)
        raw.setToolTip("No step selected: processed data with fit")
        self.list.addItem(raw)
        current_item = raw
        if ds is not None:
            for i, step in enumerate(ds.pipeline):
                res = run.results[i] if run else None
                mark = ""
                if res is not None and res.error:
                    mark = "  ✖"
                elif res is not None and res.warnings:
                    mark = "  ⚠"
                it = QtWidgets.QListWidgetItem(f"{i + 1}. {step.title}{mark}")
                it.setData(QtCore.Qt.UserRole, step.id)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if step.enabled else QtCore.Qt.Unchecked)
                if res is not None and res.error:
                    it.setForeground(QtGui.QColor(SEVERITY_COLOR["error"]))
                self.list.addItem(it)
                if step.id == self.state.selected_step:
                    current_item = it
        self.list.setCurrentItem(current_item)
        self.list.blockSignals(False)
        self._show_step(ds, run)

    def _selected(self, cur, _prev):
        if cur is None:
            return
        self.state.select_step(cur.data(QtCore.Qt.UserRole))
        ds = self.state.current()
        self._show_step(ds, self.state.run(ds) if ds else None)

    def _toggled(self, it):
        sid = it.data(QtCore.Qt.UserRole)
        if sid is None:
            return
        on = it.checkState() == QtCore.Qt.Checked
        self.state.edit("pipeline", lambda p: p.set_enabled(sid, on),
                        ("Enable: " if on else "Disable: ") + it.text())

    # ------------------------------------------------------------------ editing
    def add_step(self, op):
        ds = self.state.current()
        if ds is None:
            return
        idx = None
        if self.state.selected_step is not None:
            idx = ds.pipeline.index(self.state.selected_step) + 1
        params = self._defaults_for(op, ds)

        def mut(p):
            return p.add(op, params, index=idx).id
        step_id = self.state.edit("pipeline", mut, "Add step: " + op)
        if step_id:
            self.state.select_step(step_id)
            self.refresh()

    def _defaults_for(self, op, ds):
        """Sensible data-dependent defaults (e.g. anchors at the ends)."""
        s = self.state.run(ds).final if self.state.selected_step is None else \
            (self.state.run(ds).result(self.state.selected_step).output or ds.raw)
        if op == "baseline_anchors" and s.n > 2:
            n = 6
            xs = [float(s.x[int(round(i * (s.n - 1) / (n - 1)))]) for i in range(n)]
            return {"anchors": [[x, None] for x in xs]}
        if op == "crop" and s.n > 2:
            span = s.x[-1] - s.x[0]
            return {"xmin": float(s.x[0] + 0.05 * span), "xmax": float(s.x[-1] - 0.05 * span)}
        if op == "baseline_snip":
            return {"max_half_window": max(5, s.n // 40)}
        return {}

    def _move(self, d):
        sid = self.state.selected_step
        ds = self.state.current()
        if sid is None or ds is None:
            return
        i = ds.pipeline.index(sid)
        j = min(max(i + d, 0), len(ds.pipeline) - 1)
        if i != j:
            self.state.edit("pipeline", lambda p: p.move(sid, j), "Move step")

    def _remove(self):
        sid = self.state.selected_step
        if sid is not None:
            self.state.edit("pipeline", lambda p: p.remove(sid), "Remove step")

    # ------------------------------------------------------------------ form
    def _hint_fn(self, step, run):
        if step.op not in ("baseline_asls", "baseline_arpls", "smooth_whittaker", "baseline_snip"):
            return None
        res = run.result(step.id) if run else None
        dx = res.input.dx_median if res is not None and res.input is not None else float("nan")

        def hint(params):
            if step.op == "baseline_snip":
                return f"Half-window ≈ {params['max_half_window'] * dx:.4g} x units"
            pts = whittaker_cutoff_points(params["lam"], params.get("diff_order", 2))
            return (f"Structures with a period ≳ {pts:.3g} points (≈ {pts * dx:.4g} x units) remain in the "
                    "baseline; λ depends on the point density and is not transferable between spectra.")
        return hint

    def _show_step(self, ds, run):
        sid = self.state.selected_step
        step = None
        if ds is not None and sid is not None:
            try:
                step = ds.pipeline.get(sid)
            except KeyError:
                step = None
        if step is None:
            self.form = None
            self._form_step = None
            lab = QtWidgets.QLabel("Select a step or add one with “+ Step”.\n\n"
                                   "Recommended order: spikes → units → baseline → (normalization) → fit. "
                                   "Smoothing is for display only.")
            lab.setWordWrap(True)
            lab.setObjectName("hint")
            self.scroll.setWidget(lab)
            self.status.setText("")
            return
        res = run.result(step.id) if run else None
        msgs = []
        if res is not None and res.error:
            msgs.append(f"<span style='color:{SEVERITY_COLOR['error']}'>✖ {res.error}</span>")
        for w in (res.warnings if res is not None else []):
            msgs.append(f"<span style='color:{SEVERITY_COLOR['warning']}'>⚠ {w}</span>")
        if res is not None and not res.error:
            msgs.append(f"<span style='color:#5d6470'>{res.seconds * 1e3:.1f} ms"
                        f"{' (cached)' if res.cached else ''}</span>")
        self.status.setText("<br>".join(msgs))
        if self.form is not None and self._form_step == (step.id, step.op):
            self.form.set_params(step.params)
            return
        from ..ops import get_op
        spec = get_op(step.op)
        self.form = ParamForm(spec, step.params, hint_fn=self._hint_fn(step, run))
        self._form_step = (step.id, step.op)
        self.form.changed.connect(lambda n, v, final, s=step.id, t=spec.title: self._param_changed(s, t, n, v))
        self.form.rangeTarget.connect(self.rangeTargetChanged.emit)
        self.form.editRanges.connect(lambda n, s=step.id: self._edit_ranges(s, n))
        self.scroll.setWidget(self.form)
        target = self.form.active_ranges_param()
        if target:
            self.rangeTargetChanged.emit(target)

    def _param_changed(self, step_id, title, name, value):
        self.state.edit("pipeline", lambda p: p.update(step_id, {name: value}), f"{title}: {name}",
                        merge_key=("param", step_id, name))

    def _edit_ranges(self, step_id, name):
        ds = self.state.current()
        step = ds.pipeline.get(step_id)
        dlg = RangesDialog(step.params.get(name, []), name, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.state.edit("pipeline", lambda p: p.update(step_id, {name: dlg.ranges()}), f"Ranges: {name}")
