"""Dataset list and processing pipeline editor."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..ops import list_ops
from ..ops.baseline import whittaker_cutoff_points
from .param_form import ParamForm, RangesDialog
from .theme import SEVERITY_COLOR

CATEGORY_ORDER = ["Range", "Correction", "Baseline", "Smoothing", "Uncertainty", "Scaling", "Units"]


class _DatasetList(QtWidgets.QListWidget):
    """List whose check boxes can be ticked without making the row the active dataset."""

    def _on_checkbox(self, pos) -> QtWidgets.QListWidgetItem | None:
        it = self.itemAt(pos)
        if it is None or not (it.flags() & QtCore.Qt.ItemIsUserCheckable):
            return None
        width = self.style().pixelMetric(QtWidgets.QStyle.PM_IndicatorWidth, None, self)
        return it if pos.x() <= self.visualItemRect(it).left() + width + 6 else None

    def mousePressEvent(self, ev):
        it = self._on_checkbox(ev.position().toPoint()) if ev.button() == QtCore.Qt.LeftButton else None
        if it is not None:
            on = it.checkState() != QtCore.Qt.Checked
            it.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._on_checkbox(ev.position().toPoint()) is not None:
            ev.accept()                  # the press has toggled it already
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self._on_checkbox(ev.position().toPoint()) is not None:
            self.mousePressEvent(ev)     # fast clicking on a box toggles it, it does not start renaming
            return
        super().mouseDoubleClickEvent(ev)


class DatasetsPanel(QtWidgets.QWidget):
    """The datasets of the project. The current row is the active dataset (processing, model, fit);
    a ticked dataset stays in the plot while another one is active."""
    importRequested = QtCore.Signal()
    exampleRequested = QtCore.Signal()
    combineRequested = QtCore.Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.clipboard = None            # copied processing / model / fit options (a template dict)
        self._icons = {}
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = _DatasetList()
        self.list.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.setToolTip("Tick a spectrum to keep it in the plot while you work on another one.\n"
                             "Right-click: show/hide, colour, copy and paste processing.")
        self.list.currentRowChanged.connect(self._selected)
        self.list.itemChanged.connect(self._item_changed)
        lay.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        add = QtWidgets.QToolButton()
        add.setText("+ Add")
        add.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        add.setToolTip("Import a file, load example data, combine or duplicate datasets")
        add_menu = QtWidgets.QMenu(add)
        add_menu.addAction("Import data…", self.importRequested.emit)
        add_menu.addAction("Example data", self.exampleRequested.emit)
        add_menu.addSeparator()
        add_menu.addAction("Combine datasets (ratio, difference, formulas)…", self.combineRequested.emit)
        add_menu.addAction("Duplicate the active dataset", self._duplicate)
        add.setMenu(add_menu)
        row.addWidget(add)
        for text, slot, tip in (("Remove", self._remove, "Remove the selected datasets (can be undone)"),
                                (None, None, None),
                                ("Show all", lambda: self.state.set_shown(
                                    [d.id for d in self.state.project.datasets], True),
                                 "Overlay all datasets in the plot (Ctrl+Shift+A)"),
                                ("Only active", lambda: self.state.set_shown([], only=True),
                                 "Show only the active dataset (Ctrl+Shift+H)")):
            if text is None:
                row.addStretch()
                continue
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        state.projectChanged.connect(self.refresh)
        state.currentChanged.connect(self.refresh)
        state.viewChanged.connect(self.refresh)

    # ------------------------------------------------------------------ list
    def _icon(self, color):
        icon = self._icons.get(color)
        if icon is None:
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(color))
            painter.drawRoundedRect(1, 1, 10, 10, 2, 2)
            painter.end()
            icon = self._icons[color] = QtGui.QIcon(pm)
        return icon

    def selected_ids(self) -> list:
        return [self.list.item(i).data(QtCore.Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).isSelected()]

    def selected_datasets(self) -> list:
        """Selected datasets in list order (the active one if nothing else is selected)."""
        out = []
        for i in self.selected_ids():
            try:
                out.append(self.state.get(i))
            except KeyError:
                pass
        cur = self.state.current()
        return out or ([cur] if cur is not None else [])

    def refresh(self):
        selected = set(self.selected_ids())
        if self.state.current_id not in selected:      # the active dataset changed elsewhere: follow it
            selected = {self.state.current_id}
        self.list.blockSignals(True)
        self.list.clear()
        preview = self.state.preview
        for ds in self.state.project.datasets + ([preview] if preview is not None else []):
            it = QtWidgets.QListWidgetItem(ds.name)
            it.setData(QtCore.Qt.UserRole, ds.id)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            it.setIcon(self._icon(self.state.color_of(ds)))
            src = ds.source
            tip = (f"{src.get('filename', '')}\nN = {ds.raw.n}, σ: {ds.raw.sigma_source.label}\n"
                   f"SHA-256 {src.get('sha256', '')[:16]}…")
            if ds is preview:               # like a preview tab: replaced by the next file looked at
                font = it.font()
                font.setItalic(True)
                it.setFont(font)
                tip += "\nPreview – kept as soon as you work on it or open the file with a double click"
            else:
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if self.state.is_shown(ds.id) else QtCore.Qt.Unchecked)
            it.setToolTip(tip)
            self.list.addItem(it)
            if ds.id == self.state.current_id:
                self.list.setCurrentItem(it, QtCore.QItemSelectionModel.NoUpdate)
            it.setSelected(ds.id in selected)
        self.list.blockSignals(False)

    def _selected(self, row):
        it = self.list.item(row)
        if it is not None:
            self.state.set_current(it.data(QtCore.Qt.UserRole))

    def _item_changed(self, it):
        """A row was renamed or its box was ticked."""
        try:
            ds = self.state.get(it.data(QtCore.Qt.UserRole))
        except KeyError:
            return
        name = it.text().strip()
        if name != ds.name:
            if name:
                self.state.rename_dataset(ds, name)
            else:
                self.refresh()
            return
        if not (it.flags() & QtCore.Qt.ItemIsUserCheckable):
            return
        on = it.checkState() == QtCore.Qt.Checked
        if on != self.state.is_shown(ds.id):
            ids = self.selected_ids() if it.isSelected() else []    # a ticked selection is ticked as a whole
            self.state.set_shown(ids if len(ids) > 1 else [ds.id], on)

    # ------------------------------------------------------------------ actions
    def _remove(self):
        self.state.remove_datasets(self.selected_datasets())

    def _duplicate(self):
        from ..project import Dataset
        ds = self.state.current()
        if ds is None:
            return
        copy = Dataset(name=ds.name + " (copy)", raw=ds.raw, pipeline=ds.pipeline.copy(), model=ds.model.copy(),
                       fit_options=type(ds.fit_options).from_dict(ds.fit_options.to_dict()),
                       raw_bytes=ds.raw_bytes, raw_ext=ds.raw_ext)
        self.state.add_dataset(copy)

    def copy_setup(self, ds=None):
        """Remember processing, model and fit options of a dataset for pasting onto others."""
        from ..project import template_of
        ds = ds or self.state.current()
        if ds is not None:
            self.clipboard = {"name": ds.name, **template_of(ds)}
            self.state.message.emit(f"Copied processing and model of {ds.name}.", 4000)

    def paste_setup(self, targets=None, model: bool = False) -> int:
        """Give the selected datasets the copied processing (``model``: also model and fit options)."""
        if self.clipboard is None:
            return 0
        targets = self.selected_datasets() if targets is None else targets
        n = self.state.apply_setup(self.clipboard, targets, model=model)
        self.state.message.emit(f"Applied to {n} dataset(s)." if n else "Nothing to change.", 4000)
        return n

    def _pick_color(self, datasets):
        start = QtGui.QColor(self.state.color_of(datasets[0]))
        color = QtWidgets.QColorDialog.getColor(start, self, "Colour of the spectrum")
        if color.isValid():
            self.state.set_color([d.id for d in datasets], color.name())

    def _context_menu(self, pos):
        st = self.state
        it = self.list.itemAt(pos)
        if it is not None and not it.isSelected():
            self.list.setCurrentItem(it)
        datasets = self.selected_datasets() if it is not None else []
        real = [d for d in datasets if not st.is_preview(d)]
        ids = [d.id for d in real]
        n = len(real)
        menu = QtWidgets.QMenu(self)
        if real:
            if any(not st.is_shown(i) for i in ids):
                menu.addAction("Show in plot" if n == 1 else f"Show {n} in plot", lambda: st.set_shown(ids, True))
            if any(st.is_shown(i) for i in ids):
                menu.addAction("Hide from plot" if n == 1 else f"Hide {n} from plot",
                               lambda: st.set_shown(ids, False))
            menu.addAction("Show only this" if n == 1 else "Show only these", lambda: st.set_shown(ids, only=True))
        all_ids = [d.id for d in st.project.datasets]
        if all_ids:
            menu.addAction("Show all", lambda: st.set_shown(all_ids, True))
            menu.addAction("Hide all others", lambda: st.set_shown([], only=True))
        if real:
            menu.addSeparator()
            menu.addAction("Colour…", lambda: self._pick_color(real))
            menu.addSeparator()
            if n == 1:
                menu.addAction("Copy processing and model", lambda: self.copy_setup(real[0]))
            if self.clipboard is not None:
                src = self.clipboard["name"]
                menu.addAction(f"Paste processing of “{src}”", lambda: self.paste_setup(real))
                menu.addAction(f"Paste processing and model of “{src}”", lambda: self.paste_setup(real, model=True))
            menu.addSeparator()
            if n == 1:
                menu.addAction("Rename", lambda: self.list.editItem(it))
        if datasets:
            menu.addAction("Remove" if len(datasets) == 1 else f"Remove {len(datasets)} datasets",
                           lambda: st.remove_datasets(datasets))
        if not menu.isEmpty():
            menu.exec(self.list.viewport().mapToGlobal(pos))


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
