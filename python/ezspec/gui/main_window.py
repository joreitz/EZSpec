"""Main window: wires state, plot, panels, dialogs and background tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from .. import units as U
from .._backend import HAVE_RUST, backend_name, set_backend
from ..export.figure import curves_for, default_spec
from ..export.results import write_all
from ..export.script import generate_script
from ..fit import bootstrap, fit, profile_ci
from ..models import add_peak, find_peaks
from ..ops.baseline import anchor_values
from ..project import Dataset, Project
from ..slices import grouped_by_slices, slice_curves, slice_variable
from ..sweeps import display_order, display_order_masked
from . import peak_edit
from .dialogs import (BootstrapDialog, CalibrationDialog, CombineDialog, CompareDialog, FigureDialog,
                      ImportDialog, SeriesDialog)
from .model_panel import ModelPanel
from .pipeline_panel import DatasetsPanel, PipelinePanel
from .plot_view import PlotView
from .results_panel import ResultsPanel
from .state import AppState, start_task


def _disp(s, *arrays):
    """Arrays of spectrum ``s`` in drawing order: acquisition order for
    back-and-forth (multi-sweep) data, so up and down ramps are drawn as
    separate traces instead of a zigzag between them."""
    o = display_order(s)
    if o is None:
        return arrays
    return tuple(np.asarray(a)[o] for a in arrays)


STATS_HELP = """<h3>How EZSpec reports goodness of fit</h3>
<ul>
<li>The <b>σ source</b> is mandatory: known (column), estimated (DER_SNR / flat region), Poisson (σ² = model)
or unknown. χ² is a true χ² only when σ is known; otherwise RSS, s = √(RSS/ν) and RMSE are shown.</li>
<li><b>Covariance</b>: <i>absolute</i> when σ is known. Scaling by √χ²_ν would otherwise hide a misfit behind
larger error bars – it is possible, but flagged. With unknown σ, scaling is required.</li>
<li><b>χ²_ν</b> is shown together with its expected range 1 ± √(2/ν); it scatters even when the model is correct.</li>
<li><b>R²</b> is descriptive only. For nonlinear models it is a poor criterion for selecting the true model
(Spiess &amp; Neumeyer 2010); for model comparison use ΔAICc/ΔBIC on identical data.</li>
<li><b>Parameters at a bound</b>, <b>|ρ| &gt; 0.9</b> and rank-deficient Jacobians are flagged;
standard errors are then unreliable → use profile CI or bootstrap.</li>
<li>Do not fit <b>smoothed or interpolated data</b>: correlated noise leads to severely underestimated
errors (O'Haver). By default, smoothing in EZSpec is for display only.</li>
<li>Do not decide on <b>“one more peak”</b> with an F-test (boundary problem, Protassov et al. 2002).</li>
<li>Do not weight <b>Poisson data</b> with σ = √y (biased, Humphrey et al. 2009); the weighting “Poisson:
σ² = model” yields the Poisson maximum-likelihood estimator.</li>
<li>The numerical accuracy of the fit engine is verified against all 27 NIST StRD problems
(≥ 7 significant digits in the parameters).</li>
</ul>"""


QUICKSTART = """<h3>Quick start</h3>
<ol>
<li><b>Data</b>: Import (text/CSV, JCAMP-DX) or drag a file onto the window. Example: File → Example data.</li>
<li><b>Processing</b>: “+ Step”. The selected step shows its input and output; parameter changes apply live.
Plot tools (keys 1–4): Navigate · Anchor (click to set, drag to move, right-click to delete, Shift = free) ·
Range (masks, forced regions, noise/fit range) · Peak.</li>
<li><b>σ</b>: import a σ column or add “Estimate noise → σ” – otherwise σ is treated as unknown (no χ²).</li>
<li><b>Model</b>: Peak tool (click = new peak; drag markers = position/height/width), “Find peaks”,
“+ Component” (background, classic function, custom formula). Table: start value, bounds, vary, expression.</li>
<li><b>Fit</b>: Ctrl+R. Results panel: statistics, warnings, derived quantities, correlations, residuals.</li>
<li><b>Combine</b>: Ctrl+K – ratio, difference or arbitrary x/y formulas from several datasets
(σ is propagated, alignment is checked automatically).</li>
<li><b>Up/down sweeps</b> (e.g. a current ramp): displayed in acquisition order; “+ Step → Range →
Select sweeps / Average sweeps” separates rising and falling ramps.</li>
<li><b>Calibration</b> (e.g. λ(I, T)): on import, select column T under “Extra variables”,
“+ Component → Surface f(x, v)”, fit, then Analysis → “Apply fit as x calibration”.</li>
<li><b>Go further</b>: profile CI, bootstrap, MCMC, baseline systematics, compare variants, series/global fit (Ctrl+G).</li>
<li><b>Export</b>: figure (Ctrl+E), tables, Python script (reproduces everything from the raw data).</li>
</ol>
<p>Every action can be undone (Ctrl+Z); the history is shown in the “History” dock.</p>"""


def estimate_fwhm(x, y, x0, base, height):
    """Width at half height around x0 in the data (start value for a new peak)."""
    if len(x) < 3:
        return 1.0
    i = int(np.argmin(np.abs(x - x0)))
    target = base + 0.5 * height
    lo = i
    while lo > 0 and y[lo] > target:
        lo -= 1
    hi = i
    while hi < len(x) - 1 and y[hi] > target:
        hi += 1
    dx = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    w = float(x[hi] - x[lo])
    return float(min(max(w, 3 * abs(dx)), 0.3 * abs(span)))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState(self)
        self.settings = QtCore.QSettings("EZSpec", "EZSpec")
        self._task = None
        self._range_target = None
        self.setAcceptDrops(True)

        self.plot = PlotView()
        self.setCentralWidget(self.plot)
        self.datasets_panel = DatasetsPanel(self.state)
        self.pipeline_panel = PipelinePanel(self.state)
        self.model_panel = ModelPanel(self.state)
        self.results_panel = ResultsPanel(self.state)
        self.d_data = self._dock("Data", self.datasets_panel, QtCore.Qt.LeftDockWidgetArea)
        self.d_pipe = self._dock("Processing", self.pipeline_panel, QtCore.Qt.LeftDockWidgetArea)
        self.d_model = self._dock("Model and fit", self.model_panel, QtCore.Qt.RightDockWidgetArea)
        self.d_res = self._dock("Results", self.results_panel, QtCore.Qt.RightDockWidgetArea)
        undo_view = QtWidgets.QUndoView(self.state.undo)
        undo_view.setEmptyLabel("<Project start>")
        self.d_hist = self._dock("History", undo_view, QtCore.Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(self.d_pipe, self.d_hist)
        self.d_pipe.raise_()
        self.resizeDocks([self.d_data, self.d_pipe], [140, 520], QtCore.Qt.Vertical)
        self.resizeDocks([self.d_data, self.d_model], [380, 560], QtCore.Qt.Horizontal)
        self.resizeDocks([self.d_model, self.d_res], [520, 420], QtCore.Qt.Vertical)

        self.busy = QtWidgets.QProgressBar()
        self.busy.setMaximumWidth(160)
        self.busy.setVisible(False)
        self.cursor_label = QtWidgets.QLabel()
        self.backend_label = QtWidgets.QLabel()
        sb = self.statusBar()
        sb.addPermanentWidget(self.cursor_label)
        sb.addPermanentWidget(self.busy)
        sb.addPermanentWidget(self.backend_label)
        self._update_backend_label()

        self._refresh_timer = QtCore.QTimer(self, singleShot=True, interval=25)
        self._refresh_timer.timeout.connect(self.refresh_plot)
        self._live_timer = QtCore.QTimer(self, singleShot=True, interval=350)
        self._live_timer.timeout.connect(self.run_fit)
        self._build_menus()
        self._wire()
        self._update_title()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 920)

    # ================================================================ layout helpers
    def _dock(self, title, widget, area):
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable |
                      QtWidgets.QDockWidget.DockWidgetClosable)
        self.addDockWidget(area, d)
        return d

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction("New project", self.new_project, QtGui.QKeySequence.New)
        f.addAction("Open project…", self.open_project, QtGui.QKeySequence.Open)
        f.addAction("Save", self.save_project, QtGui.QKeySequence.Save)
        f.addAction("Save as…", lambda: self.save_project(True), QtGui.QKeySequence.SaveAs)
        f.addSeparator()
        f.addAction("Import data…", self.import_data, QtGui.QKeySequence("Ctrl+I"))
        ex = f.addMenu("Example data")
        ex.addAction("Raman spectrum with fluorescence background", lambda: self.load_example("raman"))
        ex.addAction("Decay curve (Poisson counts)", lambda: self.load_example("decay"))
        f.addSeparator()
        f.addAction("Save template…", self.save_template)
        f.addAction("Apply template…", self.apply_template)
        f.addSeparator()
        exp = f.addMenu("Export")
        exp.addAction("Figure…", self.figure_dialog, QtGui.QKeySequence("Ctrl+E"))
        exp.addAction("Fit results (CSV/JSON/report)…", self.export_results)
        exp.addAction("Python script (reproducible)…", self.export_script)
        f.addSeparator()
        f.addAction("Quit", self.close, QtGui.QKeySequence.Quit)

        e = mb.addMenu("&Edit")
        undo = self.state.undo.createUndoAction(self, "Undo")
        undo.setShortcut(QtGui.QKeySequence.Undo)
        redo = self.state.undo.createRedoAction(self, "Redo")
        redo.setShortcuts([QtGui.QKeySequence.Redo, QtGui.QKeySequence("Ctrl+Y")])
        e.addAction(undo)
        e.addAction(redo)

        a = mb.addMenu("&Analysis")
        a.addAction("Run fit", self.run_fit, QtGui.QKeySequence("Ctrl+R"))
        a.addAction("Find peaks automatically", self.auto_peaks)
        a.addAction("Profile confidence intervals", self.run_profile)
        a.addAction("Bootstrap…", self.run_bootstrap)
        a.addAction("Baseline systematics (vary λ/window)", self.run_systematics)
        a.addAction("MCMC-Posterior (emcee)…", self.run_mcmc)
        a.addSeparator()
        a.addAction("Combine datasets (ratio, difference, x/y formulas)…", self.combine_dialog,
                    QtGui.QKeySequence("Ctrl+K"))
        a.addAction("Series / global fit…", self.series_dialog, QtGui.QKeySequence("Ctrl+G"))
        a.addAction("Apply fit as x calibration (e.g. λ(I, T))…", lambda: self.calibration_dialog())
        a.addSeparator()
        a.addAction("Remember fit as variant", self.remember_variant)
        a.addAction("Compare models…", self.compare_variants)
        a.addSeparator()
        bk = a.addMenu("Numerics backend")
        grp = QtGui.QActionGroup(self)
        for key, title in (("rust", "Rust (fast)"), ("python", "Python/SciPy (reference)")):
            act = QtGui.QAction(title, self, checkable=True)
            act.setChecked(backend_name() == key)
            act.setEnabled(key == "python" or HAVE_RUST)
            act.triggered.connect(lambda _=False, k=key: self.switch_backend(k))
            grp.addAction(act)
            bk.addAction(act)

        v = mb.addMenu("&View")
        for d in (self.d_data, self.d_pipe, self.d_model, self.d_res, self.d_hist):
            v.addAction(d.toggleViewAction())
        v.addSeparator()
        for i, key in enumerate(("navigate", "anchor", "region", "peak"), 1):
            act = v.addAction(f"Tool: {self.plot.mode_actions[key].text()}",
                              lambda k=key: self.plot.set_mode(k))
            act.setShortcut(QtGui.QKeySequence(str(i)))

        h = mb.addMenu("&Help")
        h.addAction("Quick start", lambda: QtWidgets.QMessageBox.information(self, "Quick start", QUICKSTART),
                    QtGui.QKeySequence.HelpContents)
        h.addAction("Statistics notes", lambda: QtWidgets.QMessageBox.information(self, "Statistics notes", STATS_HELP))
        h.addAction("About EZSpec", self.about)

    def _wire(self):
        st = self.state
        for sig in (st.currentChanged, st.projectChanged):
            sig.connect(self.schedule_refresh)
        st.pipelineChanged.connect(lambda _id: self.schedule_refresh())
        st.modelChanged.connect(self._model_changed)
        st.fitChanged.connect(lambda _id: self.schedule_refresh())
        st.fitOptionsChanged.connect(lambda _id: self.schedule_refresh())
        st.stepSelected.connect(lambda _s: self.schedule_refresh())
        st.undo.cleanChanged.connect(lambda _c: self._update_title())
        st.projectChanged.connect(self._update_title)
        self.datasets_panel.importRequested.connect(self.import_data)
        self.datasets_panel.exampleRequested.connect(lambda: self.load_example("raman"))
        self.datasets_panel.combineRequested.connect(self.combine_dialog)
        self.pipeline_panel.calibrationRequested.connect(lambda: self.calibration_dialog(from_pipeline=True))
        self.pipeline_panel.rangeTargetChanged.connect(self.plot.set_range_target)
        self.model_panel.fitRequested.connect(self.run_fit)
        self.model_panel.autoPeaksRequested.connect(self.auto_peaks)
        self.results_panel.profileRequested.connect(self.run_profile)
        self.results_panel.bootstrapRequested.connect(self.run_bootstrap)
        self.results_panel.variantRequested.connect(self.remember_variant)
        self.results_panel.compareRequested.connect(self.compare_variants)
        self.results_panel.exportRequested.connect(self.export_results)
        p = self.plot
        p.anchorsEdited.connect(self._anchors_edited)
        p.rangesEdited.connect(self._ranges_edited)
        p.rangeEdited.connect(self._range_edited)
        p.peakAdded.connect(self._peak_added)
        p.peakDragged.connect(self._peak_dragged)
        p.cursorMoved.connect(lambda x, y: self.cursor_label.setText(f"x = {x:.6g}   y = {y:.6g}"))

    # ================================================================ refresh
    def schedule_refresh(self, *args):
        self._refresh_timer.start()

    def _model_changed(self, _id):
        self.schedule_refresh()
        if self.model_panel.live.isChecked():
            self._live_timer.start()

    def _values_for_handles(self, ds):
        if self.state.fit_current(ds):
            return ds.fit_result.values
        try:
            return ds.model.initial_values()
        except Exception:  # noqa: BLE001
            return None

    def refresh_plot(self):
        ds = self.state.current()
        if ds is None:
            self.plot.set_scene(data=([], []))
            self.plot.set_tools()
            return
        run = self.state.run(ds)
        sid = self.state.selected_step
        step = None
        if sid is not None:
            try:
                step = ds.pipeline.get(sid)
            except KeyError:
                step = None
        if step is not None:
            res = run.result(step.id)
            inp = res.input if res.input is not None else ds.raw
            out = res.output
            xl = U.PLAIN_LABELS.get(inp.x_unit, inp.x_label)
            kw = {"x_label": xl, "y_label": inp.y_label}
            anchors_eff = None
            if out is None:
                self.plot.set_scene(data=_disp(inp, inp.x, inp.y), **kw)
            elif step.op.startswith("baseline_"):
                corrected = (out.x, out.y) if step.params.get("subtract", True) else (out.x, out.y - out.aux["baseline"])
                self.plot.set_scene(data=_disp(inp, inp.x, inp.y), baseline=_disp(out, out.x, out.aux["baseline"]),
                                    residuals=(*_disp(out, *corrected), False), **kw)
                self.plot.p_res.setLabel("left", "corrected")
                if step.op == "baseline_anchors":
                    try:
                        anchors_eff = anchor_values(inp, step.params.get("anchors", []), step.params.get("window", 3))
                    except Exception:  # noqa: BLE001
                        anchors_eff = None
            elif "smoothed" in out.aux and "smoothed" not in inp.aux:
                self.plot.set_scene(data=_disp(out, out.x, out.y), smoothed=_disp(out, out.x, out.aux["smoothed"]),
                                    **kw)
            elif step.op == "estimate_noise" or step.op == "exclude":
                excl = out.exclude if out.exclude is not None else np.zeros(out.n, bool)
                self.plot.set_scene(data=_disp(out, out.x, out.y), excluded=(out.x[excl], out.y[excl]), **kw)
            else:
                self.plot.set_scene(data=_disp(out, out.x, out.y), input=_disp(inp, inp.x, inp.y), **kw)
            self.plot.set_tools(step=step, anchors_eff=anchors_eff)
            return

        s = run.final
        xl = U.PLAIN_LABELS.get(s.x_unit, s.x_label)
        self.plot.p_res.setLabel("left", "Res.")
        mask = s.fit_mask
        excluded = (s.x[~mask], s.y[~mask]) if (~mask).any() else None
        smoothed = _disp(s, s.x, s.aux["smoothed"]) if "smoothed" in s.aux else None
        fit_xy = comps = resid = preview = None
        current = self.state.fit_current(ds)
        r = ds.fit_result
        if r is not None:
            c = curves_for(s, None, r, r._internals.get("model"), n_dense=1500)
            fit_xy = c["fit_dense"][:2]
            comps = [(k.split(":", 1)[1].rstrip("_"), *v[:2]) for k, v in c.items()
                     if k.startswith("component_dense:")]
            if len(comps) < 2:
                comps = []
            rx, ry = r.x, r.normalized_residuals
            o = display_order_masked(s, r.mask) if len(r.mask) == s.n else None
            if o is not None:
                rx, ry = rx[o], ry[o]
            resid = (rx, ry, r.stats.chi2 is not None)
        if ds.model.components and (r is None or not current) and s.n > 1:
            try:
                xd = np.linspace(s.x.min(), s.x.max(), 1500)
                preview = (xd, ds.model.evaluate(xd, ds.model.initial_values()))
            except Exception:  # noqa: BLE001 - invalid model: no preview
                preview = None
        # y(x, v) data such as a calibration surface: one group/curve per value of v
        fit_model = r._internals.get("model") if r is not None else None
        var = slice_variable(s, fit_model if fit_model is not None else ds.model)
        slices = ()
        if var is not None:
            slices = slice_curves(s, fit_model, r.values if r is not None else None, var)
            fit_xy, comps, preview = None, [], None
            if resid is not None and len(r.mask) == s.n:
                resid = (*grouped_by_slices(slices, r.mask, r.x, r.normalized_residuals), resid[2])
        self.plot.set_scene(data=_disp(s, s.x, s.y), excluded=excluded, smoothed=smoothed, fit=fit_xy,
                            fit_stale=not current, components=comps or (), preview=preview, residuals=resid,
                            x_label=xl, y_label=s.y_label, scatter=s.meta.get("plot_style") == "scatter",
                            slices=slices)
        handles = []
        vals = self._values_for_handles(ds)
        if vals is not None and ds.model.peaks:
            handles = peak_edit.handles(ds.model, vals)
        self.plot.set_tools(step=None, peaks=handles, fit_range=ds.fit_options.x_range)

    # ================================================================ plot interactions
    def _step_edit(self, params, text):
        sid = self.state.selected_step
        if sid is None:
            return
        self.state.edit("pipeline", lambda p: p.update(sid, params), text)

    def _anchors_edited(self, anchors):
        self._step_edit({"anchors": anchors}, "Edit anchors")

    def _ranges_edited(self, param, lst):
        self._step_edit({param: [[min(a, b), max(a, b)] for a, b in lst]}, f"Ranges: {param}")

    def _range_edited(self, param, ab):
        a, b = ab
        lo = None if a is None or b is None else min(a, b)
        hi = None if a is None or b is None else max(a, b)
        if param == "crop":
            self._step_edit({"xmin": lo, "xmax": hi}, "Crop range")
        elif param == "fit_range":
            self.state.edit("options", lambda o: setattr(o, "x_range", None if lo is None else [lo, hi]),
                            "Fit range")
        else:
            self._step_edit({param: [lo, hi]}, f"Range: {param}")

    def _peak_added(self, x, y):
        ds = self.state.current()
        if ds is None:
            return
        s = self.state.run(ds).final
        kind = self.model_panel.peak_kind.currentData()
        base = 0.0
        try:
            vals = ds.model.initial_values()
            base = float(sum(c.evaluate(np.array([x]), vals)[0] for c in ds.model.components if not c.is_peak))
        except Exception:  # noqa: BLE001
            pass
        h = y - base
        fw = estimate_fwhm(s.x, s.y, x, base, h)
        self.state.edit("model", lambda m: add_peak(m, kind, x, h, fw), f"Peak at {x:.5g}")

    def _peak_dragged(self, prefix, handle, x, y, final):
        ds = self.state.current()
        if ds is None:
            return
        vals = self._values_for_handles(ds)
        if vals is None:
            return
        try:
            comp = ds.model.component(prefix)
            new = peak_edit.drag(ds.model, comp, vals, handle, x, y)
        except Exception:  # noqa: BLE001
            return
        if not final:
            v = dict(vals)
            v.update({comp.full_name(k): val for k, val in new.items()})
            s = self.state.run(ds).final
            xd = np.linspace(s.x.min(), s.x.max(), 1500)
            try:
                self.plot.update_preview_curve((xd, ds.model.evaluate(xd, v)))
            except Exception:  # noqa: BLE001
                pass
            return

        def mut(m):
            m.apply_values(vals)
            c = m.component(prefix)
            for k, val in new.items():
                st = c.settings[k]
                st.value = float(min(max(val, st.min), st.max))
        self.state.edit("model", mut, f"Move peak {prefix.rstrip('_')}")

    # ================================================================ data / project
    def import_data(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import data", self.settings.value("last_dir", ""),
                "Spectra (*.txt *.csv *.dat *.tsv *.xy *.asc *.prn *.jdx *.dx *.jcm);;All files (*)")
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        try:
            dlg = ImportDialog(path, self)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import", f"Cannot read file:\n{e}")
            return
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self.state.undo.beginMacro(f"Import {Path(path).name}")
        for ds in dlg.datasets:
            self.state.add_dataset(ds)
        self.state.undo.endMacro()

    def load_example(self, which):
        from ..examples import decay_example, raman_example, to_csv_bytes
        s = raman_example() if which == "raman" else decay_example()
        name = "example_raman.csv" if which == "raman" else "example_decay.csv"
        ds = Dataset.from_bytes(to_csv_bytes(s), name)
        ds.name = s.meta["name"]
        if which == "raman":
            ds.pipeline.add("set_units", {"x_unit": "raman", "y_label": "Intensity (counts)"})
        self.state.add_dataset(ds)

    def save_template(self):
        from ..project import save_template
        ds = self.state.current()
        if ds is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save template", f"{ds.name}.ezspec-template.json",
                                                        "EZSpec template (*.json)")
        if path:
            save_template(ds, path)
            self.statusBar().showMessage(f"Template saved: {path}", 4000)

    def apply_template(self, path=None):
        from ..project import load_template
        ds = self.state.current()
        if ds is None:
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Apply template", "", "EZSpec template (*.json)")
        if not path:
            return
        try:
            t = load_template(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Template", str(e))
            return
        from ..fit import FitOptions
        from ..models import Model
        from ..pipeline import Pipeline
        self.state.undo.beginMacro("Apply template")
        self.state.edit("pipeline", lambda p: p.steps.__setitem__(slice(None), Pipeline.from_dict(t["pipeline"]).steps),
                        "Pipeline from template")
        self.state.edit("model", lambda m: m.components.__setitem__(slice(None), Model.from_dict(t["model"]).components),
                        "Model from template")
        self.state.edit("options", lambda o: o.__dict__.update(FitOptions.from_dict(t["fit_options"]).to_dict()),
                        "Fit options from template")
        self.state.undo.endMacro()

    def new_project(self):
        if not self._confirm_discard():
            return
        self.state.new_project(Project())
        self._update_title()

    def open_project(self, path=None):
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", self.settings.value("last_dir", ""),
                                                            "EZSpec project (*.ezspec)")
        if not path:
            return
        try:
            proj = Project.load(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Open", str(e))
            return
        self.state.new_project(proj)
        self._update_title()
        self._refit_loaded()

    def _refit_loaded(self):
        """Recompute stored fits (deterministic) so that live results are available."""
        for ds in self.state.project.datasets:
            if ds.fit_record is not None and ds.model.components:
                try:
                    ds.run_fit()
                    self.state.fitChanged.emit(ds.id)
                except Exception as e:  # noqa: BLE001
                    self.statusBar().showMessage(f"Fit of {ds.name} could not be reproduced: {e}", 8000)

    def save_project(self, save_as=False):
        proj = self.state.project
        path = proj.path
        if save_as or path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project",
                                                            str(Path(self.settings.value("last_dir", "")) /
                                                                "project.ezspec"), "EZSpec project (*.ezspec)")
            if not path:
                return False
            if not path.endswith(".ezspec"):
                path += ".ezspec"
        try:
            proj.save(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Save", str(e))
            return False
        self.state.undo.setClean()
        self._update_title()
        self.statusBar().showMessage(f"Saved: {path}", 4000)
        return True

    def _confirm_discard(self):
        if self.state.undo.isClean() or not self.state.project.datasets:
            return True
        r = QtWidgets.QMessageBox.question(self, "Unsaved changes", "Save changes?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
                                           QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return r == QtWidgets.QMessageBox.Discard

    def _update_title(self):
        p = self.state.project.path
        name = Path(p).name if p else "Untitled"
        try:
            dirty = "" if self.state.undo.isClean() else " *"
        except RuntimeError:  # undo stack already destroyed during shutdown
            return
        self.setWindowTitle(f"{name}{dirty} – EZSpec {__version__}")

    # ================================================================ fitting
    def _busy(self, on, text=""):
        self.busy.setVisible(on)
        self.busy.setRange(0, 0)
        self.model_panel.fit_btn.setEnabled(not on)
        if text:
            self.statusBar().showMessage(text, 0 if on else 4000)

    def run_fit(self):
        ds = self.state.current()
        if ds is None or not ds.model.components:
            self.statusBar().showMessage("Add model components first (e.g. with the 'Peak' tool).", 5000)
            return
        if self._task is not None:
            self._live_timer.start()
            return
        s = self.state.run(ds).final
        model = ds.model.copy()
        options = type(ds.fit_options).from_dict(ds.fit_options.to_dict())
        self._busy(True, "Fitting…")
        ds_id = ds.id

        def done(result):
            self._task = None
            self._busy(False, "Fit done.")
            try:
                d = self.state.project.get(ds_id)
            except KeyError:
                return
            d.fit_result = result
            d.fit_record = result.to_dict(include_curves=False)
            self.state.fitChanged.emit(ds_id)

        def failed(msg):
            self._task = None
            self._busy(False, "Fit failed.")
            QtWidgets.QMessageBox.warning(self, "Fit", msg.split("\n\n")[0])

        self._task = start_task(fit, done, failed, None, s, model, options)

    def auto_peaks(self):
        ds = self.state.current()
        if ds is None:
            return
        s = self.state.run(ds).final
        m = s.fit_mask
        guesses = find_peaks(s.x[m], s.y[m])
        if not guesses:
            self.statusBar().showMessage("No peaks found above 5 × noise.", 5000)
            return
        kind = self.model_panel.peak_kind.currentData()
        existing = []
        try:
            for c in ds.model.peaks:
                existing.append(c.derived(ds.model.initial_values())["center"])
        except Exception:  # noqa: BLE001
            pass

        def mut(model):
            n = 0
            for g in guesses:
                if any(abs(g.center - e) < 0.5 * g.fwhm for e in existing):
                    continue
                add_peak(model, kind, g.center, g.height, g.fwhm)
                n += 1
            return n
        n = self.state.edit("model", mut, "Find peaks automatically")
        self.statusBar().showMessage(f"{n or 0} peak(s) added.", 4000)

    def _require_fit(self):
        ds = self.state.current()
        if ds is None or ds.fit_result is None:
            QtWidgets.QMessageBox.information(self, "Analysis", "Run a fit first.")
            return None
        return ds

    def run_profile(self):
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        self._busy(True, "Profile confidence intervals…")

        def done(_res):
            self._task = None
            self._busy(False, "Profile CI done.")
            self.state.fitChanged.emit(ds.id)
            self.results_panel.tabs.setCurrentIndex(4)

        def failed(msg):
            self._task = None
            self._busy(False)
            QtWidgets.QMessageBox.warning(self, "Profile CI", msg.split("\n\n")[0])
        self._task = start_task(profile_ci, done, failed, None, ds.fit_result)

    def run_bootstrap(self):
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        dlg = BootstrapDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._busy(True, "Bootstrap…")

        def done(_res):
            self._task = None
            self._busy(False, "Bootstrap done.")
            self.state.fitChanged.emit(ds.id)
            self.results_panel.tabs.setCurrentIndex(4)

        def failed(msg):
            self._task = None
            self._busy(False)
            QtWidgets.QMessageBox.warning(self, "Bootstrap", msg.split("\n\n")[0])
        self._task = start_task(bootstrap, done, failed, None, ds.fit_result, dlg.n.value(),
                                dlg.kind.currentData(), dlg.seed.value())

    def combine_dialog(self):
        if not self.state.project.datasets:
            QtWidgets.QMessageBox.information(self, "Combine", "Import data first.")
            return
        CombineDialog(self.state, self).exec()

    def calibration_dialog(self, from_pipeline=False):
        st = self.state
        cur = st.current()
        sources = [d for d in st.project.datasets if d.fit_result is not None and st.fit_current(d)]
        if from_pipeline:
            sources = [d for d in sources if cur is None or d.id != cur.id]
        if not sources:
            QtWidgets.QMessageBox.information(
                self, "Calibration",
                "No calibration fit available: select the calibration data (e.g. λ vs. I, with column T as an "
                "additional variable), add + Component → Surface f(x, v) → Plane, fit – then open this dialog.")
            return
        if len(st.project.datasets) < 2:
            QtWidgets.QMessageBox.information(self, "Calibration", "No target datasets – import the measurement data.")
            return
        source = next((d for d in sources if cur is not None and d.id == cur.id), sources[0])
        try:
            dlg = CalibrationDialog(st, sources, source=source, target=cur if from_pipeline else None, parent=self)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Calibration", str(e))
            return
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self.apply_calibration(dlg.params, dlg.targets)

    def apply_calibration(self, params, target_ids):
        self.state.undo.beginMacro("Apply x calibration")
        try:
            for tid in target_ids:
                t = self.state.project.get(tid)
                self.state.edit("pipeline", lambda p: p.add("calibrate_x", params), "Calibrate x", ds=t)
        finally:
            self.state.undo.endMacro()
        self.schedule_refresh()

    def series_dialog(self):
        if len(self.state.project.datasets) < 2:
            QtWidgets.QMessageBox.information(self, "Series", "Import at least two datasets.")
            return
        SeriesDialog(self.state, self).exec()

    def run_systematics(self):
        from ..fit import baseline_systematics
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        self._busy(True, "Baseline systematics: fitting variants…")
        result = ds.fit_result

        def done(res):
            self._task = None
            self._busy(False, "Baseline systematics done.")
            result.extra["baseline_systematics"] = res
            self.state.fitChanged.emit(ds.id)
            self.results_panel.tabs.setCurrentIndex(4)

        def failed(msg):
            self._task = None
            self._busy(False)
            QtWidgets.QMessageBox.warning(self, "Baseline systematics", msg.split("\n\n")[0])
        self._task = start_task(baseline_systematics, done, failed, None, ds.raw, ds.pipeline.copy(),
                                ds.model.copy(), type(ds.fit_options).from_dict(ds.fit_options.to_dict()))

    def run_mcmc(self):
        from ..fit import mcmc
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        steps, ok = QtWidgets.QInputDialog.getInt(self, "MCMC", "Steps per walker:", 3000, 200, 200000, 500)
        if not ok:
            return
        self._busy(True, "MCMC running…")
        result = ds.fit_result

        def done(_res):
            self._task = None
            self._busy(False, "MCMC done.")
            self.state.fitChanged.emit(ds.id)
            self.results_panel.tabs.setCurrentIndex(4)

        def failed(msg):
            self._task = None
            self._busy(False)
            QtWidgets.QMessageBox.warning(self, "MCMC", msg.split("\n\n")[0])
        self._task = start_task(mcmc, done, failed, None, result, steps, steps // 3, max(1, steps // 300))

    def remember_variant(self):
        ds = self._require_fit()
        if ds is None:
            return
        variants = self.state.variants.setdefault(ds.id, {})
        default = " + ".join(c.type.title if c.kind != "formula" else "Formula" for c in ds.model.components)
        name, ok = QtWidgets.QInputDialog.getText(self, "Remember variant", "Name:", text=default)
        if ok and name:
            variants[name] = ds.fit_result
            self.statusBar().showMessage(f"Variant “{name}” saved ({len(variants)} in total).", 4000)

    def compare_variants(self):
        ds = self.state.current()
        variants = dict(self.state.variants.get(ds.id, {})) if ds else {}
        if ds is not None and ds.fit_result is not None and ds.fit_result not in variants.values():
            variants["current fit"] = ds.fit_result
        if len(variants) < 2:
            QtWidgets.QMessageBox.information(self, "Comparison", "At least two fits are needed: run a fit, "
                                                                 "“Remember fit as variant”, change the model, "
                                                                 "fit again.")
            return
        CompareDialog(variants, self).exec()

    def switch_backend(self, key):
        set_backend(key)
        for ds in self.state.project.datasets:
            ds.pipeline._cache.clear()
        self.state._runs.clear()
        self._update_backend_label()
        self.schedule_refresh()

    def _update_backend_label(self):
        name = backend_name()
        self.backend_label.setText("Backend: Rust" if name == "rust" else "Backend: Python (reference)")

    # ================================================================ export
    def figure_dialog(self):
        ds = self.state.current()
        if ds is None:
            return
        key = ds.id
        dlg = FigureDialog(ds, self.state.project.figures.get(key), self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.state.project.figures[key] = dlg.spec
            self.statusBar().showMessage("Figure saved in project.", 3000)

    def export_results(self):
        ds = self._require_fit()
        if ds is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results (file stem)",
                                                        f"{ds.name}_fit", "All files (*)")
        if not path:
            return
        files = write_all(ds.fit_result, path)
        self.statusBar().showMessage(f"{len(files)} files written.", 4000)

    def export_script(self):
        ds = self.state.current()
        if ds is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Python script",
                                                        f"{Path(ds.name).stem}_analysis.py", "Python (*.py)")
        if not path:
            return
        path = Path(path)
        src = ds.source
        raw_path = src.get("path")
        if not raw_path or not Path(raw_path).exists():
            if ds.raw_bytes is None:
                QtWidgets.QMessageBox.warning(self, "Script", "No raw data file available.")
                return
            raw_path = path.with_name(path.stem + "_rawdata" + (ds.raw_ext or ".csv"))
            raw_path.write_bytes(ds.raw_bytes)
        spec = self.state.project.figures.get(ds.id)
        if spec is None:
            run = ds.run_pipeline()
            curves = curves_for(run.final, ds.raw, ds.fit_result)
            spec = default_spec(ds.id, curves, U.AXIS_LABELS.get(run.final.x_unit, run.final.x_label),
                                run.final.y_label)
        code = generate_script(ds, raw_path=str(raw_path), figure_spec=spec,
                               out_stem=str(path.with_name(path.stem + "_results")))
        path.write_text(code, encoding="utf-8")
        self.statusBar().showMessage(f"Script written: {path}", 5000)

    # ================================================================ misc
    def about(self):
        QtWidgets.QMessageBox.about(
            self, "About EZSpec",
            f"<b>EZSpec {__version__}</b><br>Spectral analysis and curve fitting with honest statistics."
            f"<br><br>Numerics backend: {'Rust (ezspec._core)' if HAVE_RUST else 'not compiled'} · active: "
            f"{backend_name()}<br>Reference engine: lmfit/SciPy · baselines validated against pybaselines · fit engine "
            "validated against NIST StRD.")

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if p.endswith(".ezspec"):
                self.open_project(p)
            else:
                self.import_data(p)

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        ev.accept()
