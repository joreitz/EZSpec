"""Main window: wires state, plot, panels, dialogs and background tasks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__, numfmt
from .. import units as U
from .._backend import HAVE_RUST, backend_name, set_backend
from ..export.figure import curves_for, default_spec
from ..export.page import PAGES_KEY, capture, check_page, new_page
from ..export.results import write_all
from ..export.script import generate_script
from ..fit import bootstrap, fit, profile_ci
from ..io import file_dialog_filter
from ..models import add_peak, find_peaks
from ..project import Dataset, Project
from ..view import VIEW_KEY, ViewState
from ..workspace import (WORKSPACE_FILE, SpectraCache, datasets_of_file, load_files, profiles_of, set_profile,
                         spectrum_files, workspace_file)
from . import peak_edit
from .dialogs import (BootstrapDialog, CalibrationDialog, CombineDialog, CompareDialog, FigureDialog,
                      ImportDialog, SeriesDialog)
from .explorer_panel import ExplorerPanel
from .model_panel import ModelPanel
from .page_dialog import PageComposer
from .pipeline_panel import DatasetsPanel, PipelinePanel
from .plot_view import PlotView
from .quick_pick import QuickPick, menu_commands
from .results_panel import ResultsPanel
from .scene import build_view, handle_values
from .shell import ActivityBar, WelcomePage
from .state import AppState, start_task

LAYOUT_VERSION = 3          # bump when docks are added or renamed: older saved layouts are ignored
MAX_RECENT = 10
CONFIRM_OPEN_ABOVE = 100    # opening more files than this at once asks first


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
<li><b>Data</b>: File → Open Folder (Ctrl+Shift+O) shows a folder of spectra in the Explorer: a click
previews a file, a double click or Enter opens it, Ctrl/Shift+click selects several. Files and folders can also be
dropped onto the window. “Open with import options…” (or File → Import data) sets columns and units and can remember
them for all files of that type. Example: File → Example data.</li>
<li><b>Several spectra</b>: tick datasets in the Data list to keep them in the plot while you work on another
one (the selected row is the active dataset; a click on an overlaid curve activates it). “Stack” and “Norm” (normalise)
only change the display. Right-click in the Data list: colour, copy/paste processing and model onto a selection.</li>
<li><b>Zoom regions</b>: tool “Zoom” (key 5) – drag over a part of the spectrum to show it enlarged next to the
full spectrum; drag the band to move it; the × of a panel or a right-click on the band or panel removes it. All tools also work inside the panels.</li>
<li><b>Processing</b>: “+ Step”. The selected step shows its input and output; parameter changes apply live.
Plot tools (keys 1–5): Navigate · Anchor (click to set, drag to move, right-click to delete, Shift = free) ·
Range (masks, forced regions, noise/fit range) · Peak · Zoom.</li>
<li><b>σ</b>: import a σ column or add “Estimate noise → σ” – otherwise σ is treated as unknown (no χ²).</li>
<li><b>Model</b>: Peak tool (click = new peak; drag markers = position/height/width), “Find peaks”,
“+ Component” (background, classic function, custom formula). Table: start value, bounds, vary, expression.</li>
<li><b>Fit</b>: Ctrl+R. Results panel: statistics, warnings, derived quantities, correlations, residuals.
The report always states the baseline. More significant digits: View → Number format.</li>
<li><b>Combine</b>: Ctrl+K – ratio, difference or arbitrary x/y formulas from several datasets
(σ is propagated, alignment is checked automatically).</li>
<li><b>Up/down sweeps</b> (e.g. a current ramp): displayed in acquisition order; “+ Step → Range →
Select sweeps / Average sweeps” separates rising and falling ramps.</li>
<li><b>Calibration</b> (e.g. λ(I, T)): on import, select column T under “Extra variables”,
“+ Component → Surface f(x, v)”, fit, then Analysis → “Apply fit as x calibration”.</li>
<li><b>Go further</b>: profile CI, bootstrap, MCMC, baseline systematics, compare variants, series/global fit (Ctrl+G).</li>
<li><b>Export</b>: publication figure of the current view (Ctrl+E – result, raw data or a selected
processing step), tables, Python script (reproduces everything from the raw data).</li>
<li><b>Pages</b>: “Figure → Add this view to the page” (Ctrl+Shift+E) captures what the plot shows – overlays,
stacking, zoom regions – as frames. In the Pages window (Ctrl+Alt+P) drag, resize and align the frames freely
and export the sheet as one PDF.</li>
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
        try:
            numfmt.set_precision(int(self.settings.value("numfmt/unc_digits", 2)),
                                 int(self.settings.value("numfmt/digits", 6)))
        except (TypeError, ValueError):
            numfmt.set_precision(2, 6)
        self._task = None
        self._range_target = None
        self._cache = SpectraCache()       # parsed files of the Explorer
        self._io_tasks = set()             # file loads running in the background (kept alive here)
        self._preview_request = 0          # only the newest preview request is shown
        self._composer = None              # page composer window (created when first used)
        self.setAcceptDrops(True)

        self.plot = PlotView()
        self.welcome = WelcomePage()
        self.central = QtWidgets.QStackedWidget()      # the welcome page until there is something to plot
        self.central.addWidget(self.welcome)
        self.central.addWidget(self.plot)
        self.setCentralWidget(self.central)
        self._pick = None                  # open quick pick (command palette / go to file)
        self._side_memory = {}             # dock area -> docks that were visible before the side was hidden
        self.explorer = ExplorerPanel()
        self.d_explorer = self._dock("Explorer", self.explorer, QtCore.Qt.LeftDockWidgetArea)
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
        self.resizeDocks([self.d_explorer, self.d_data, self.d_pipe], [250, 130, 420], QtCore.Qt.Vertical)
        self.activity = ActivityBar(self)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.activity)
        for dock, kind, tip in ((self.d_explorer, "explorer", "Explorer – the opened folder"),
                                (self.d_data, "data", "Data – the opened spectra"),
                                (self.d_pipe, "processing", "Processing steps"),
                                (self.d_model, "model", "Model and fit"),
                                (self.d_res, "results", "Results"),
                                (self.d_hist, "history", "History (undo)")):
            self.activity.add_dock(dock, kind, tip + " · click to show or hide")
        self.activity.add_stretch()
        self.activity.add_command("search", "Go to file or dataset (Ctrl+P)", self.quick_open)
        self.activity.add_command("pages", "Pages – arrange several views on a sheet (Ctrl+Alt+P)", self.open_pages)
        self.resizeDocks([self.d_data, self.d_model], [300, 520], QtCore.Qt.Horizontal)
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
        self._update_central()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1500, 920)
        layout = self.settings.value("layout")
        if layout is not None:
            self.restoreState(layout, LAYOUT_VERSION)

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
        f.addAction("Open Folder…", lambda: self.open_folder(), QtGui.QKeySequence("Ctrl+Shift+O"))
        f.addAction("Open project…", self.open_project, QtGui.QKeySequence.Open)
        self.recent_menu = f.addMenu("Open Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.act_close_folder = f.addAction("Close Folder", self.close_folder)
        self.act_close_folder.setEnabled(False)
        f.addSeparator()
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
        exp.addAction("Figure of the current view…", lambda: self.figure_dialog(), QtGui.QKeySequence("Ctrl+E"))
        exp.addAction("Figure of the raw data…", lambda: self.figure_dialog("raw"))
        exp.addAction("Add the current view to the page", self.capture_view, QtGui.QKeySequence("Ctrl+Shift+E"))
        exp.addAction("Pages (arrange several views)…", self.open_pages, QtGui.QKeySequence("Ctrl+Alt+P"))
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
        v.addAction("Command Palette…", self.command_palette, QtGui.QKeySequence("Ctrl+Shift+P"))
        v.addAction("Go to File or Dataset…", self.quick_open, QtGui.QKeySequence("Ctrl+P"))
        v.addSeparator()
        v.addAction("Toggle Side Bar", lambda: self.toggle_side(QtCore.Qt.LeftDockWidgetArea),
                    QtGui.QKeySequence("Ctrl+B"))
        v.addAction("Toggle Right Side Bar", lambda: self.toggle_side(QtCore.Qt.RightDockWidgetArea),
                    QtGui.QKeySequence("Ctrl+Alt+B"))
        v.addSeparator()
        for d in (self.d_explorer, self.d_data, self.d_pipe, self.d_model, self.d_res, self.d_hist):
            v.addAction(d.toggleViewAction())
        v.addSeparator()
        v.addAction("Overlay all datasets", lambda: self.show_all(True), QtGui.QKeySequence("Ctrl+Shift+A"))
        v.addAction("Show only the active dataset", lambda: self.show_all(False),
                    QtGui.QKeySequence("Ctrl+Shift+H"))
        v.addSeparator()
        nf = v.addMenu("Number format")
        nf.setToolTipsVisible(True)
        prec = numfmt.get_precision()
        g1 = QtGui.QActionGroup(self)
        sub = nf.addMenu("Values with uncertainty")
        for n in (1, 2, 3, 4, 6, 8, 0):
            a = QtGui.QAction("no rounding" if n == 0 else f"{n} significant digit{'s' if n > 1 else ''} of the "
                              "uncertainty", self, checkable=True)
            a.setChecked(prec["unc_digits"] == n)
            a.triggered.connect(lambda _=False, n=n: self.set_number_format(unc_digits=n))
            g1.addAction(a)
            sub.addAction(a)
        g2 = QtGui.QActionGroup(self)
        sub2 = nf.addMenu("Values without uncertainty")
        for n in (4, 6, 8, 10, 12, 15, 17):
            a = QtGui.QAction(f"{n} significant digits" + (" (full double precision)" if n == 17 else ""), self,
                              checkable=True)
            a.setChecked(prec["digits"] == n)
            a.triggered.connect(lambda _=False, n=n: self.set_number_format(digits=n))
            g2.addAction(a)
            sub2.addAction(a)
        v.addSeparator()
        for i, key in enumerate(("navigate", "anchor", "region", "peak", "zoom"), 1):
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
        st.projectChanged.connect(self._sync_explorer)
        st.projectChanged.connect(self._update_central)
        wl = self.welcome
        wl.openFolderRequested.connect(lambda: self.open_folder())
        wl.openProjectRequested.connect(lambda: self.open_project())
        wl.importRequested.connect(lambda: self.import_data())
        wl.exampleRequested.connect(lambda: self.load_example("raman"))
        wl.recentRequested.connect(lambda p: self.open_paths([p]))
        wl.helpRequested.connect(lambda: QtWidgets.QMessageBox.information(self, "Quick start", QUICKSTART))
        st.viewChanged.connect(self._view_changed)
        st.message.connect(lambda text, ms: self.statusBar().showMessage(text, ms))
        self.explorer.overlayRequested.connect(lambda paths: self.open_files(paths, show=True))
        self.plot.datasetClicked.connect(self.state.set_current)
        self.plot.viewOptionChanged.connect(self._view_option)
        self.plot.regionsEdited.connect(lambda regions: self.state.set_view(regions=regions))
        self.explorer.openFolderRequested.connect(lambda: self.open_folder())
        self.explorer.previewRequested.connect(self.preview_file)
        self.explorer.openRequested.connect(self.open_files)
        self.explorer.importRequested.connect(self.import_data)
        self.datasets_panel.importRequested.connect(self.import_data)
        self.datasets_panel.exampleRequested.connect(lambda: self.load_example("raman"))
        self.datasets_panel.combineRequested.connect(self.combine_dialog)
        self.pipeline_panel.calibrationRequested.connect(lambda: self.calibration_dialog(from_pipeline=True))
        self.pipeline_panel.rangeTargetChanged.connect(self.plot.set_range_target)
        self.plot.figureRequested.connect(lambda: self.figure_dialog())
        self.plot.captureRequested.connect(self.capture_view)
        self.plot.pagesRequested.connect(self.open_pages)
        st.modifiedChanged.connect(self._update_title)
        for sig in (st.projectChanged, st.viewChanged):
            sig.connect(self._refresh_pages)
        for sig in (st.pipelineChanged, st.modelChanged, st.fitChanged, st.fitOptionsChanged):
            sig.connect(lambda _id: self._refresh_pages())
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

    # ================================================================ shell
    def _update_central(self):
        """The plot as soon as there is something to plot, the welcome page before."""
        empty = not self.state.project.datasets and self.state.preview is None
        if empty:
            self.welcome.set_recent([p for p in self._recent() if Path(p).exists()][:6])
        self.central.setCurrentWidget(self.welcome if empty else self.plot)

    def toggle_side(self, area):
        """Hide the panels on one side of the plot, or bring back the ones that were hidden."""
        docks = [d for d in self.findChildren(QtWidgets.QDockWidget)
                 if self.dockWidgetArea(d) == area and not d.isFloating()]
        shown = [d for d in docks if not d.isHidden()]
        if shown:
            self._side_memory[area] = shown
            for d in shown:
                d.hide()
        else:
            for d in self._side_memory.pop(area, None) or docks:
                d.show()

    def _show_pick(self, placeholder, items):
        self._pick = QuickPick(self, placeholder, items)
        self._pick.show()
        return self._pick

    def command_palette(self):
        """Every command of the menus, found by typing a few letters of its name."""
        items = [(title, shortcut, action.trigger) for title, shortcut, action in menu_commands(self.menuBar())
                 if not title.endswith("Command Palette")]
        # what lives in the plot toolbar and in the context menu of the Data list
        plot, data = self.plot, self.datasets_panel
        items += [("Plot › Auto-range", "", plot.autorange),
                  ("Plot › Stack spectra (on/off)", "", plot.act_stack.toggle),
                  ("Plot › Remove all zoom regions", "", lambda: plot.regionsEdited.emit([]))]
        items += [(f"Plot › Normalise: {a.text()}", "", a.trigger) for a in plot.norm_actions.values()]
        items += [(f"Plot › Show {a.text().lower()} (on/off)" if a is not plot.act_invert else "Plot › Invert x axis",
                   "", a.toggle) for a in (plot.act_components, plot.act_resid, plot.act_points, plot.act_legend,
                                           plot.act_invert)]
        items += [("Data › Copy processing and model of the active dataset", "", data.copy_setup),
                  ("Data › Paste processing onto the selected datasets", "", data.paste_setup),
                  ("Data › Paste processing and model onto the selected datasets", "",
                   lambda: data.paste_setup(model=True)),
                  ("Data › Remove the selected datasets", "", data._remove)]
        return self._show_pick("Type the name of a command", items)

    def quick_open(self):
        """Go to an open dataset or open a file of the folder by typing a part of its name."""
        st = self.state
        items = [(d.name, "open dataset", lambda i=d.id: st.set_current(i)) for d in st.project.datasets]
        root = st.workspace_root
        if root is not None:
            is_open = {str(Path(d.source["path"]).resolve()) for d in st.project.datasets if d.source.get("path")}
            for p in spectrum_files(root, recursive=True)[:5000]:
                if str(p) not in is_open:
                    rel = p.parent.relative_to(root)
                    items.append((p.name, "" if str(rel) == "." else str(rel), lambda p=p: self.open_files([p])))
        return self._show_pick("Go to a dataset or a file of the folder" if root is not None
                               else "Go to a dataset (open a folder to find its files here)", items)

    def set_number_format(self, unc_digits=None, digits=None):
        """Display precision of results, report and tables (exports keep full precision)."""
        numfmt.set_precision(unc_digits, digits)
        prec = numfmt.get_precision()
        self.settings.setValue("numfmt/unc_digits", prec["unc_digits"])
        self.settings.setValue("numfmt/digits", prec["digits"])
        self.model_panel.refresh()
        self.results_panel.refresh()
        self.pipeline_panel.refresh()

    # ================================================================ refresh
    def schedule_refresh(self, *args):
        self._refresh_timer.start()

    def _model_changed(self, _id):
        self.schedule_refresh()
        if self.model_panel.live.isChecked():
            self._live_timer.start()

    def _values_for_handles(self, ds):
        return handle_values(ds, self.state.fit_current(ds))

    def current_scene(self):
        """What the plot shows: the active dataset with the spectra overlaid on it."""
        st = self.state
        ds = st.current()
        if ds is None:
            return build_view(None, None)
        entries = [(d, st.run(d), st.color_of(d)) for d in st.shown_datasets()]
        return build_view(ds, st.run(ds), entries, st.view, st.selected_step, st.fit_current(ds))

    def refresh_plot(self):
        self.plot.show_scene(self.current_scene())

    def _view_changed(self):
        self.plot.set_view_options(self.state.view)
        self.plot.set_regions(self.state.view.regions)
        self.schedule_refresh()

    def _view_option(self, key, value):
        """Stack / normalise switched in the plot toolbar: the y scale changes as a whole."""
        self.state.set_view(**{key: value})
        self.plot.autorange()

    def show_all(self, on=True):
        """Overlay all datasets (or only the active one)."""
        self.state.set_shown([d.id for d in self.state.project.datasets] if on else [], only=True)

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
        """Import one file through the import dialog (columns, units, …)."""
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import data", self.settings.value("last_dir", ""), file_dialog_filter())
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        root = self.state.workspace_root
        in_folder = root is not None and root in Path(path).resolve().parents
        try:
            dlg = ImportDialog(path, self, remember=in_folder)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import", f"Cannot read file:\n{e}")
            return
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        if dlg.remember_profile:
            set_profile(self.state.project, path, dlg.reader, dlg.options)
        if self.state.preview is not None and datasets_of_file([self.state.preview], path):
            self.state.set_preview(None)
        self.state.add_datasets(dlg.datasets, f"Import {Path(path).name}")

    # ================================================================ folder / explorer
    def open_folder(self, path=None):
        """Open a folder as workspace: the Explorer shows its spectra and the work is kept in
        ``workspace.ezspec`` inside it (loaded here if it exists)."""
        if not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder",
                                                              self.settings.value("last_dir", ""))
        if not path:
            return False
        root = Path(path).resolve()
        if not root.is_dir():
            QtWidgets.QMessageBox.warning(self, "Open folder", f"Not a folder:\n{root}")
            return False
        if not self._confirm_discard():
            return False
        proj = Project()
        ws = workspace_file(root)
        if ws.exists():
            try:
                proj = Project.load(ws)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Open folder", f"{ws.name} could not be read and is left "
                                                                   f"untouched:\n{e}")
        self.settings.setValue("last_dir", str(root))
        self._set_workspace(root)
        self.state.new_project(proj)
        self._remember_recent(root)
        self._update_title()
        self._refit_loaded()
        self.d_explorer.show()
        self.d_explorer.raise_()
        return True

    def close_folder(self):
        if self.state.workspace_root is None or not self._confirm_discard():
            return
        self._set_workspace(None)
        self.state.new_project(Project())
        self._update_title()

    def _set_workspace(self, root):
        self.state.workspace_root = root
        self._cache.clear()
        self.explorer.set_root(root)
        self.settings.setValue("session/folder", str(root) if root is not None else "")
        self.act_close_folder.setEnabled(root is not None)

    def restore_session(self):
        """Reopen the folder of the last session (called at program start without arguments)."""
        folder = self.settings.value("session/folder", "")
        if folder and Path(folder).is_dir():
            self.open_folder(folder)

    def _sync_explorer(self):
        paths = set()
        for ds in self.state.project.datasets:
            src = ds.source.get("path")
            if src:
                paths.add(str(Path(src).resolve()))
        self.explorer.set_open_paths(paths)

    def _recent(self) -> list:
        v = self.settings.value("recent", [])
        if isinstance(v, str):               # a one-element list comes back as a string
            v = [v] if v else []
        return [str(x) for x in (v or [])]

    def _remember_recent(self, path):
        path = str(path)
        self.settings.setValue("recent", ([path] + [p for p in self._recent() if p != path])[:MAX_RECENT])
        self._update_central()

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        for p in self._recent():
            path = Path(p)
            if not path.exists():
                continue
            if path.is_dir():
                self.recent_menu.addAction(f"{path.name}  –  {path.parent}", lambda p=p: self.open_folder(p))
            else:
                self.recent_menu.addAction(f"{path.name}  –  {path.parent}", lambda p=p: self.open_project(p))
        if self.recent_menu.isEmpty():
            self.recent_menu.addAction("(nothing opened yet)").setEnabled(False)

    def _load_in_background(self, paths, on_loaded, text=""):
        """Read files in the thread pool (import profile or automatic detection, no dialog)."""
        paths = [Path(p) for p in paths]
        profiles = copy.deepcopy(profiles_of(self.state.project))
        many = len(paths) > 1
        if many:
            self.busy.setRange(0, len(paths))
            self.busy.setValue(0)
            self.busy.setVisible(True)
        if text:
            self.statusBar().showMessage(text)

        def finished():
            self._io_tasks.discard(task)
            if many and self._task is None:
                self.busy.setVisible(False)

        def done(results):
            finished()
            on_loaded(results)

        def failed(msg):
            finished()
            QtWidgets.QMessageBox.warning(self, "Open", msg.split("\n\n")[0])

        def progress(i, n):
            if self._task is None:
                self.busy.setRange(0, n)
                self.busy.setValue(i)
        task = start_task(load_files, done, failed, progress if many else None, paths, profiles, self._cache)
        self._io_tasks.add(task)

    def preview_file(self, path):
        """Show a file in the plot without adding it to the project (single click in the Explorer)."""
        shown = self.state.project.datasets + ([self.state.preview] if self.state.preview is not None else [])
        known = datasets_of_file(shown, path)
        if known:                               # already open or already previewed: just select it
            self._preview_request += 1
            self.state.set_current(known[0].id)
            return
        self._preview_request += 1
        request = self._preview_request

        def loaded(results):
            if request != self._preview_request:
                return                          # the selection has moved on
            res = results[0]
            if not res.ok:
                self.statusBar().showMessage(res.error, 8000)
                return
            self.state.set_preview(res.datasets[0])
            n = len(res.datasets)
            msg = f"Preview: {res.path.name}"
            if n > 1:
                msg += f" – first of {n} spectra in this file"
            self.statusBar().showMessage(res.note or msg + " · double click or Enter opens it", 8000)
        self._load_in_background([path], loaded)

    def open_files(self, paths, show=False):
        """Open files as datasets without a dialog. Files that are already open are selected instead.
        ``show``: overlay them in the plot."""
        paths = [Path(p) for p in paths]
        projects = [p for p in paths if p.suffix.lower() == ".ezspec"]
        if projects:
            self.open_project(str(projects[0]))
            return
        self._preview_request += 1              # a pending preview must not replace what is opened now
        todo, existing = [], []
        for p in paths:
            found = datasets_of_file(self.state.project.datasets, p)
            (existing.extend(found) if found else todo.append(p))
        if show:
            self.state.set_shown([d.id for d in existing], True)
        if not todo:
            if existing:
                self.state.set_current(existing[0].id)
            return
        if len(todo) > CONFIRM_OPEN_ABOVE:
            r = QtWidgets.QMessageBox.question(self, "Open", f"Open {len(todo)} files as datasets?")
            if r != QtWidgets.QMessageBox.Yes:
                return

        def loaded(results):
            ok = [r for r in results if r.ok]
            datasets = [d for r in ok for d in r.datasets]
            if datasets:
                pv = self.state.preview
                if pv is not None and any(datasets_of_file([pv], r.path) for r in ok):
                    self.state.set_preview(None)        # the preview is replaced by the opened file
                title = ok[0].path.name if len(ok) == 1 else f"{len(ok)} files"
                self.state.add_datasets(datasets, f"Open {title}")
                if show:
                    self.state.set_shown([d.id for d in datasets], True)
            notes = [r.note for r in ok if r.note]
            errors = [r.error for r in results if not r.ok]
            self.statusBar().showMessage(notes[0] if notes else f"{len(datasets)} dataset(s) opened.", 6000)
            if errors:
                more = f"\n… and {len(errors) - 12} more" if len(errors) > 12 else ""
                QtWidgets.QMessageBox.warning(self, "Open", "Could not be read:\n" + "\n".join(errors[:12]) + more)
        self._load_in_background(todo, loaded, f"Opening {len(todo)} file(s)…" if len(todo) > 1 else "")

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
        path = Path(path).resolve()
        self.settings.setValue("last_dir", str(path.parent))
        if path.name == WORKSPACE_FILE:         # the workspace of a folder: show the folder as well
            self._set_workspace(path.parent)
            self._remember_recent(path.parent)
        else:
            self._remember_recent(path)
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
        root = self.state.workspace_root
        if not save_as and path is None and root is not None:
            path = workspace_file(root)         # the work on a folder is kept inside the folder
            if path.exists():                   # … but a workspace that is not the loaded one is not ours
                r = QtWidgets.QMessageBox.question(
                    self, "Save", f"{path.name} already exists in this folder and was not loaded.\n"
                                  "Replace it with the current project?")
                if r != QtWidgets.QMessageBox.Yes:
                    return False
        elif save_as or path is None:
            start = root if root is not None else Path(self.settings.value("last_dir", ""))
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", str(start / "project.ezspec"),
                                                            "EZSpec project (*.ezspec)")
            if not path:
                return False
            if not path.endswith(".ezspec"):
                path += ".ezspec"
        view = ViewState.from_dict(self.state.view.to_dict())
        view.prune(d.id for d in proj.datasets)
        proj.meta[VIEW_KEY] = view.to_dict()
        try:
            proj.save(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Save", str(e))
            return False
        self.state.mark_saved()
        self._update_title()
        self.statusBar().showMessage(f"Saved: {path}", 4000)
        return True

    def _confirm_discard(self):
        if not self.state.modified or not self.state.project.datasets:
            return True
        r = QtWidgets.QMessageBox.question(self, "Unsaved changes", "Save changes?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
                                           QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return r == QtWidgets.QMessageBox.Discard

    def _update_title(self):
        p = self.state.project.path
        root = self.state.workspace_root
        if root is not None and (p is None or Path(p) == workspace_file(root)):
            name = root.name or str(root)
        else:
            name = Path(p).name if p else "Untitled"
        try:
            dirty = " *" if self.state.modified else ""
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
                d = self.state.get(ds_id)
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

    # ================================================================ pages
    def pages(self) -> list:
        """Pages of the project: views arranged freely on a sheet (stored in the project file)."""
        pages = self.state.project.meta.setdefault(PAGES_KEY, [])
        if not pages:
            pages.append(new_page())
        for page in pages:
            check_page(page)
        return pages

    def _page_resolver(self, ids):
        st = self.state
        return [(d, st.run(d), st.color_of(d), st.fit_current(d)) for d in st.project.datasets if d.id in ids]

    def _page_names(self, ids) -> str:
        names = []
        for i in ids:
            try:
                names.append(self.state.project.get(i).name)
            except KeyError:
                names.append("(removed)")
        return names[0] + (f" + {len(names) - 1} overlaid" if len(names) > 1 else "")

    def open_pages(self):
        """Show the page composer."""
        if self._composer is None:
            self._composer = PageComposer(self.pages(), self._page_resolver, self._page_names, self)
            self._composer.captureRequested.connect(self.capture_view)
            self._composer.changed.connect(self.state.touch)
        elif self._composer.pages is not self.pages():
            self._composer.set_pages(self.pages(), 0)
        self._composer.show()
        self._composer.raise_()
        return self._composer

    def _refresh_pages(self):
        c = self._composer
        if c is None:
            return
        if c.pages is not self.state.project.meta.get(PAGES_KEY):     # another project was opened
            c.set_pages(self.pages(), 0)
        elif c.isVisible():
            c.refresh()

    def capture_view(self):
        """Add what the plot shows to the current page: the view as a frame, each zoom region as a
        frame next to it."""
        st = self.state
        ds = st.current()
        if ds is None:
            return []
        if st.is_preview(ds):
            st.pin_preview()                 # a page refers to datasets of the project
        composer = self._composer
        pages = self.pages()
        page = composer.page if composer is not None and composer.pages is pages else pages[0]
        x_range, y_range = self.plot.visible_ranges()
        shown = [d.id for d in st.shown_datasets() if d is not ds]
        frames = capture(page, ds.id, shown, st.view, step=st.selected_step, x_range=x_range, y_range=y_range,
                         regions=st.view.regions)
        st.touch()
        composer = self.open_pages() if composer is None or not composer.isVisible() else composer
        if frames[0] not in [it.frame for it in composer.items]:
            composer.add_frames(frames)
        self.statusBar().showMessage(f"Added {len(frames)} frame(s) to “{page['name']}”.", 5000)
        return frames

    # ================================================================ export
    @staticmethod
    def _figure_key(ds, stage):
        return ds.id if stage == "result" else f"{ds.id}__{stage}"

    def figure_dialog(self, stage=None):
        """Publication figure of a stage: 'result', 'raw' or a step id (default: what the plot shows)."""
        ds = self.state.current()
        if ds is None:
            return
        if stage is None:
            stage = self.state.selected_step or "result"
        dlg = FigureDialog(ds, self.state.project.figures.get(self._figure_key(ds, stage)), self, stage=stage)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            if self.state.is_preview(ds):
                self.state.pin_preview()
            self.state.project.figures[self._figure_key(ds, dlg.stage)] = dlg.spec
            self.state.touch()
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
        extra = {}
        steps = [st.id for st in ds.pipeline]
        for key, fspec in self.state.project.figures.items():
            if key.startswith(ds.id + "__"):
                stage = key[len(ds.id) + 2:]
                suffix = "raw" if stage == "raw" else f"step{steps.index(stage) + 1}" if stage in steps else None
                if suffix:
                    extra[suffix] = fspec
        code = generate_script(ds, raw_path=str(raw_path), figure_spec=spec,
                               out_stem=str(path.with_name(path.stem + "_results")), extra_figures=extra)
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
        paths = [url.toLocalFile() for url in ev.mimeData().urls()]
        pos = self.plot.mapFrom(self, ev.position().toPoint())
        on_plot = self.central.currentWidget() is self.plot and self.plot.rect().contains(pos)
        self.open_paths(paths, show=on_plot)             # dropped onto the plot: overlay with what is shown

    def open_paths(self, paths, show=False):
        """Open whatever is given: a folder as workspace, a project, or spectrum files (dropped onto
        the window or passed on the command line). ``show``: overlay the files in the plot."""
        paths = [Path(p) for p in paths if p]
        folders = [p for p in paths if p.is_dir()]
        projects = [p for p in paths if p.suffix.lower() == ".ezspec"]
        if folders:
            self.open_folder(folders[0])
        elif projects:
            self.open_project(str(projects[0]))
        files = [p for p in paths if p.is_file() and p.suffix.lower() != ".ezspec"]
        if files:
            self.open_files(files, show=show)

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("layout", self.saveState(LAYOUT_VERSION))
        if self._composer is not None:
            self._composer.close()
        ev.accept()
