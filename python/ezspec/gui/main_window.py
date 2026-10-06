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
from . import peak_edit
from .dialogs import BootstrapDialog, CombineDialog, CompareDialog, FigureDialog, ImportDialog, SeriesDialog
from .model_panel import ModelPanel
from .pipeline_panel import DatasetsPanel, PipelinePanel
from .plot_view import PlotView
from .results_panel import ResultsPanel
from .state import AppState, start_task

STATS_HELP = """<h3>Wie EZSpec Fit-Güte berichtet</h3>
<ul>
<li><b>σ-Quelle</b> ist Pflichtangabe: bekannt (Spalte), geschätzt (DER_SNR / flacher Bereich), Poisson (σ² = Modell)
oder unbekannt. Nur mit bekanntem σ ist χ² ein χ²; sonst werden RSS, s = √(RSS/ν) und RMSE angezeigt.</li>
<li><b>Kovarianz</b>: bei bekanntem σ <i>absolut</i>. Skalieren mit √χ²_ν verdeckt sonst eine Fehlanpassung hinter
größeren Fehlerbalken – es ist möglich, wird aber markiert. Bei unbekanntem σ ist Skalieren nötig.</li>
<li><b>χ²_ν</b> wird mit seinem Erwartungsband 1 ± √(2/ν) gezeigt; auch bei korrektem Modell streut es.</li>
<li><b>R²</b> ist nur deskriptiv. Für nichtlineare Modelle wählt es das wahre Modell schlecht aus
(Spiess &amp; Neumeyer 2010); zum Modellvergleich ΔAICc/ΔBIC auf identischen Daten verwenden.</li>
<li><b>Parameter am Bound</b>, <b>|ρ| &gt; 0,9</b> und rangdefiziente Jacobi-Matrizen werden markiert;
Standardfehler sind dann nicht belastbar → Profil-CI oder Bootstrap.</li>
<li><b>Geglättete oder interpolierte Daten</b> nicht fitten: korreliertes Rauschen lässt Fehler stark
unterschätzen (O'Haver). Glätten in EZSpec dient standardmäßig nur der Anzeige.</li>
<li><b>„Ein Peak mehr“</b> nicht per F-Test entscheiden (Randproblem, Protassov et al. 2002).</li>
<li><b>Poisson-Daten</b> nicht mit σ = √y gewichten (verzerrt, Humphrey et al. 2009); Gewichtung „Poisson:
σ² = Modell“ liefert den Poisson-Maximum-Likelihood-Schätzer.</li>
<li>Die numerische Genauigkeit der Fit-Engine ist gegen alle 27 NIST-StRD-Probleme geprüft
(≥ 7 Stellen in den Parametern).</li>
</ul>"""


QUICKSTART = """<h3>Kurzanleitung</h3>
<ol>
<li><b>Daten</b>: Importieren (Text/CSV, JCAMP-DX) oder Datei ins Fenster ziehen. Beispiel: Datei → Beispieldaten.</li>
<li><b>Verarbeitung</b>: „+ Schritt“. Gewählter Schritt zeigt Eingang und Ergebnis; Parameter wirken live.
Werkzeuge im Plot (Tasten 1–4): Navigieren · Anker (Klick setzen, ziehen, Rechtsklick löschen, Shift = frei) ·
Bereich (Masken, Pflichtbereiche, Rausch-/Fitbereich) · Peak.</li>
<li><b>σ</b>: σ-Spalte importieren oder „Rauschen schätzen → σ“ – sonst gilt σ als unbekannt (kein χ²).</li>
<li><b>Modell</b>: Werkzeug Peak (Klick = neuer Peak; Marker ziehen = Lage/Höhe/Breite), „Peaks finden“,
„+ Komponente“ (Untergrund, klassische Funktion, eigene Formel). Tabelle: Start, Grenzen, frei, Ausdruck.</li>
<li><b>Fit</b>: Strg+R. Ergebnis-Panel: Statistik, Warnungen, abgeleitete Größen, Korrelationen, Residuen.</li>
<li><b>Verrechnen</b>: Strg+K – Quotient, Differenz oder beliebige x/y-Formeln aus mehreren Datensätzen
(σ wird fortgepflanzt, Ausrichtung automatisch geprüft).</li>
<li><b>Vertiefen</b>: Profil-CI, Bootstrap, MCMC, Baseline-Systematik, Varianten vergleichen, Serie/global (Strg+G).</li>
<li><b>Export</b>: Abbildung (Strg+E), Tabellen, Python-Skript (reproduziert alles aus den Rohdaten).</li>
</ol>
<p>Alles ist rückgängig machbar (Strg+Z); der Verlauf steht im Dock „Verlauf“.</p>"""


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
        self.d_data = self._dock("Daten", self.datasets_panel, QtCore.Qt.LeftDockWidgetArea)
        self.d_pipe = self._dock("Verarbeitung", self.pipeline_panel, QtCore.Qt.LeftDockWidgetArea)
        self.d_model = self._dock("Modell und Fit", self.model_panel, QtCore.Qt.RightDockWidgetArea)
        self.d_res = self._dock("Ergebnis", self.results_panel, QtCore.Qt.RightDockWidgetArea)
        undo_view = QtWidgets.QUndoView(self.state.undo)
        undo_view.setEmptyLabel("<Projektbeginn>")
        self.d_hist = self._dock("Verlauf", undo_view, QtCore.Qt.LeftDockWidgetArea)
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
        f = mb.addMenu("&Datei")
        f.addAction("Neues Projekt", self.new_project, QtGui.QKeySequence.New)
        f.addAction("Projekt öffnen…", self.open_project, QtGui.QKeySequence.Open)
        f.addAction("Speichern", self.save_project, QtGui.QKeySequence.Save)
        f.addAction("Speichern unter…", lambda: self.save_project(True), QtGui.QKeySequence.SaveAs)
        f.addSeparator()
        f.addAction("Daten importieren…", self.import_data, QtGui.QKeySequence("Ctrl+I"))
        ex = f.addMenu("Beispieldaten")
        ex.addAction("Raman-Spektrum mit Fluoreszenzuntergrund", lambda: self.load_example("raman"))
        ex.addAction("Abklingkurve (Poisson-Zählraten)", lambda: self.load_example("decay"))
        f.addSeparator()
        f.addAction("Vorlage speichern…", self.save_template)
        f.addAction("Vorlage anwenden…", self.apply_template)
        f.addSeparator()
        exp = f.addMenu("Exportieren")
        exp.addAction("Abbildung…", self.figure_dialog, QtGui.QKeySequence("Ctrl+E"))
        exp.addAction("Fit-Ergebnisse (CSV/JSON/Bericht)…", self.export_results)
        exp.addAction("Python-Skript (reproduzierbar)…", self.export_script)
        f.addSeparator()
        f.addAction("Beenden", self.close, QtGui.QKeySequence.Quit)

        e = mb.addMenu("&Bearbeiten")
        undo = self.state.undo.createUndoAction(self, "Rückgängig")
        undo.setShortcut(QtGui.QKeySequence.Undo)
        redo = self.state.undo.createRedoAction(self, "Wiederholen")
        redo.setShortcuts([QtGui.QKeySequence.Redo, QtGui.QKeySequence("Ctrl+Y")])
        e.addAction(undo)
        e.addAction(redo)

        a = mb.addMenu("&Analyse")
        a.addAction("Fit ausführen", self.run_fit, QtGui.QKeySequence("Ctrl+R"))
        a.addAction("Peaks automatisch finden", self.auto_peaks)
        a.addAction("Profil-Konfidenzintervalle", self.run_profile)
        a.addAction("Bootstrap…", self.run_bootstrap)
        a.addAction("Baseline-Systematik (λ/Fenster variieren)", self.run_systematics)
        a.addAction("MCMC-Posterior (emcee)…", self.run_mcmc)
        a.addSeparator()
        a.addAction("Daten verrechnen (Quotient, Differenz, x/y-Formeln)…", self.combine_dialog,
                    QtGui.QKeySequence("Ctrl+K"))
        a.addAction("Serie / globaler Fit…", self.series_dialog, QtGui.QKeySequence("Ctrl+G"))
        a.addSeparator()
        a.addAction("Fit als Variante merken", self.remember_variant)
        a.addAction("Modelle vergleichen…", self.compare_variants)
        a.addSeparator()
        bk = a.addMenu("Numerik-Kern")
        grp = QtGui.QActionGroup(self)
        for key, title in (("rust", "Rust (schnell)"), ("python", "Python/SciPy (Referenz)")):
            act = QtGui.QAction(title, self, checkable=True)
            act.setChecked(backend_name() == key)
            act.setEnabled(key == "python" or HAVE_RUST)
            act.triggered.connect(lambda _=False, k=key: self.switch_backend(k))
            grp.addAction(act)
            bk.addAction(act)

        v = mb.addMenu("&Ansicht")
        for d in (self.d_data, self.d_pipe, self.d_model, self.d_res, self.d_hist):
            v.addAction(d.toggleViewAction())
        v.addSeparator()
        for i, key in enumerate(("navigate", "anchor", "region", "peak"), 1):
            act = v.addAction(f"Werkzeug: {self.plot.mode_actions[key].text()}",
                              lambda k=key: self.plot.set_mode(k))
            act.setShortcut(QtGui.QKeySequence(str(i)))

        h = mb.addMenu("&Hilfe")
        h.addAction("Kurzanleitung", lambda: QtWidgets.QMessageBox.information(self, "Kurzanleitung", QUICKSTART),
                    QtGui.QKeySequence.HelpContents)
        h.addAction("Statistik-Hinweise", lambda: QtWidgets.QMessageBox.information(self, "Statistik", STATS_HELP))
        h.addAction("Über EZSpec", self.about)

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
                self.plot.set_scene(data=(inp.x, inp.y), **kw)
            elif step.op.startswith("baseline_"):
                corrected = (out.x, out.y) if step.params.get("subtract", True) else (out.x, out.y - out.aux["baseline"])
                self.plot.set_scene(data=(inp.x, inp.y), baseline=(out.x, out.aux["baseline"]),
                                    residuals=(corrected[0], corrected[1], False), **kw)
                self.plot.p_res.setLabel("left", "korrigiert")
                if step.op == "baseline_anchors":
                    try:
                        anchors_eff = anchor_values(inp, step.params.get("anchors", []), step.params.get("window", 3))
                    except Exception:  # noqa: BLE001
                        anchors_eff = None
            elif "smoothed" in out.aux and "smoothed" not in inp.aux:
                self.plot.set_scene(data=(out.x, out.y), smoothed=(out.x, out.aux["smoothed"]), **kw)
            elif step.op == "estimate_noise" or step.op == "exclude":
                excl = out.exclude if out.exclude is not None else np.zeros(out.n, bool)
                self.plot.set_scene(data=(out.x, out.y), excluded=(out.x[excl], out.y[excl]), **kw)
            else:
                self.plot.set_scene(data=(out.x, out.y), input=(inp.x, inp.y), **kw)
            self.plot.set_tools(step=step, anchors_eff=anchors_eff)
            return

        s = run.final
        xl = U.PLAIN_LABELS.get(s.x_unit, s.x_label)
        self.plot.p_res.setLabel("left", "Res.")
        mask = s.fit_mask
        excluded = (s.x[~mask], s.y[~mask]) if (~mask).any() else None
        smoothed = (s.x, s.aux["smoothed"]) if "smoothed" in s.aux else None
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
            resid = (r.x, r.normalized_residuals, r.stats.chi2 is not None)
        if ds.model.components and (r is None or not current) and s.n > 1:
            try:
                xd = np.linspace(s.x.min(), s.x.max(), 1500)
                preview = (xd, ds.model.evaluate(xd, ds.model.initial_values()))
            except Exception:  # noqa: BLE001 - invalid model: no preview
                preview = None
        self.plot.set_scene(data=(s.x, s.y), excluded=excluded, smoothed=smoothed, fit=fit_xy,
                            fit_stale=not current, components=comps or (), preview=preview, residuals=resid,
                            x_label=xl, y_label=s.y_label, scatter=s.meta.get("plot_style") == "scatter")
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
        self._step_edit({"anchors": anchors}, "Anker bearbeiten")

    def _ranges_edited(self, param, lst):
        self._step_edit({param: [[min(a, b), max(a, b)] for a, b in lst]}, f"Bereiche: {param}")

    def _range_edited(self, param, ab):
        a, b = ab
        lo = None if a is None or b is None else min(a, b)
        hi = None if a is None or b is None else max(a, b)
        if param == "crop":
            self._step_edit({"xmin": lo, "xmax": hi}, "Bereich beschneiden")
        elif param == "fit_range":
            self.state.edit("options", lambda o: setattr(o, "x_range", None if lo is None else [lo, hi]),
                            "Fitbereich")
        else:
            self._step_edit({param: [lo, hi]}, f"Bereich: {param}")

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
        self.state.edit("model", lambda m: add_peak(m, kind, x, h, fw), f"Peak bei {x:.5g}")

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
        self.state.edit("model", mut, f"Peak {prefix.rstrip('_')} verschieben")

    # ================================================================ data / project
    def import_data(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Daten importieren", self.settings.value("last_dir", ""),
                "Spektren (*.txt *.csv *.dat *.tsv *.xy *.asc *.prn *.jdx *.dx *.jcm);;Alle Dateien (*)")
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        try:
            dlg = ImportDialog(path, self)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import", f"Datei kann nicht gelesen werden:\n{e}")
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
        name = "beispiel_raman.csv" if which == "raman" else "beispiel_abklingkurve.csv"
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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Vorlage speichern", f"{ds.name}.ezspec-template.json",
                                                        "EZSpec-Vorlage (*.json)")
        if path:
            save_template(ds, path)
            self.statusBar().showMessage(f"Vorlage gespeichert: {path}", 4000)

    def apply_template(self, path=None):
        from ..project import load_template
        ds = self.state.current()
        if ds is None:
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Vorlage anwenden", "", "EZSpec-Vorlage (*.json)")
        if not path:
            return
        try:
            t = load_template(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Vorlage", str(e))
            return
        from ..fit import FitOptions
        from ..models import Model
        from ..pipeline import Pipeline
        self.state.undo.beginMacro("Vorlage anwenden")
        self.state.edit("pipeline", lambda p: p.steps.__setitem__(slice(None), Pipeline.from_dict(t["pipeline"]).steps),
                        "Pipeline aus Vorlage")
        self.state.edit("model", lambda m: m.components.__setitem__(slice(None), Model.from_dict(t["model"]).components),
                        "Modell aus Vorlage")
        self.state.edit("options", lambda o: o.__dict__.update(FitOptions.from_dict(t["fit_options"]).to_dict()),
                        "Fit-Optionen aus Vorlage")
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
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Projekt öffnen", self.settings.value("last_dir", ""),
                                                            "EZSpec-Projekt (*.ezspec)")
        if not path:
            return
        try:
            proj = Project.load(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Öffnen", str(e))
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
                    self.statusBar().showMessage(f"Fit von {ds.name} nicht reproduzierbar: {e}", 8000)

    def save_project(self, save_as=False):
        proj = self.state.project
        path = proj.path
        if save_as or path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Projekt speichern",
                                                            str(Path(self.settings.value("last_dir", "")) /
                                                                "projekt.ezspec"), "EZSpec-Projekt (*.ezspec)")
            if not path:
                return False
            if not path.endswith(".ezspec"):
                path += ".ezspec"
        try:
            proj.save(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Speichern", str(e))
            return False
        self.state.undo.setClean()
        self._update_title()
        self.statusBar().showMessage(f"Gespeichert: {path}", 4000)
        return True

    def _confirm_discard(self):
        if self.state.undo.isClean() or not self.state.project.datasets:
            return True
        r = QtWidgets.QMessageBox.question(self, "Ungespeicherte Änderungen", "Änderungen speichern?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
                                           QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return r == QtWidgets.QMessageBox.Discard

    def _update_title(self):
        p = self.state.project.path
        name = Path(p).name if p else "Unbenannt"
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
            self.statusBar().showMessage("Zuerst Modellkomponenten hinzufügen (z. B. Werkzeug 'Peak').", 5000)
            return
        if self._task is not None:
            self._live_timer.start()
            return
        s = self.state.run(ds).final
        model = ds.model.copy()
        options = type(ds.fit_options).from_dict(ds.fit_options.to_dict())
        self._busy(True, "Fit läuft …")
        ds_id = ds.id

        def done(result):
            self._task = None
            self._busy(False, "Fit fertig.")
            try:
                d = self.state.project.get(ds_id)
            except KeyError:
                return
            d.fit_result = result
            d.fit_record = result.to_dict(include_curves=False)
            self.state.fitChanged.emit(ds_id)

        def failed(msg):
            self._task = None
            self._busy(False, "Fit fehlgeschlagen.")
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
            self.statusBar().showMessage("Keine Peaks über 5·Rauschen gefunden.", 5000)
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
        n = self.state.edit("model", mut, "Peaks automatisch")
        self.statusBar().showMessage(f"{n or 0} Peak(s) hinzugefügt.", 4000)

    def _require_fit(self):
        ds = self.state.current()
        if ds is None or ds.fit_result is None:
            QtWidgets.QMessageBox.information(self, "Analyse", "Zuerst einen Fit ausführen.")
            return None
        return ds

    def run_profile(self):
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        self._busy(True, "Profil-Konfidenzintervalle …")

        def done(_res):
            self._task = None
            self._busy(False, "Profil-CI fertig.")
            self.state.fitChanged.emit(ds.id)
            self.results_panel.tabs.setCurrentIndex(4)

        def failed(msg):
            self._task = None
            self._busy(False)
            QtWidgets.QMessageBox.warning(self, "Profil-CI", msg.split("\n\n")[0])
        self._task = start_task(profile_ci, done, failed, None, ds.fit_result)

    def run_bootstrap(self):
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        dlg = BootstrapDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._busy(True, "Bootstrap …")

        def done(_res):
            self._task = None
            self._busy(False, "Bootstrap fertig.")
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
            QtWidgets.QMessageBox.information(self, "Verrechnen", "Zuerst Daten importieren.")
            return
        CombineDialog(self.state, self).exec()

    def series_dialog(self):
        if len(self.state.project.datasets) < 2:
            QtWidgets.QMessageBox.information(self, "Serie", "Mindestens zwei Datensätze importieren.")
            return
        SeriesDialog(self.state, self).exec()

    def run_systematics(self):
        from ..fit import baseline_systematics
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        self._busy(True, "Baseline-Systematik: Varianten werden gefittet …")
        result = ds.fit_result

        def done(res):
            self._task = None
            self._busy(False, "Baseline-Systematik fertig.")
            result.extra["baseline_systematics"] = res
            self.state.fitChanged.emit(ds.id)
            self.results_panel.tabs.setCurrentIndex(4)

        def failed(msg):
            self._task = None
            self._busy(False)
            QtWidgets.QMessageBox.warning(self, "Baseline-Systematik", msg.split("\n\n")[0])
        self._task = start_task(baseline_systematics, done, failed, None, ds.raw, ds.pipeline.copy(),
                                ds.model.copy(), type(ds.fit_options).from_dict(ds.fit_options.to_dict()))

    def run_mcmc(self):
        from ..fit import mcmc
        ds = self._require_fit()
        if ds is None or self._task is not None:
            return
        steps, ok = QtWidgets.QInputDialog.getInt(self, "MCMC", "Schritte pro Walker:", 3000, 200, 200000, 500)
        if not ok:
            return
        self._busy(True, "MCMC läuft …")
        result = ds.fit_result

        def done(_res):
            self._task = None
            self._busy(False, "MCMC fertig.")
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
        default = " + ".join(c.type.title if c.kind != "formula" else "Formel" for c in ds.model.components)
        name, ok = QtWidgets.QInputDialog.getText(self, "Variante merken", "Name:", text=default)
        if ok and name:
            variants[name] = ds.fit_result
            self.statusBar().showMessage(f"Variante „{name}“ gespeichert ({len(variants)} insgesamt).", 4000)

    def compare_variants(self):
        ds = self.state.current()
        variants = dict(self.state.variants.get(ds.id, {})) if ds else {}
        if ds is not None and ds.fit_result is not None and ds.fit_result not in variants.values():
            variants["aktueller Fit"] = ds.fit_result
        if len(variants) < 2:
            QtWidgets.QMessageBox.information(self, "Vergleich", "Mindestens zwei Fits nötig: Fit ausführen, "
                                                                 "„Variante merken“, Modell ändern, erneut fitten.")
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
        self.backend_label.setText("Kern: Rust" if name == "rust" else "Kern: Python (Referenz)")

    # ================================================================ export
    def figure_dialog(self):
        ds = self.state.current()
        if ds is None:
            return
        key = ds.id
        dlg = FigureDialog(ds, self.state.project.figures.get(key), self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.state.project.figures[key] = dlg.spec
            self.statusBar().showMessage("Abbildung im Projekt gespeichert.", 3000)

    def export_results(self):
        ds = self._require_fit()
        if ds is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Ergebnisse exportieren (Dateistamm)",
                                                        f"{ds.name}_fit", "Alle (*)")
        if not path:
            return
        files = write_all(ds.fit_result, path)
        self.statusBar().showMessage(f"{len(files)} Dateien geschrieben.", 4000)

    def export_script(self):
        ds = self.state.current()
        if ds is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Python-Skript exportieren",
                                                        f"{Path(ds.name).stem}_analyse.py", "Python (*.py)")
        if not path:
            return
        path = Path(path)
        src = ds.source
        raw_path = src.get("path")
        if not raw_path or not Path(raw_path).exists():
            if ds.raw_bytes is None:
                QtWidgets.QMessageBox.warning(self, "Skript", "Keine Rohdatendatei verfügbar.")
                return
            raw_path = path.with_name(path.stem + "_rohdaten" + (ds.raw_ext or ".csv"))
            raw_path.write_bytes(ds.raw_bytes)
        spec = self.state.project.figures.get(ds.id)
        if spec is None:
            run = ds.run_pipeline()
            curves = curves_for(run.final, ds.raw, ds.fit_result)
            spec = default_spec(ds.id, curves, U.AXIS_LABELS.get(run.final.x_unit, run.final.x_label),
                                run.final.y_label)
        code = generate_script(ds, raw_path=str(raw_path), figure_spec=spec,
                               out_stem=str(path.with_name(path.stem + "_ergebnis")))
        path.write_text(code, encoding="utf-8")
        self.statusBar().showMessage(f"Skript geschrieben: {path}", 5000)

    # ================================================================ misc
    def about(self):
        QtWidgets.QMessageBox.about(
            self, "Über EZSpec",
            f"<b>EZSpec {__version__}</b><br>Spektrenauswertung und Kurvenanpassung mit ehrlicher Statistik."
            f"<br><br>Numerik-Kern: {'Rust (ezspec._core)' if HAVE_RUST else 'nicht kompiliert'} · aktiv: "
            f"{backend_name()}<br>Referenz-Engine: lmfit/SciPy · Baselines geprüft gegen pybaselines · Fit-Engine "
            "geprüft gegen NIST StRD.")

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
