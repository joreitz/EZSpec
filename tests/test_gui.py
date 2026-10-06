"""GUI smoke and interaction tests (offscreen Qt)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore  # noqa: E402

from ezspec.gui.dialogs import CompareDialog, FigureDialog  # noqa: E402
from ezspec.gui.main_window import MainWindow  # noqa: E402
from ezspec.gui.theme import apply_palette, configure_pyqtgraph  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_pyqtgraph()
    apply_palette(a)
    return a


def pump(app, n=8):
    for _ in range(n):
        app.processEvents()
        QtCore.QThread.msleep(10)


def wait(app, w):
    for _ in range(3000):
        pump(app, 1)
        if w._task is None:
            break
    pump(app, 4)


@pytest.fixture
def win(app, tmp_path):
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    w = MainWindow()
    w.show()
    w.load_example("raman")
    pump(app)
    yield w
    w.state.undo.setClean()
    w.close()


def test_pipeline_editing_and_undo(app, win):
    w = win
    ds = w.state.current()
    assert ds is not None and ds.raw.n == 1500
    n0 = len(ds.pipeline)
    w.pipeline_panel.add_step("baseline_arpls")
    pump(app)
    sid = w.state.selected_step
    assert len(ds.pipeline) == n0 + 1 and ds.pipeline.get(sid).op == "baseline_arpls"
    # slider-like successive edits merge into one undo command
    for lam in (1e5, 1e6, 1e7):
        w.pipeline_panel._param_changed(sid, "arPLS", "lam", lam)
    pump(app)
    assert ds.pipeline.get(sid).params["lam"] == 1e7
    w.state.undo.undo()
    pump(app)
    assert ds.pipeline.get(sid).params["lam"] == 1e5          # default restored in one step
    w.state.undo.redo()
    assert ds.pipeline.get(sid).params["lam"] == 1e7
    # region tool adds an exclusion range
    w.plot.rangesEdited.emit("exclude_ranges", [[990.0, 1010.0]])
    pump(app)
    assert ds.pipeline.get(sid).params["exclude_ranges"] == [[990.0, 1010.0]]
    run = w.state.run(ds)
    assert run.ok and "baseline" in run.final.aux


def test_anchor_baseline_from_plot(app, win):
    w = win
    ds = w.state.current()
    w.pipeline_panel.add_step("baseline_anchors")
    pump(app)
    anchors = [[x, None] for x in (420, 700, 950, 1250, 1500, 1780)]
    w.plot.anchorsEdited.emit(anchors)
    pump(app)
    step = ds.pipeline.get(w.state.selected_step)
    assert step.params["anchors"] == anchors
    assert len(w.plot._anchor_items) == 6
    w.plot.anchorsEdited.emit(anchors[:-1])
    pump(app)
    assert len(ds.pipeline.get(w.state.selected_step).params["anchors"]) == 5


def test_peaks_fit_statistics_and_analysis(app, win, tmp_path):
    w = win
    ds = w.state.current()
    w.pipeline_panel.add_step("despike")
    w.pipeline_panel.add_step("baseline_arpls")
    pump(app)
    sid = w.state.selected_step
    w.state.edit("pipeline", lambda p: p.update(sid, {"lam": 1e7}), "lam")
    w.state.select_step(None)
    pump(app)
    w.model_panel.peak_kind.setCurrentIndex(w.model_panel.peak_kind.findData("lorentzian"))
    for x, y in ((1001, 92), (1032, 26), (1157, 20), (1602, 37)):
        w.plot.peakAdded.emit(x, y)
    pump(app)
    assert len(ds.model.peaks) == 4
    # drag the top handle of p1 (non-final preview, then final commit)
    w.plot.peakDragged.emit("p1_", "top", 1000.5, 80.0, False)
    w.plot.peakDragged.emit("p1_", "top", 1000.5, 80.0, True)
    pump(app)
    assert ds.model.component("p1_").settings["center"].value == pytest.approx(1000.5, abs=1e-6)
    w.run_fit()
    wait(app, w)
    r = ds.fit_result
    assert r is not None and r.success and w.state.fit_current(ds)
    assert r.value("p4_center") == pytest.approx(1602, abs=0.5)
    assert w.results_panel.stats.rowCount() > 10
    assert w.model_panel.table.item(0, w.model_panel.C_RES).text()
    # model change makes the fit stale
    w.state.edit("model", lambda m: setattr(m.component("p4_").settings["fwhm"], "max", 100.0), "max")
    pump(app)
    assert not w.state.fit_current(ds)
    w.run_fit()
    wait(app, w)
    w.state.variants.setdefault(ds.id, {})["Lorentz"] = ds.fit_result
    # profile CI in background
    w.run_profile()
    wait(app, w)
    assert "profile_ci" in ds.fit_result.extra
    # comparison dialog builds
    m2 = ds.model.copy()
    from ezspec.fit import fit
    from ezspec.models import Model, add_peak
    g = Model()
    for c in m2.peaks:
        add_peak(g, "gaussian", c.settings["center"].value, 10, 8)
    r2 = fit(w.state.run(ds).final, g, ds.fit_options)
    dlg = CompareDialog({"Lorentz": ds.fit_result, "Gauß": r2})
    assert dlg.null.count() == 2
    # figure dialog renders and edits are undoable
    fd = FigureDialog(ds, None, w)
    fd._set("panels.0.ylabel", "I (a.u.)")
    assert fd.spec["panels"][0]["ylabel"] == "I (a.u.)"
    fd.undo.undo()
    assert fd.spec["panels"][0]["ylabel"] != "I (a.u.)"
    fd.close()
    # save / load project and refit identically
    path = tmp_path / "p.ezspec"
    w.state.project.save(path)
    from ezspec.project import Project
    back = Project.load(path)
    r3 = back.datasets[0].run_fit()
    assert r3.values == ds.fit_result.values


def test_fit_range_and_options(app, win):
    w = win
    ds = w.state.current()
    w.state.select_step(None)
    w.plot.rangeEdited.emit("fit_range", [900.0, 1100.0])
    pump(app)
    assert ds.fit_options.x_range == [900.0, 1100.0]
    w.model_panel.weighting.setCurrentIndex(w.model_panel.weighting.findData("none"))
    pump(app)
    assert ds.fit_options.weighting == "none"
    w.state.undo.undo()
    assert ds.fit_options.weighting == "auto"


@pytest.mark.skipif(not __import__("ezspec").HAVE_RUST, reason="compiled extension not available")
def test_backend_switch_keeps_results_equal(app, win):
    w = win
    ds = w.state.current()
    w.pipeline_panel.add_step("baseline_arpls")
    pump(app)
    a = w.state.run(ds).final.y.copy()
    before = __import__("ezspec").backend_name()
    w.switch_backend("python" if before == "rust" else "rust")
    b = w.state.run(ds).final.y.copy()
    w.switch_backend(before)
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8)


def test_series_dialog(app, win):
    from ezspec.gui.dialogs import SeriesDialog
    w = win
    w.load_example("raman")
    pump(app)
    ds2 = w.state.current()
    first = w.state.project.datasets[0]
    w.state.set_current(first.id)
    w.pipeline_panel.add_step("baseline_arpls")
    pump(app)
    sid = w.state.selected_step
    w.state.edit("pipeline", lambda p: p.update(sid, {"lam": 1e7}), "lam")
    w.state.select_step(None)
    for x, y in ((1001, 92), (1602, 37)):
        w.plot.peakAdded.emit(x, y)
    pump(app)
    dlg = SeriesDialog(w.state, w)
    dlg.run()
    for _ in range(3000):
        pump(app, 1)
        if dlg._task is None:
            break
    assert dlg.series is not None and all(r is not None for r in dlg.series.results)
    assert len(ds2.pipeline) == len(first.pipeline) and ds2.fit_result is not None
    v, e = dlg.series.parameter("p2_center")
    assert np.allclose(v, v[0])          # identical example data -> identical fits
    dlg.mode.setCurrentIndex(2)
    dlg.shared.item(2).setCheckState(QtCore.Qt.Checked)      # p1_fwhm
    dlg.run()
    for _ in range(3000):
        pump(app, 1)
        if dlg._task is None:
            break
    assert dlg.global_result is not None and "p1_fwhm" in dlg.global_result.params
    dlg.close()
