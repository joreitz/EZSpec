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
    dlg = CompareDialog({"Lorentz": ds.fit_result, "Gaussian": r2})
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


def test_templates_roundtrip(app, win, tmp_path):
    from ezspec.project import save_template
    w = win
    ds = w.state.current()
    w.pipeline_panel.add_step("baseline_snip")
    w.state.select_step(None)
    w.plot.peakAdded.emit(1001.0, 90.0)
    pump(app)
    path = tmp_path / "t.ezspec-template.json"
    save_template(ds, path)
    w.load_example("decay")
    pump(app)
    other = w.state.current()
    assert other is not ds and not other.model.components
    w.apply_template(str(path))
    pump(app)
    assert [s.op for s in other.pipeline] == [s.op for s in ds.pipeline]
    assert len(other.model.components) == 1
    w.state.undo.undo()                       # one macro
    assert not other.model.components


ENTRY_SCRIPT = """
import sys
from PySide6 import QtCore, QtWidgets
from ezspec.gui import app as app_module

app = QtWidgets.QApplication(["ezspec"])
seen = {}


def accept_dialog():
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w.__class__.__name__ == "ImportDialog" and w.isVisible():
            w.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.Ok).click()


def finish():
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w.__class__.__name__ == "MainWindow" and w.isVisible():
            seen["n"] = len(w.state.project.datasets)
            w.state.undo.setClean()
            w.close()
    QtWidgets.QApplication.quit()


QtCore.QTimer.singleShot(500, accept_dialog)
QtCore.QTimer.singleShot(1500, finish)
QtCore.QTimer.singleShot(10000, QtWidgets.QApplication.quit)
code = app_module.main(["ezspec", sys.argv[1]])
print("RESULT", code, seen.get("n"))
"""


def test_entry_point_starts_imports_and_quits(tmp_path):
    import subprocess
    import sys

    from ezspec.examples import raman_example, to_csv_bytes
    f = tmp_path / "r.csv"
    f.write_bytes(to_csv_bytes(raman_example()))
    script = tmp_path / "entry.py"
    script.write_text(ENTRY_SCRIPT)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=str(tmp_path))
    res = subprocess.run([sys.executable, str(script), str(f)], capture_output=True, text=True, timeout=120,
                         env=env)
    assert "RESULT 0 1" in res.stdout, res.stdout + res.stderr


def test_combine_dialog_creates_derived_dataset(app, win):
    from ezspec.gui.dialogs import CombineDialog
    from ezspec.project import Dataset
    w = win
    t = np.linspace(0, 10, 101)
    for name, y in (("U", 2 + 0.5 * t), ("I", 0.4 + 0.1 * t)):
        data = ("t,val,sig\n" + "\n".join(f"{a!r},{b!r},0.01" for a, b in zip(t.tolist(), y.tolist()))).encode()
        ds = Dataset.from_bytes(data, f"{name}.csv", y_cols=[1], sigma_col=2)
        ds.name = name
        w.state.add_dataset(ds)
    pump(app)
    dlg = CombineDialog(w.state, w)
    for i in range(dlg.table.rowCount()):        # use only U (alias b) and I (alias c)
        name = dlg.table.item(i, 2).text()
        dlg.table.item(i, 0).setCheckState(QtCore.Qt.Checked if name in ("U", "I") else QtCore.Qt.Unchecked)
    dlg.yexpr.setText("b/c")
    dlg.name.setText("R")
    dlg.update_preview()
    assert dlg.result is not None and dlg.result.sigma is not None, dlg.info.text()
    assert "identical x grid" in dlg.info.text()
    n_before = len(w.state.project.datasets)
    dlg._accept()
    pump(app)
    new = w.state.current()
    assert len(w.state.project.datasets) == n_before + 1 and new.name == "R"
    np.testing.assert_allclose(new.raw.y, (2 + 0.5 * t) / (0.4 + 0.1 * t))
    dlg2 = CombineDialog(w.state, w)
    dlg2.xexpr.setText("nonsense_name")
    dlg2.update_preview()
    assert dlg2.result is None and "unknown names" in dlg2.info.text()


def test_back_and_forth_ramp_is_drawn_in_acquisition_order(app, win):
    from ezspec.project import Dataset
    w = win
    up = np.linspace(10.0, 50.0, 120)
    x = np.concatenate([up, up[::-1], up, up[::-1]])
    y = np.concatenate([np.exp(-0.5 * ((x[:240] - c) / 2) ** 2) for c in (30.0,)] * 2)
    data = ("I,S\n" + "\n".join(f"{a!r},{b!r}" for a, b in zip(x.tolist(), y.tolist()))).encode()
    ds = Dataset.from_bytes(data, "ramp.csv")
    w.state.add_dataset(ds)
    pump(app)
    w.state.set_current(ds.id)
    w.refresh_plot()
    xd, _ = w.plot.c_data.getData()
    np.testing.assert_allclose(xd, x)                   # not re-sorted: no zigzag between the ramps
    assert not w.plot.c_data.opts["clipToView"]
    w.plot.act_points.setChecked(True)
    assert w.plot.c_data.opts["symbol"] == "o"
    w.plot.act_points.setChecked(False)
    w.pipeline_panel.add_step("select_sweeps")
    pump(app)
    w.state.select_step(None)
    w.refresh_plot()
    xd, _ = w.plot.c_data.getData()
    assert len(xd) == 240 and np.sum(np.diff(xd) < 0) == 1    # two up ramps, drawn one after the other


def test_calibration_surface_slices_and_dialog(app, win):
    from ezspec.gui.dialogs import CalibrationDialog
    from ezspec.project import Dataset
    w = win
    rng = np.random.default_rng(0)
    I = np.tile(np.linspace(20, 80, 13), 5)
    T = np.repeat([15.0, 20, 25, 30, 35], 13)
    lam = 1530 + 0.01 * (I - 50) + 0.1 * (T - 25) + rng.normal(scale=0.002, size=I.size)
    rows = "\n".join(f"{a!r},{b!r},{c!r}" for a, b, c in zip(I.tolist(), T.tolist(), lam.tolist()))
    cal = Dataset.from_bytes(("I_mA,T / °C,lambda_nm\n" + rows).encode(), "cal.csv", y_cols=[2],
                             extra_cols={"T": 1})
    cal.name = "Kalibrierung"
    xs = np.linspace(30, 70, 200)
    meas = Dataset.from_bytes(("I_mA,S\n" + "\n".join(f"{a!r},{b!r}" for a, b in zip(xs.tolist(), np.sin(xs).tolist()))
                               ).encode(), "meas.csv")
    meas.name = "Messung"
    w.state.add_dataset(meas)
    w.state.add_dataset(cal)
    pump(app)
    w.state.set_current(cal.id)
    w.refresh_plot()
    assert len(w.plot._slice_items) == 5 and not w.plot.c_data.isVisible()   # one group per T, no zigzag
    assert w.model_panel._extra_variables() == ["T"]
    w.model_panel.add_surface("plane", "T")
    pump(app)
    w.run_fit()
    wait(app, w)
    assert cal.fit_result is not None and cal.fit_result.success
    w.refresh_plot()
    assert len(w.plot._slice_items) == 10                                  # points + slice curve per T
    sources = [cal]
    dlg = CalibrationDialog(w.state, sources, source=cal, target=meas, parent=w)
    assert list(dlg.fixed_edits) == ["T"] and dlg.fixed_edits["T"].value() == 25.0
    targets = dlg.checked_targets()
    assert [t.id for t in targets] == [meas.id]
    assert "within calibrated range" in dlg.slice_info.text()
    dlg._accept()
    w.apply_calibration(dlg.params, dlg.targets)
    pump(app)
    assert meas.pipeline.steps[-1].op == "calibrate_x"
    out = w.state.run(meas).final
    v = cal.fit_result.values
    np.testing.assert_allclose(out.x, v["c0"] + v["cx"] * (xs - 50) + v["cT"] * 0.0, atol=1e-9)
    assert out.x_unit == "nm" or out.x_label
    w.state.undo.undo()
    pump(app)
    assert not meas.pipeline.steps or meas.pipeline.steps[-1].op != "calibrate_x"


def test_figure_of_raw_data_and_single_step(app, win, tmp_path):
    w = win
    ds = w.state.current()
    w.pipeline_panel.add_step("baseline_arpls")
    pump(app)
    sid = w.state.selected_step
    dlg = FigureDialog(ds, None, w, stage=sid)              # what the plot shows for the selected step
    assert dlg.stage == sid and len(dlg.spec["panels"]) == 2
    curves = [t["curve"] for t in dlg.spec["panels"][0]["traces"]]
    assert f"step:{sid}:baseline" in curves
    dlg.stage_combo.setCurrentIndex(dlg.stage_combo.findData("raw"))
    assert dlg.stage == "raw" and [t["curve"] for t in dlg.spec["panels"][0]["traces"]] == ["raw:data"]
    dlg._add_curve(f"step:{sid}:baseline")
    assert dlg.spec["panels"][0]["traces"][-1]["curve"] == f"step:{sid}:baseline"
    dlg.traces.cellWidget(0, 4).setCurrentText("scatter")
    assert dlg.spec["panels"][0]["traces"][0]["kind"] == "scatter"
    dlg.traces.setCurrentCell(1, 1)
    dlg._remove_curve()
    assert len(dlg.spec["panels"][0]["traces"]) == 1
    for _ in range(4):                                     # remove, style, add, content
        dlg.undo.undo()
    assert dlg.stage == sid and dlg.stage_combo.currentData() == sid      # content change is undoable
    out = tmp_path / "step.pdf"
    from ezspec.export.figure import make_resolver, save_figure
    save_figure(dlg.spec, make_resolver(dlg.curves), out)
    assert out.stat().st_size > 1000
    dlg.accept()
    w.state.project.figures[w._figure_key(ds, dlg.stage)] = dlg.spec
    assert f"{ds.id}__{sid}" in w.state.project.figures
