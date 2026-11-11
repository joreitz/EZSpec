"""Reader registry, folder workspaces, import profiles and the Explorer."""

import os

import numpy as np
import pytest

from ezspec.examples import raman_example, to_csv_bytes
from ezspec.io import (Reader, file_dialog_filter, find_reader, read_file, read_spectra, register_reader,
                       source_meta, supported_extensions, unregister_reader)
from ezspec.project import Dataset, Project
from ezspec.spectrum import Spectrum
from ezspec.workspace import (WORKSPACE_FILE, SpectraCache, datasets_of_file, load_file, load_files, profiles_of,
                              set_profile, spectrum_files)


def write_table(path, n_y=1, n=51, sep=",", shift=0.0):
    x = np.linspace(100.0, 200.0, n)
    cols = [x] + [np.exp(-0.5 * ((x - 150.0 - shift) / 5.0) ** 2) * (k + 1) for k in range(n_y)]
    header = sep.join(["x"] + [f"y{k + 1}" for k in range(n_y)])
    rows = [sep.join(f"{v:.8g}" for v in row) for row in np.column_stack(cols)]
    path.write_text("\n".join([header] + rows) + "\n")
    return path


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "measurements"
    (d / "sub").mkdir(parents=True)
    for name in ("T10.csv", "T2.csv", "T1.csv"):
        write_table(d / name)
    write_table(d / "multi.csv", n_y=3)
    (d / "sub" / "raman.csv").write_bytes(to_csv_bytes(raman_example()))
    (d / "notes.md").write_text("not a spectrum")
    return d


# ============================================================================ reader registry
@pytest.fixture
def myformat():
    """A third-party format: two spectra per file, recognised by extension or magic bytes."""
    def read(path, raw, scale=1.0):
        vals = np.array([float(t) for t in raw.decode().split()[1:]])
        x = np.arange(len(vals), dtype=float)
        out = []
        for k in (1, 2):
            meta = {"name": f"{path.stem}-{k}", "source": source_meta(path, raw, "myformat", {"scale": scale})}
            out.append(Spectrum(x, vals * scale * k, x_unit="nm", meta=meta))
        return out

    register_reader(Reader("myformat", "My instrument", ("myf",), read,
                           detect=lambda path, raw: raw.startswith(b"MYF1"), options=("scale",)))
    yield
    unregister_reader("myformat")


def test_registered_reader_is_used_everywhere(tmp_path, myformat):
    f = tmp_path / "a.myf"
    f.write_text("MYF1 1 2 3 4")
    assert ".myf" in supported_extensions() and "*.myf" in file_dialog_filter()
    specs = read_spectra(f, scale=2.0)
    assert [s.meta["name"] for s in specs] == ["a-1", "a-2"]
    assert np.allclose(specs[1].y, [4, 8, 12, 16])
    assert [s.meta["source"]["index"] for s in specs] == [0, 1]      # position within the file is recorded
    # recognised by content when the extension says nothing
    g = tmp_path / "b.bin"
    g.write_text("MYF1 5 6")
    assert find_reader(g, g.read_bytes()).name == "myformat"
    assert read_file(g, index=1).meta["name"] == "b-2"
    # options of another format are ignored (one profile for a mixed folder), unknown ones are an error
    assert len(read_spectra(f, x_col=0, scale=1.0)) == 2
    with pytest.raises(TypeError):
        read_spectra(f, no_such_option=1)
    # the reproducible script names the reader, its options and the spectrum
    from ezspec.export.script import _reader_call
    assert _reader_call(specs[1]) == "read_file(RAW, index=1, reader='myformat', scale=2.0)"
    datasets = Dataset.all_from_file(f)
    assert len(datasets) == 2 and datasets[0].raw_bytes == f.read_bytes() and datasets[0].raw_ext == ".myf"


def test_readers_from_installed_packages(tmp_path, monkeypatch):
    """Entry points of the group ``ezspec.readers`` are registered on first use; a broken one only warns."""
    import importlib.metadata

    from ezspec.io import registry

    def read(path, raw):
        return [Spectrum([0.0, 1.0], [1.0, 2.0], meta={"name": "plugin"})]

    class EP:
        def __init__(self, name, obj):
            self.name, self.obj = name, obj

        def load(self):
            if isinstance(self.obj, Exception):
                raise self.obj
            return self.obj

    eps = [EP("good", lambda: [Reader("plug", "Plugin format", (".plg",), read)]), EP("bad", ImportError("boom"))]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps if group == "ezspec.readers" else [])
    try:
        with pytest.warns(UserWarning, match="bad"):
            registry.load_plugins(force=True)
        f = tmp_path / "x.plg"
        f.write_bytes(b"\x00\x01")
        assert read_file(f).meta["name"] == "plugin" and ".plg" in supported_extensions()
    finally:
        unregister_reader("plug")


def test_builtin_formats_unchanged(tmp_path):
    f = write_table(tmp_path / "t.csv", n_y=2)
    assert find_reader(f).name == "text"
    assert [s.meta["name"] for s in read_spectra(f)] == ["y1", "y2"]
    j = tmp_path / "noext"
    j.write_text("##TITLE=demo\n##JCAMP-DX=4.24\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##XFACTOR=1\n##YFACTOR=1\n"
                 "##FIRSTX=1\n##LASTX=3\n##NPOINTS=3\n##XYPOINTS=(XY..XY)\n1,10\n2,20\n3,30\n##END=\n")
    assert find_reader(j, j.read_bytes()).name == "jcamp"
    assert read_file(j).n == 3


# ============================================================================ workspace (no Qt)
def test_spectrum_files_natural_order(folder):
    assert [p.name for p in spectrum_files(folder)] == ["multi.csv", "T1.csv", "T2.csv", "T10.csv"]
    assert [p.name for p in spectrum_files(folder, recursive=True)][:2] == ["raman.csv", "multi.csv"]  # folders first


def test_load_with_profile_fallback_and_cache(folder):
    proj = Project()
    cache = SpectraCache()
    res = load_file(folder / "multi.csv", profiles_of(proj), cache)
    assert res.ok and [d.name for d in res.datasets] == ["y1", "y2", "y3"] and not res.note
    # a profile for *.csv: only the third y column, x in nm
    set_profile(proj, folder / "multi.csv", "text", {"x_col": 0, "y_cols": [3], "x_unit": "nm", "bogus": 1})
    assert profiles_of(proj)[".csv"] == {"reader": "text", "options": {"x_col": 0, "y_cols": [3], "x_unit": "nm"}}
    res = load_file(folder / "multi.csv", profiles_of(proj), cache)
    assert [d.name for d in res.datasets] == ["multi"] and res.datasets[0].raw.x_unit == "nm"
    assert np.isclose(res.datasets[0].raw.y.max(), 3.0, rtol=1e-3)
    # the profile does not fit a two-column file: automatic detection, and the user is told
    res = load_file(folder / "T1.csv", profiles_of(proj), cache)
    assert res.ok and len(res.datasets) == 1 and "do not fit" in res.note
    # cached: same parsed spectrum, but a fresh dataset every time
    again = load_file(folder / "T1.csv", profiles_of(proj), cache)
    assert again.datasets[0].raw is res.datasets[0].raw and again.datasets[0] is not res.datasets[0]
    # a file that changed on disk is read again
    write_table(folder / "T1.csv", n=80)
    os.utime(folder / "T1.csv", ns=(1, 1))
    assert load_file(folder / "T1.csv", profiles_of(proj), cache).datasets[0].raw.n == 80
    # unreadable files are reported, the others still load
    (folder / "bad.csv").write_text("no numbers here\n")
    out = load_files([folder / "bad.csv", folder / "T2.csv"], {}, cache)
    assert not out[0].ok and "bad.csv" in out[0].error and out[1].ok
    assert datasets_of_file(out[1].datasets, folder / "T2.csv") == out[1].datasets


def test_scene_is_built_without_qt():
    pytest.importorskip("PySide6")
    from ezspec import add_peak
    from ezspec.gui.scene import build_scene
    ds = Dataset.from_bytes(to_csv_bytes(raman_example()), "r.csv")
    step = ds.pipeline.add("baseline_arpls")
    run = ds.run_pipeline()
    sc = build_scene(ds, run, step.id)
    assert sc.step is step and sc.baseline is not None and sc.residual_label == "corrected"
    assert len(sc.data[0]) == ds.raw.n and sc.fit is None and not sc.peaks
    add_peak(ds.model, "lorentzian", 1001.0, 90.0, 8.0)
    sc = build_scene(ds, run, None, fit_is_current=False)
    assert sc.step is None and sc.preview is not None and sc.fit is None and sc.fit_stale
    assert len(sc.peaks) == 1 and sc.peaks[0]["prefix"] == "p1_"
    ds.run_fit()
    sc = build_scene(ds, run, None, fit_is_current=True)
    assert sc.fit is not None and sc.preview is None and sc.residuals is not None and not sc.fit_stale
    assert build_scene(None, None).data[0].size == 0


# ============================================================================ GUI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ezspec.gui.theme import apply_palette, configure_pyqtgraph
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_pyqtgraph()
    apply_palette(a)
    return a


def pump(app, n=6):
    for _ in range(n):
        app.processEvents()
        QtCore.QThread.msleep(5)


def wait_for(app, cond, what="condition"):
    for _ in range(2000):
        if cond():
            pump(app)
            return
        pump(app, 1)
    raise AssertionError(f"timed out waiting for {what}")


def settle(app, w):
    wait_for(app, lambda: not w._io_tasks, "file loading")


@pytest.fixture
def win(app, tmp_path):
    from ezspec.gui.main_window import MainWindow
    w = MainWindow()
    w.show()
    pump(app)
    yield w
    w.state.undo.setClean()
    w.close()


def test_explorer_lists_filters_and_signals(app, win, folder):
    w = win
    assert w.explorer.stack.currentIndex() == 0 and not w.act_close_folder.isEnabled()
    assert w.open_folder(folder)
    ex = w.explorer
    wait_for(app, lambda: len(ex.visible_files()) == 4, "folder listing")
    assert [p.name for p in ex.visible_files()] == ["multi.csv", "T1.csv", "T2.csv", "T10.csv"]   # natural order
    assert w.windowTitle().startswith("measurements") and w.act_close_folder.isEnabled()
    ex.filter.setText("t1")
    wait_for(app, lambda: len(ex.visible_files()) == 2, "filter")
    assert [p.name for p in ex.visible_files()] == ["T1.csv", "T10.csv"]
    ex.filter.setText("*2.CSV")
    wait_for(app, lambda: len(ex.visible_files()) == 1, "glob filter")
    ex.filter.clear()
    wait_for(app, lambda: len(ex.visible_files()) == 4, "filter cleared")
    write_table(folder / "T3.csv")                          # a spectrum written while the program runs
    wait_for(app, lambda: len(ex.visible_files()) == 5, "new file to appear")
    assert [p.name for p in ex.visible_files()][2:4] == ["T2.csv", "T3.csv"]
    # resting on a file previews it; Enter opens the selection
    previews, opens = [], []
    ex.previewRequested.connect(previews.append)
    ex.openRequested.connect(opens.append)
    assert ex.select_path(folder / "T2.csv")
    wait_for(app, lambda: previews, "preview signal")
    assert previews == [str(folder / "T2.csv")]
    settle(app, w)
    assert w.state.preview is not None and w.state.preview.name == "T2"
    ex.tree.returnPressed.emit()
    wait_for(app, lambda: opens, "open signal")
    assert opens == [[str(folder / "T2.csv")]]
    settle(app, w)
    assert [d.name for d in w.state.project.datasets] == ["T2"] and w.state.preview is None
    assert ex.proxy.open_paths == {(folder / "T2.csv").as_posix()}          # drawn in bold


def test_preview_is_transient_until_worked_on(app, win, folder):
    w = win
    st = w.state
    w.open_folder(folder)
    w.preview_file(folder / "T1.csv")
    settle(app, w)
    first = st.preview
    assert first is not None and st.current() is first and not st.project.datasets
    assert st.undo.isClean()                                # looking at a file is not a change
    items = [w.datasets_panel.list.item(i) for i in range(w.datasets_panel.list.count())]
    assert [i.text() for i in items] == ["T1"] and items[0].font().italic()
    pump(app, 10)
    assert len(w.plot.layer.c_data.getData()[0]) == 51      # … but it is drawn
    # the next preview replaces it
    w.preview_file(folder / "T10.csv")
    settle(app, w)
    assert st.preview is not first and st.preview.name == "T10" and not st.project.datasets
    # the first edit keeps it: one undo step that also removes it again
    w.pipeline_panel.add_step("baseline_arpls")
    pump(app)
    ds = st.current()
    assert st.preview is None and st.project.datasets == [ds] and ds.name == "T10" and len(ds.pipeline) == 1
    assert not w.datasets_panel.list.item(0).font().italic()
    st.undo.undo()
    assert not st.project.datasets and st.preview is None and st.current() is None
    st.undo.redo()
    assert st.project.datasets == [ds] and len(ds.pipeline) == 1
    # previewing a file that is open selects the open dataset instead of reading it again
    w.preview_file(folder / "T1.csv")
    settle(app, w)
    assert st.current() is st.preview
    w.preview_file(folder / "T10.csv")
    assert st.current() is ds and st.preview is not None
    # "Remove" on a preview just closes it
    st.set_current(st.preview.id)
    w.datasets_panel._remove()
    assert st.preview is None and st.project.datasets == [ds] and st.current() is ds


def test_open_many_files_is_one_undo_step(app, win, folder):
    w = win
    st = w.state
    w.open_folder(folder)
    w.open_files(spectrum_files(folder))
    settle(app, w)
    assert [d.name for d in st.project.datasets] == ["y1", "y2", "y3", "T1", "T2", "T10"]
    assert st.current() is st.project.datasets[0]
    n = len(st.project.datasets)
    w.open_files([folder / "T2.csv"])                       # already open: selected, not duplicated
    settle(app, w)
    assert len(st.project.datasets) == n and st.current().name == "T2"
    st.undo.undo()
    assert not st.project.datasets and st.current() is None
    st.undo.redo()
    assert len(st.project.datasets) == n
    # dropping files or a folder onto the window (or passing them on the command line) goes the same way
    w.open_paths([str(folder / "sub" / "raman.csv")])
    settle(app, w)
    assert len(st.project.datasets) == n + 1 and st.current().raw.n == 1500
    st.undo.setClean()
    w.open_paths([str(folder / "sub")])
    assert st.workspace_root == folder / "sub" and not st.project.datasets
    wait_for(app, lambda: [p.name for p in w.explorer.visible_files()] == ["raman.csv"], "sub folder listing")


def test_workspace_file_restores_the_folder(app, win, folder, monkeypatch):
    w = win
    st = w.state
    w.open_folder(folder)
    set_profile(st.project, folder / "multi.csv", "text", {"x_col": 0, "y_cols": [2]})
    w.open_files([folder / "multi.csv", folder / "T1.csv"])
    settle(app, w)
    assert [d.name for d in st.project.datasets] == ["multi", "T1"]         # profile: one column of multi.csv
    assert np.isclose(st.project.datasets[0].raw.y.max(), 2.0, rtol=1e-3)
    w.pipeline_panel.add_step("normalize")
    w.preview_file(folder / "T2.csv")
    settle(app, w)
    assert st.project.path is None and not st.undo.isClean()
    assert w.save_project()                                 # no question: the work is kept in the folder
    ws = folder / WORKSPACE_FILE
    assert ws.exists() and st.project.path == ws and st.undo.isClean()
    w.close_folder()
    assert st.workspace_root is None and not st.project.datasets and w.explorer.stack.currentIndex() == 0
    # reopening the folder brings everything back – except the preview, which was never part of it
    w.restore_session()                                     # nothing to restore after "Close Folder"
    assert st.workspace_root is None
    w.open_folder(folder)
    assert [d.name for d in st.project.datasets] == ["multi", "T1"]
    assert [s.op for s in st.project.get(st.project.datasets[0].id).pipeline] == ["normalize"]
    assert profiles_of(st.project)[".csv"]["options"]["y_cols"] == [2]
    assert st.preview is None and st.project.path == ws
    wait_for(app, lambda: (folder / "T1.csv").as_posix() in w.explorer.proxy.open_paths, "open-file marks")
    # a new session starts in that folder again
    assert w.settings.value("session/folder") == str(folder)
    assert w._recent()[0] == str(folder)
    w.close_folder()
    w.settings.setValue("session/folder", str(folder))
    w.restore_session()
    assert st.workspace_root == folder and len(st.project.datasets) == 2
    w._fill_recent_menu()
    assert [a.text().split("  –  ")[0] for a in w.recent_menu.actions()] == ["measurements"]
    # opening the workspace file as a project opens its folder too
    w.close_folder()
    w.open_project(str(ws))
    assert st.workspace_root == folder and len(st.project.datasets) == 2
    # a workspace file that was not loaded is never overwritten silently
    w.new_project()
    w.load_example("raman")
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QtWidgets.QMessageBox.No)
    assert not w.save_project() and asked and WORKSPACE_FILE in asked[0]
    assert len(Project.load(ws).datasets) == 2


def test_import_dialog_remembers_profile(app, win, folder):
    from ezspec.gui.dialogs import ImportDialog
    dlg = ImportDialog(folder / "multi.csv", win, remember=True)
    assert dlg.remember is not None and dlg.remember.isChecked()
    for j in range(dlg.ycols.count()):                      # keep only y2
        dlg.ycols.item(j).setCheckState(QtCore.Qt.Checked if j == 2 else QtCore.Qt.Unchecked)
    dlg._accept()
    assert dlg.remember_profile and dlg.reader == "text" and len(dlg.datasets) == 1
    assert dlg.options["y_cols"] == [2] and dlg.options["delimiter"] == "auto"
    proj = Project()
    set_profile(proj, dlg.path, dlg.reader, dlg.options)
    assert [d.raw.y.max() for d in load_file(folder / "multi.csv", profiles_of(proj)).datasets] == \
        [dlg.datasets[0].raw.y.max()]
    assert ImportDialog(folder / "multi.csv", win).remember is None       # plain import: no switch


def test_layout_is_remembered(app, win):
    from ezspec.gui.main_window import MainWindow
    w = win
    assert w.d_explorer.isVisible()
    w.d_hist.close()
    w.d_explorer.setFloating(True)
    w.close()
    w2 = MainWindow()
    w2.show()
    pump(app)
    try:
        assert w2.d_explorer.isFloating() and not w2.d_hist.isVisible() and w2.d_model.isVisible()
    finally:
        w2.close()
