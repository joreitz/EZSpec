"""Window shell: command palette, go to file, activity bar, welcome page, drop onto the plot."""

import os

import numpy as np
import pytest

from ezspec.gui.quick_pick import fuzzy_score, rank

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtGui, QtTest  # noqa: E402


# ============================================================================ matching (no window)
def test_fuzzy_matching_prefers_words_and_rejects_scattered_letters():
    assert fuzzy_score("open folder", "File › Open Folder") is not None
    assert fuzzy_score("opfo", "File › Open Folder") is not None            # letters in order
    assert fuzzy_score("folder open", "File › Open Folder") is not None     # words in any order
    assert fuzzy_score("xyz", "File › Open Folder") is None
    assert fuzzy_score("stack", "Raman spectrum with fluorescence background") is None
    items = [("Analysis › Run fit", "Ctrl+R", 1), ("File › Export › Fit results (CSV/JSON/report)", "", 2),
             ("Plot › Auto-range", "", 3), ("sample_10K.csv", "calibration", 4)]
    assert [i[2] for i in rank("fit", items)] == [1, 2]                     # the shorter, earlier match first
    assert [i[2] for i in rank("calib", items)] == [4]                      # found through its detail
    assert rank("", items) == items and rank("nothing here", items) == []
    assert len(rank("", items * 100, limit=7)) == 7


# ============================================================================ window
@pytest.fixture(scope="module")
def app():
    from ezspec.gui.theme import apply_palette, configure_pyqtgraph
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_pyqtgraph()
    apply_palette(a)
    return a


def pump(app, n=8):
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


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "series"
    (d / "calibration").mkdir(parents=True)
    x = np.linspace(100.0, 200.0, 51)
    for name, h in (("T2.csv", 1.0), ("T10.csv", 2.0), ("calibration/neon.csv", 3.0)):
        y = h * np.exp(-0.5 * ((x - 150.0) / 5.0) ** 2)
        (d / name).write_text("x,y\n" + "\n".join(f"{a:.8g},{b:.8g}" for a, b in zip(x, y)) + "\n")
    return d


@pytest.fixture
def win(app):
    from ezspec.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1300, 800)
    w.show()
    pump(app)
    yield w
    if w._pick is not None:
        w._pick.close()
    w.state.mark_saved()
    w.close()


def test_welcome_page_until_there_is_something_to_plot(app, win, folder):
    w = win
    assert w.central.currentWidget() is w.welcome and not w.welcome.recent_buttons
    w.open_folder(folder)
    pump(app)
    assert w.central.currentWidget() is w.welcome                           # a folder, but nothing opened yet
    assert [b.toolTip() for b in w.welcome.recent_buttons] == [str(folder.resolve())]
    w.preview_file(folder / "T2.csv")
    wait_for(app, lambda: w.state.preview is not None, "preview")
    assert w.central.currentWidget() is w.plot                              # a preview is something to look at
    w.state.set_preview(None)
    pump(app)
    assert w.central.currentWidget() is w.welcome
    # the links of the page do what the menu does
    w.welcome.exampleRequested.emit()
    pump(app)
    assert w.central.currentWidget() is w.plot and len(w.state.project.datasets) == 1
    w.state.mark_saved()
    w.close_folder()
    pump(app)
    assert w.central.currentWidget() is w.welcome and w.state.workspace_root is None
    w.welcome.recent_buttons[0].click()                                     # back into the folder
    pump(app)
    assert w.state.workspace_root == folder.resolve()


def test_command_palette_runs_menu_and_plot_commands(app, win):
    from ezspec.gui.quick_pick import menu_commands
    w = win
    w.load_example("raman")
    w.load_example("decay")
    pump(app)
    commands = {title: (shortcut, action) for title, shortcut, action in menu_commands(w.menuBar())}
    assert commands["File › Open Folder"][0] == "Ctrl+Shift+O"
    assert "File › Export › Python script (reproducible)" in commands
    assert "View › Number format › Values with uncertainty › 2 significant digits of the uncertainty" in commands

    pick = w.command_palette()
    assert pick.isVisible() and pick.list.count() > 20
    QtTest.QTest.keyClicks(pick.edit, "overlay all")
    assert pick._shown[0][0] == "View › Overlay all datasets"
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Return)
    pump(app)
    assert not pick.isVisible() and len(w.state.view.shown) == 2
    # commands of the plot toolbar are found as well; the arrow keys move through the list
    pick = w.command_palette()
    QtTest.QTest.keyClicks(pick.edit, "stack")
    assert [t for t, _d, _c in pick._shown] == ["Plot › Stack spectra (on/off)"]
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Return)
    pump(app)
    assert w.state.view.stacked
    pick = w.command_palette()
    QtTest.QTest.keyClicks(pick.edit, "normalise")
    assert pick.list.count() == 4 and pick.list.currentRow() == 0
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Down)
    assert pick.list.currentRow() == 1
    chosen = pick._shown[1][0]
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Return)
    pump(app)
    assert w.state.view.normalize != "none"
    assert chosen == f"Plot › Normalise: {w.plot.norm_actions[w.state.view.normalize].text()}"
    pick = w.command_palette()
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Escape)
    pump(app)
    assert not pick.isVisible()


def test_go_to_file_opens_files_of_the_folder(app, win, folder):
    w = win
    w.open_folder(folder)
    w.open_files([folder / "T2.csv"])
    wait_for(app, lambda: len(w.state.project.datasets) == 1 and not w._io_tasks, "open")
    pick = w.quick_open()
    titles = [(t, d) for t, d, _c in pick.items]
    assert titles == [("T2", "open dataset"), ("neon.csv", "calibration"), ("T10.csv", "")]
    QtTest.QTest.keyClicks(pick.edit, "neon")
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Return)
    wait_for(app, lambda: len(w.state.project.datasets) == 2 and not w._io_tasks, "open")
    assert w.state.current().name == "neon"
    pick = w.quick_open()                                                   # an open dataset: just go there
    QtTest.QTest.keyClicks(pick.edit, "T2")
    QtTest.QTest.keyClick(pick.edit, QtCore.Qt.Key_Return)
    pump(app)
    assert w.state.current().name == "T2" and len(w.state.project.datasets) == 2


def test_activity_bar_and_side_bar_toggles(app, win):
    w = win
    left = QtCore.Qt.LeftDockWidgetArea
    actions = {a.toolTip().split(" ")[0]: a for a in w.activity.actions() if a.toolTip()}
    assert {"Explorer", "Data", "Processing", "Model", "Results", "History", "Go", "Pages"} <= set(actions)
    assert actions["Explorer"].isChecked() and not w.d_explorer.isHidden()
    actions["Explorer"].trigger()
    pump(app)
    assert w.d_explorer.isHidden() and not actions["Explorer"].isChecked()
    actions["Explorer"].trigger()
    pump(app)
    assert not w.d_explorer.isHidden()
    # Ctrl+B: everything on the left goes and comes back as it was
    w.d_hist.hide()
    width = w.central.width()
    w.toggle_side(left)
    pump(app)
    assert all(d.isHidden() for d in (w.d_explorer, w.d_data, w.d_pipe)) and w.central.width() > width + 200
    assert not w.d_model.isHidden()
    w.toggle_side(left)
    pump(app)
    assert not any(d.isHidden() for d in (w.d_explorer, w.d_data, w.d_pipe)) and w.d_hist.isHidden()
    w.toggle_side(QtCore.Qt.RightDockWidgetArea)
    pump(app)
    assert w.d_model.isHidden() and w.d_res.isHidden()


def test_files_dropped_onto_the_plot_are_overlaid(app, win, folder):
    w = win
    w.open_folder(folder)
    w.open_files([folder / "T2.csv"])
    wait_for(app, lambda: len(w.state.project.datasets) == 1 and not w._io_tasks, "open")
    assert w.explorer.tree.dragEnabled()

    def drop(path, pos):
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(path))])
        w.dropEvent(QtGui.QDropEvent(QtCore.QPointF(pos), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton,
                                     QtCore.Qt.NoModifier))

    drop(folder / "T10.csv", w.plot.mapTo(w, w.plot.rect().center()))        # onto the plot
    wait_for(app, lambda: len(w.state.project.datasets) == 2 and not w._io_tasks, "open")
    t2, t10 = w.state.project.datasets
    assert w.state.view.shown == [t10.id] and w.state.current() is t10
    w.state.set_current(t2.id)
    pump(app)
    assert set(w.plot.layer.overlay_items) == {t10.id}
    drop(folder / "calibration" / "neon.csv", w.d_data.mapTo(w, w.d_data.rect().center()))   # elsewhere: just open
    wait_for(app, lambda: len(w.state.project.datasets) == 3 and not w._io_tasks, "open")
    assert w.state.view.shown == [t10.id]
