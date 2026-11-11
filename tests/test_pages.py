"""Pages: captured views placed freely on a sheet and exported together."""

import os

import numpy as np
import pytest

from ezspec.export.page import (MARGIN, PAGES_KEY, align, capture, check_page, frame_scene, free_rect, new_frame,
                                new_page, project_resolver, render_page, save_page)
from ezspec.models import add_peak
from ezspec.project import Dataset, Project
from ezspec.spectrum import Spectrum
from ezspec.view import ViewState


def dataset(name, height=1.0, center=150.0):
    x = np.linspace(100.0, 200.0, 201)
    y = height * np.exp(-0.5 * ((x - center) / 4.0) ** 2)
    return Dataset(name=name, raw=Spectrum(x, y, x_label="Shift", y_label="Counts", meta={"name": name}))


@pytest.fixture
def project():
    proj = Project()
    for name, h, c in (("a", 1.0, 140.0), ("b", 2.0, 150.0), ("c", 3.0, 160.0)):
        proj.add(dataset(name, h, c))
    return proj


def overlaps(r, s):
    return r[0] < s[0] + s[2] and s[0] < r[0] + r[2] and r[1] < s[1] + s[3] and s[1] < r[1] + r[3]


# ============================================================================ model (no Qt)
def test_captured_views_are_placed_next_to_each_other(project):
    a, b, c = project.datasets
    page = new_page()
    first = capture(page, a.id, [b.id], ViewState(stacked=True, gap=0.5, normalize="max"))
    assert len(first) == 1 and first[0]["rect"][:2] == [MARGIN, MARGIN]
    assert first[0]["shown"] == [b.id] and first[0]["stacked"] and first[0]["normalize"] == "max"
    # a view with zoom regions: the overview and one frame per region, arranged as in the plot
    group = capture(page, c.id, [], regions=[[130, 140], [155, 165]], x_range=[110, 190])
    assert [f["x_range"] for f in group] == [[110, 190], [130.0, 140.0], [155.0, 165.0]]
    over, z1, z2 = (f["rect"] for f in group)
    assert z1[0] == pytest.approx(over[0] + over[2]) and z2[1] == pytest.approx(z1[1] + z1[3])
    assert z1[3] + z2[3] == pytest.approx(over[3])
    rects = [f["rect"] for f in page["frames"]]
    assert not any(overlaps(r, s) for i, r in enumerate(rects) for s in rects[i + 1:])
    W, H = page["width_mm"], page["height_mm"]
    assert all(r[0] >= MARGIN - 1e-9 and r[0] + r[2] <= W - MARGIN + 1e-9 and r[1] + r[3] <= H - MARGIN + 1e-9
               for r in rects)
    # a full page still takes frames (cascaded), and never a frame larger than the page
    small = new_page(size="col_1")
    for _ in range(6):
        capture(small, a.id)
    assert len(small["frames"]) == 6 and all(f["rect"][2] <= 70.0 for f in small["frames"])
    assert free_rect(new_page(), 1000, 1000)[2:] == [297.0 - 2 * MARGIN, 210.0 - 2 * MARGIN]


def test_pages_from_older_files_are_completed():
    page = check_page({"name": "old", "frames": [{"active": "x", "rect": [1, 2, 30, 40], "normalize": "?"},
                                                 {"rect": [0, 0, 1, 1]}, "junk"]})
    assert page["width_mm"] == 297.0 and len(page["frames"]) == 1
    assert page["frames"][0]["normalize"] == "none" and page["frames"][0]["fit"] is True


def test_align_and_distribute():
    frames = [new_frame("a", rect=[10, 10, 40, 30]), new_frame("b", rect=[70, 25, 20, 20]),
              new_frame("c", rect=[150, 60, 30, 10])]
    align(frames, "top")
    assert [f["rect"][1] for f in frames] == [10, 10, 10]
    align(frames, "height")
    assert [f["rect"][3] for f in frames] == [30, 30, 30]
    align(frames, "hspace")
    r = [f["rect"] for f in frames]
    assert r[0][0] == 10 and r[2][0] + r[2][2] == 180
    assert r[1][0] - (r[0][0] + r[0][2]) == pytest.approx(r[2][0] - (r[1][0] + r[1][2]))
    align(frames, "right")
    assert [f["rect"][0] + f["rect"][2] for f in frames] == [180, 180, 180]
    align(frames[:1], "left")                                # a single frame: nothing to align with
    with pytest.raises(ValueError):
        align(frames, "diagonal")


def test_page_renders_every_frame_inside_its_box(project, tmp_path):
    a, b, c = project.datasets
    add_peak(c.model, "gaussian", 160.0, 3.0, 9.0)
    page = new_page()
    capture(page, c.id, [a.id, b.id], regions=[[130, 150]])
    capture(page, a.id, [b.id, c.id], ViewState(stacked=True, normalize="max"))
    resolve = project_resolver(project)
    fig = render_page(page, resolve)
    assert len(fig.axes) == 3
    W, H = page["width_mm"], page["height_mm"]
    renderer = fig.canvas.get_renderer()
    for ax, frame in zip(fig.axes, page["frames"]):
        x, y, w, h = frame["rect"]
        t = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        eps = 1e-3
        assert t.x0 >= x / W - eps and t.x1 <= (x + w) / W + eps, "labels stay inside the frame"
        assert t.y0 >= 1 - (y + h) / H - eps and t.y1 <= 1 - y / H + eps
    over, zoom, stacked = fig.axes
    assert over.get_xlabel() == "Shift" and zoom.get_xlabel() == ""          # zoom frames carry no labels
    assert zoom.get_xlim() == (130.0, 150.0)
    assert 1.9 < zoom.get_ylim()[1] < 2.3                                    # b peaks inside, c (3) does not
    assert over.get_ylim()[1] > 3.0
    assert stacked.get_ylim()[1] > 2.9 and "stacked" in stacked.get_ylabel()
    # no fit yet: data and overlays only; after the fit the same page shows it
    assert not any(line.get_label() == "Fit" for line in over.lines)
    c.run_fit()
    fig = render_page(page, project_resolver(project))
    assert any(line.get_label() == "Fit" for line in fig.axes[0].lines)
    page["frames"][0]["fit"] = False
    assert not any(line.get_label() == "Fit" for line in render_page(page, project_resolver(project)).axes[0].lines)

    for ext, magic in ((".pdf", b"%PDF"), (".png", b"\x89PNG"), (".svg", b"<?xml")):
        page["dpi"] = 60
        save_page(page, resolve, tmp_path / f"page{ext}")
        assert (tmp_path / f"page{ext}").read_bytes().startswith(magic)
    # a frame of a dataset that was removed is left empty instead of failing the export
    project.datasets.remove(c)
    assert frame_scene(page["frames"][0], project_resolver(project)) is None
    assert len(render_page(page, project_resolver(project)).axes) == 3
    # a single frame at its own size (the preview of the composer)
    one = render_page(page, project_resolver(project), only=page["frames"][2])
    assert one.get_size_inches() * 25.4 == pytest.approx(page["frames"][2]["rect"][2:])


# ============================================================================ GUI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtTest  # noqa: E402


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
def win(app):
    from ezspec.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1300, 800)
    w.show()
    w.state.add_datasets([dataset(n, h, c) for n, h, c in (("a", 1.0, 140.0), ("b", 2.0, 150.0),
                                                           ("c", 3.0, 160.0))], "three spectra")
    w.state.mark_saved()
    pump(app)
    yield w
    w.state.mark_saved()
    w.close()


def rendered(c):
    return bool(c.items) and all(it.pixmap is not None for it in c.items) and not c._dirty


def test_views_are_captured_frame_by_frame(app, win, tmp_path):
    w = win
    a, b, c = w.state.project.datasets
    w.state.set_shown([b.id])
    w.plot.regionsEdited.emit([[130.0, 150.0]])
    pump(app)
    assert not w.state.modified
    frames = w.capture_view()
    assert [f["active"] for f in frames] == [a.id, a.id] and frames[0]["shown"] == [b.id]
    assert frames[0]["x_range"] is None and frames[1]["x_range"] == [130.0, 150.0]
    assert w.state.modified and w.windowTitle().split(" – ")[0].endswith("*")     # pages are worth saving
    comp = w._composer
    assert comp.isVisible() and len(comp.items) == 2
    wait_for(app, lambda: rendered(comp), "frames to render")

    # another view: stacked, zoomed with the mouse wheel to 120 … 180
    w.plot.regionsEdited.emit([])
    w.plot.act_stack.setChecked(True)
    w.plot.p_main.getViewBox().setXRange(120.0, 180.0, padding=0)
    pump(app)
    third = w.capture_view()
    assert len(third) == 1 and third[0]["stacked"] and third[0]["x_range"] == pytest.approx([120.0, 180.0])
    assert len(comp.items) == 3 and [it.frame for it in comp.selected()] == third
    rects = [it.frame["rect"] for it in comp.items]
    assert not any(overlaps(r, s) for i, r in enumerate(rects) for s in rects[i + 1:])
    assert comp.data_label.text() == "a + 1 overlaid"

    # the page is part of the project file
    w.state.project.path = tmp_path / "p.ezspec"
    assert w.save_project() and not w.state.modified
    w.new_project()
    pump(app)
    assert comp.page["frames"] == [] and not comp.items
    w.open_project(str(tmp_path / "p.ezspec"))
    pump(app)
    assert len(w.state.project.meta[PAGES_KEY][0]["frames"]) == 3 and len(comp.items) == 3
    assert comp.export(str(tmp_path / "page.pdf")) and (tmp_path / "page.pdf").read_bytes().startswith(b"%PDF")


def test_frames_are_moved_resized_and_aligned(app, win):
    w = win
    a, b, c = w.state.project.datasets
    w.capture_view()
    w.state.set_current(b.id)
    pump(app)
    w.capture_view()
    comp = w._composer
    comp.resize(1200, 760)
    wait_for(app, lambda: rendered(comp), "frames to render")
    first, second = comp.items
    assert first.frame["rect"] == [10.0, 10.0, 90.0, 68.0] and second.frame["rect"][0] == 105.0
    view = comp.view.viewport()

    def at(x, y):
        return comp.view.mapFromScene(QtCore.QPointF(x, y))

    def drag(start, end):
        QtTest.QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
        for k in range(1, 5):
            QtTest.QTest.mouseMove(view, start + (end - start) * k / 4)
            pump(app, 1)
        QtTest.QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, end)
        pump(app)

    # move the second frame below the first: its left edge snaps to the first one's
    drag(at(150.0, 40.0), at(56.5, 121.3))
    x, y, wd, ht = second.frame["rect"]
    assert x == 10.0 and (wd, ht) == (90.0, 68.0) and 90.0 <= y <= 92.0
    assert y == round(y)                                                # full millimetres
    assert comp.selected() == [second] and comp.geo[0].value() == 10.0
    pix = second.pixmap
    # resize it with the bottom-right handle: the frame is drawn again at the new size
    drag(at(x + wd, y + ht), at(x + wd + 30.2, y + ht + 10.4))
    assert second.frame["rect"][2] == pytest.approx(120.0, abs=1.0) and second.frame["rect"][3] > ht
    wait_for(app, lambda: rendered(comp) and second.pixmap is not pix, "new preview")
    assert second.pixmap.width() > pix.width()
    assert w.state.modified

    # align and equalise through the menu actions
    comp.select_all()
    comp.align_selected("width")
    comp.align_selected("left")
    assert first.frame["rect"][2] == second.frame["rect"][2] == 90.0
    assert first.frame["rect"][0] == second.frame["rect"][0] == 10.0
    # options of the selected frames
    comp.checks["legend"].setChecked(True)
    comp.x_from.setText("130")
    comp.x_to.setText("170")
    comp.x_to.editingFinished.emit()
    assert all(it.frame["legend"] and it.frame["x_range"] == [130.0, 170.0] for it in comp.items)
    comp.x_from.setText("")
    comp.x_from.editingFinished.emit()
    assert first.frame["x_range"] is None
    # duplicate and delete
    comp.scene.clearSelection()
    first.setSelected(True)
    comp.duplicate_selected()
    assert len(comp.items) == 3 and comp.page["frames"][2]["rect"][:2] == [16.0, 16.0]
    assert comp.page["frames"][2]["id"] != first.frame["id"]
    comp.delete_selected()
    assert len(comp.items) == 2 and len(comp.page["frames"]) == 2
    # pages
    comp.size_box.setCurrentIndex(comp.size_box.findData("a4_portrait"))
    comp._size_chosen(0)
    assert (comp.page["width_mm"], comp.page["height_mm"]) == (210.0, 297.0)
    comp.add_page()
    assert len(w.state.project.meta[PAGES_KEY]) == 2 and comp.page["frames"] == [] and not comp.items
    assert comp.page["width_mm"] == 210.0                               # a new page takes over the format
    w.capture_view()
    assert len(comp.page["frames"]) == 1 and len(w.state.project.meta[PAGES_KEY][0]["frames"]) == 2
    comp.page_box.setCurrentIndex(0)
    assert len(comp.items) == 2


def test_pages_follow_the_analysis(app, win):
    w = win
    a, b, c = w.state.project.datasets
    w.capture_view()
    comp = w._composer
    wait_for(app, lambda: rendered(comp), "frames to render")
    before = comp.items[0].pixmap
    w.state.edit("pipeline", lambda p: p.add("offset_scale", {"factor": 3.0}), "scale")
    wait_for(app, lambda: rendered(comp) and comp.items[0].pixmap is not before, "the frame to be drawn again")
    assert before.toImage() != comp.items[0].pixmap.toImage()
    # removing the dataset empties the frame, it does not break the page
    w.state.remove_dataset(a)
    wait_for(app, lambda: rendered(comp), "frames to render")
    assert comp.data_label.text() in ("(removed)", "Select a frame on the page.")
    w.state.undo.undo()
    wait_for(app, lambda: rendered(comp), "frames to render")
