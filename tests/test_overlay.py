"""Several spectra in one plot: overlays, colours, stacking and normalisation (display only)."""

import os

import numpy as np
import pytest

from ezspec.models import add_peak
from ezspec.project import Dataset, Project
from ezspec.spectrum import Spectrum
from ezspec.view import PALETTE, ViewState, display_transforms, next_color, normalization


def spectrum(height=1.0, center=150.0, offset=0.0, n=201, name="s"):
    x = np.linspace(100.0, 200.0, n)
    y = offset + height * np.exp(-0.5 * ((x - center) / 4.0) ** 2)
    return Spectrum(x, y, meta={"name": name})


def dataset(name, **kw):
    return Dataset(name=name, raw=spectrum(name=name, **kw))


# ============================================================================ view state (no Qt)
def test_view_state_round_trip_and_tolerant_loading():
    v = ViewState(shown=["a", "b"], colors={"a": "#112233"}, stacked=True, gap=0.5, normalize="max",
                  regions=[[120.0, 130.0]])
    assert ViewState.from_dict(v.to_dict()) == v
    assert ViewState.from_dict(None) == ViewState()
    # a file written by a newer or broken version must not stop the project from opening
    odd = ViewState.from_dict({"normalize": "weird", "gap": "x", "regions": [[3, 1], ["a", 2], [5, 5]],
                               "shown": None})
    assert odd.normalize == "none" and odd.gap == 1.0 and odd.regions == [[1.0, 3.0]] and odd.shown == []
    v.prune(["b"])
    assert v.shown == ["b"] and v.colors == {}
    assert ViewState().transformed is False and ViewState(normalize="area").transformed is True


def test_colours_are_distinct_until_the_palette_is_used_up():
    used = []
    for _ in PALETTE:
        used.append(next_color(used))
    assert sorted(used) == sorted(PALETTE)
    assert next_color(used) in PALETTE


def test_normalisation_modes():
    s = spectrum(height=4.0, offset=2.0)
    for mode, lo, hi in (("none", 2.0, 6.0), ("max", 1 / 3, 1.0), ("range", 0.0, 1.0)):
        a, b = normalization(s, mode)
        y = a * s.y + b
        assert y.min() == pytest.approx(lo, abs=1e-6) and y.max() == pytest.approx(hi, abs=1e-12), mode
    a, b = normalization(s, "area")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    assert trapezoid(a * s.y + b, s.x) == pytest.approx(1.0)
    # a masked spike does not set the scale
    y = s.y.copy()
    y[10] = 1e6
    exclude = np.zeros(s.n, bool)
    exclude[10] = True
    spiky = Spectrum(s.x, y, exclude=exclude)
    assert normalization(spiky, "max") == pytest.approx(normalization(s, "max"))
    # nothing to scale by: left alone instead of dividing by zero
    assert normalization(Spectrum(s.x, np.zeros(s.n)), "max") == (1.0, 0.0)
    assert normalization(Spectrum(s.x, np.full(s.n, 3.0)), "range") == (1.0, -3.0)


def test_stacking_keeps_the_order_and_scales_with_the_tallest():
    spectra = [("a", spectrum(1.0)), ("b", spectrum(4.0)), ("c", spectrum(2.0))]
    assert display_transforms(spectra, ViewState()) == {"a": (1.0, 0.0), "b": (1.0, 0.0), "c": (1.0, 0.0)}
    t = display_transforms(spectra, ViewState(stacked=True, gap=0.5))
    assert [t[k][1] for k in "abc"] == pytest.approx([0.0, 2.0, 4.0])      # 0.5 × tallest (4)
    t = display_transforms(spectra, ViewState(stacked=True, normalize="max"))
    assert [t[k] for k in "abc"] == [pytest.approx((1.0, 0.0)), pytest.approx((0.25, 1.0)),
                                     pytest.approx((0.5, 2.0))]
    # a single spectrum is not moved
    assert display_transforms(spectra[:1], ViewState(stacked=True)) == {"a": (1.0, 0.0)}


def test_view_scene_overlays_and_maps_tools_back_to_data():
    from ezspec.gui.scene import build_view, data_y
    a, b = dataset("a", height=2.0), dataset("b", height=1.0, center=160.0)
    add_peak(b.model, "gaussian", 160.0, 1.0, 8.0)
    entries = [(a, a.run_pipeline(), "#111111"), (b, b.run_pipeline(), "#222222")]

    plain = build_view(b, entries[1][1], entries, ViewState())
    assert plain.y_map == (1.0, 0.0) and plain.color == "#222222" and plain.name == "b"
    assert [(o.id, o.name, o.color) for o in plain.overlays] == [(a.id, "a", "#111111")]
    assert plain.overlays[0].y.max() == pytest.approx(2.0)
    top = plain.peaks[0]["top"]
    assert top == pytest.approx((160.0, 1.0))

    stacked = build_view(b, entries[1][1], entries, ViewState(stacked=True, normalize="max"))
    assert stacked.y_map == pytest.approx((1.0, 1.0))                  # second from the bottom
    assert stacked.overlays[0].y.max() == pytest.approx(1.0)           # a: scaled by 1/2, at the bottom
    assert stacked.data[1].max() == pytest.approx(2.0)
    assert stacked.peaks[0]["top"] == pytest.approx((160.0, 2.0))       # the handle sits on the drawn curve
    assert stacked.preview[1].max() == pytest.approx(2.0, abs=1e-3)
    assert data_y(stacked, stacked.peaks[0]["top"][1]) == pytest.approx(1.0)
    assert stacked.y_label.endswith("(normalised, stacked)")

    # while a step is edited only the active spectrum is drawn, untransformed
    step = b.pipeline.add("offset_scale", {})
    run = b.run_pipeline()
    sc = build_view(b, run, [(a, entries[0][1], "#111111"), (b, run, "#222222")],
                    ViewState(stacked=True, normalize="max"), step_id=step.id)
    assert sc.step is step and sc.overlays == () and sc.y_map == (1.0, 0.0) and sc.color == "#222222"
    assert build_view(None, None).data[0].size == 0


def test_view_is_stored_in_the_project_file(tmp_path):
    proj = Project()
    a, b = proj.add(dataset("a")), proj.add(dataset("b"))
    proj.meta["view"] = ViewState(shown=[a.id, b.id], colors={a.id: "#123456"}, stacked=True, gap=0.3).to_dict()
    proj.save(tmp_path / "p.ezspec")
    v = ViewState.from_dict(Project.load(tmp_path / "p.ezspec").meta["view"])
    assert v.shown == [a.id, b.id] and v.colors == {a.id: "#123456"} and v.stacked and v.gap == 0.3


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
    """Window with three spectra a (active), b, c of heights 1, 2, 3."""
    from ezspec.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1300, 800)
    w.show()
    w.state.add_datasets([dataset(n, height=h, center=c) for n, h, c in
                          (("a", 1.0, 140.0), ("b", 2.0, 150.0), ("c", 3.0, 160.0))], "three spectra")
    pump(app)
    yield w
    w.state.undo.setClean()
    w.close()


def click_plot(app, w, x, y):
    """A real left click at the data position (x, y) of the main panel."""
    vb = w.plot.p_main.getViewBox()
    pos = w.plot.glw.mapFromScene(vb.mapViewToScene(QtCore.QPointF(x, y)))
    QtTest.QTest.mouseMove(w.plot.glw.viewport(), pos)
    pump(app, 2)
    QtTest.QTest.mouseClick(w.plot.glw.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pos)
    pump(app)


def test_ticked_spectra_stay_in_the_plot(app, win):
    w = win
    a, b, c = w.state.project.datasets
    layer = w.plot.layer
    assert w.state.current() is a and not layer.overlay_items          # as before: one spectrum
    assert w.state.undo.isClean() is False                             # (adding the datasets)
    w.state.undo.setClean()

    w.state.set_shown([b.id, c.id])
    pump(app)
    assert set(layer.overlay_items) == {b.id, c.id}
    assert layer.overlay_items[c.id].getData()[1].max() == pytest.approx(3.0)
    colors = {d.id: w.state.color_of(d) for d in (a, b, c)}
    assert len(set(colors.values())) == 3
    assert layer.overlay_items[b.id].opts["pen"].color().name() == colors[b.id]
    assert layer.c_data.opts["pen"].color().name() == colors[a.id]
    labels = [label.text for _sample, label in w.plot.legend.items]
    assert labels == ["a", "b", "c"]                                   # the data are named, not "Data"
    assert w.state.undo.isClean()                                      # looking at data is not an edit

    # another active dataset: the ticked ones stay, the active one is drawn with its tools
    w.state.set_current(b.id)
    pump(app)
    assert set(layer.overlay_items) == {c.id} and w.state.shown_datasets() == [b, c]
    w.state.set_shown([a.id])
    pump(app)
    assert set(layer.overlay_items) == {a.id, c.id}

    # a removed dataset leaves the plot and comes back with undo
    w.state.remove_dataset(c)
    pump(app)
    assert set(layer.overlay_items) == {a.id}
    w.state.undo.undo()
    w.state.set_current(b.id)
    pump(app)
    assert set(layer.overlay_items) == {a.id, c.id}

    w.show_all(False)
    pump(app)
    assert not layer.overlay_items
    labels = [label.text for _sample, label in w.plot.legend.items]
    assert labels == ["Data"]


def test_tick_box_does_not_change_the_active_dataset(app, win):
    w = win
    a, b, c = w.state.project.datasets
    lst = w.datasets_panel.list
    rect = lst.visualItemRect(lst.item(2))
    QtTest.QTest.mouseClick(lst.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                            QtCore.QPoint(rect.left() + 8, rect.center().y()))
    pump(app)
    assert w.state.view.shown == [c.id] and w.state.current() is a
    assert lst.item(2).checkState() == QtCore.Qt.Checked and lst.item(0).checkState() == QtCore.Qt.Unchecked
    # a click on the name makes it the active dataset
    rect = lst.visualItemRect(lst.item(1))
    QtTest.QTest.mouseClick(lst.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                            QtCore.QPoint(rect.left() + 60, rect.center().y()))
    pump(app)
    assert w.state.current() is b and w.state.view.shown == [c.id]
    # ticking one row of a selection ticks the selection
    lst.item(0).setSelected(True)
    lst.item(1).setSelected(True)
    lst.item(0).setCheckState(QtCore.Qt.Checked)
    pump(app)
    assert set(w.state.view.shown) == {a.id, b.id, c.id}
    assert w.datasets_panel.selected_ids() == [a.id, b.id]              # the selection survives the refresh
    lst.item(1).setCheckState(QtCore.Qt.Unchecked)
    pump(app)
    assert w.state.view.shown == [c.id]


def test_click_on_an_overlaid_spectrum_activates_it(app, win):
    w = win
    a, b, c = w.state.project.datasets
    w.state.set_shown([b.id, c.id])
    pump(app)
    click_plot(app, w, 160.0, 3.0)                                     # the top of c
    assert w.state.current() is c
    # with a tool selected a click belongs to the tool, even on top of another spectrum
    w.plot.set_mode("peak")
    click_plot(app, w, 150.0, 2.0)                                     # the top of b
    assert w.state.current() is c and len(c.model.peaks) == 1
    assert c.model.peaks[0].derived(c.model.initial_values())["center"] == pytest.approx(150.0, abs=0.6)


def test_tools_work_on_the_data_in_a_stacked_view(app, win):
    w = win
    a, b, c = w.state.project.datasets
    w.state.set_shown([a.id, b.id, c.id])
    w.state.set_current(b.id)
    w.plot.act_stack.setChecked(True)                                  # the toolbar switch
    pump(app)
    assert w.state.view.stacked and w.plot._y_map == pytest.approx((1.0, 3.0))   # 100 % of the tallest (3)
    assert w.plot.layer.c_data.getData()[1].max() == pytest.approx(5.0)
    assert w.plot.layer.overlay_items[c.id].getData()[1].max() == pytest.approx(9.0)
    assert "stacked" in w.plot.p_main.getAxis("left").labelText

    w.plot.set_mode("peak")
    click_plot(app, w, 150.0, 5.0)                                     # the drawn top of b
    assert len(b.model.peaks) == 1
    d = b.model.peaks[0].derived(b.model.initial_values())
    assert d["height"] == pytest.approx(2.0, abs=0.1)                  # the height of the data, not 5
    handle = w.plot._peak_items[0]
    assert handle.pos().y() == pytest.approx(3.0 + d["height"], abs=1e-6)
    # dragging the handle: positions are reported in data coordinates
    seen = []
    w.plot.peakDragged.connect(lambda *args: seen.append(args))
    handle.setPos(151.0, 4.5)
    handle.sigPositionChangeFinished.emit(handle)
    pump(app)
    assert seen[-1][2:] == (151.0, 1.5, True)
    d = b.model.peaks[0].derived(b.model.initial_values())
    assert d["center"] == pytest.approx(151.0) and d["height"] == pytest.approx(1.5, abs=1e-6)

    # normalise: every spectrum has height 1, 50 % apart
    w.plot.norm_actions["max"].trigger()
    w.plot.gap_spin.setValue(50)
    pump(app)
    assert w.state.view.normalize == "max" and w.state.view.gap == 0.5
    assert w.plot._y_map == pytest.approx((0.5, 0.5))
    assert w.plot.layer.overlay_items[c.id].getData()[1].max() == pytest.approx(2.0)
    assert w.plot.norm_button.text() == "Norm: max"
    # a step is edited on the untransformed spectrum alone
    w.pipeline_panel.add_step("offset_scale")
    pump(app)
    assert w.plot._y_map == (1.0, 0.0) and not w.plot.layer.overlay_items
    w.state.select_step(None)
    pump(app)
    assert len(w.plot.layer.overlay_items) == 2


def test_copy_and_paste_processing_to_a_selection(app, win):
    w = win
    a, b, c = w.state.project.datasets
    panel = w.datasets_panel
    w.pipeline_panel.add_step("offset_scale")
    w.state.edit("pipeline", lambda p: p.update(w.state.selected_step, {"factor": 2.0}), "factor")
    w.state.edit("model", lambda m: add_peak(m, "gaussian", 140.0, 2.0, 8.0), "peak")
    w.state.select_step(None)
    pump(app)
    panel.copy_setup()
    n_undo = w.state.undo.count()
    for i in range(3):
        panel.list.item(i).setSelected(True)
    assert panel.paste_setup() == 2                                     # a already has it
    pump(app)
    assert [len(d.pipeline) for d in (a, b, c)] == [1, 1, 1] and not b.model.components
    assert w.state.run(c).final.y.max() == pytest.approx(6.0)
    assert w.state.undo.count() == n_undo + 1                           # one step for all
    w.state.undo.undo()
    pump(app)
    assert [len(d.pipeline) for d in (a, b, c)] == [1, 0, 0]
    assert panel.paste_setup(model=True) == 2
    assert len(c.model.peaks) == 1 and len(c.pipeline) == 1
    assert panel.paste_setup(model=True) == 0                           # nothing left to change
    # removing a selection is one step as well
    n_undo = w.state.undo.count()
    panel._remove()
    assert w.state.project.datasets == [] and w.state.undo.count() == n_undo + 1
    w.state.undo.undo()
    assert w.state.project.datasets == [a, b, c]


def test_view_survives_saving_and_reopening(app, win, tmp_path):
    w = win
    a, b, c = w.state.project.datasets
    w.state.set_shown([b.id, c.id])
    w.state.set_color([b.id], "#00aa00")
    w.state.set_view(stacked=True, gap=0.4, normalize="range")
    w.state.remove_dataset(c)
    w.state.project.path = tmp_path / "p.ezspec"
    assert w.save_project()
    assert ViewState.from_dict(Project.load(tmp_path / "p.ezspec").meta["view"]).shown == [b.id]
    assert c.id in w.state.view.shown                                   # still there if the removal is undone
    w.open_project(str(tmp_path / "p.ezspec"))
    pump(app)
    v = w.state.view
    assert v.shown == [b.id] and v.stacked and v.gap == 0.4 and v.normalize == "range"
    assert w.state.color_of(w.state.project.get(b.id)) == "#00aa00"
    assert w.plot.act_stack.isChecked() and w.plot.gap_spin.value() == 40
    assert w.plot.norm_actions["range"].isChecked()
    assert list(w.plot.layer.overlay_items) == [b.id]
    w.new_project()
    pump(app)
    assert w.state.view == ViewState() and not w.plot.act_stack.isChecked()


def test_previews_are_compared_with_a_ticked_reference(app, win, tmp_path):
    w = win
    a, b, c = w.state.project.datasets
    folder = tmp_path / "series"
    folder.mkdir()
    for name, h in (("one", 5.0), ("two", 7.0)):
        s = spectrum(height=h)
        (folder / f"{name}.csv").write_text("x,y\n" + "\n".join(f"{u:.8g},{v:.8g}" for u, v in zip(s.x, s.y)))
    w.state.set_shown([a.id])
    w.preview_file(folder / "one.csv")
    wait_for(app, lambda: w.state.preview is not None, "preview")
    pv = w.state.preview
    assert w.state.current() is pv and set(w.plot.layer.overlay_items) == {a.id}
    assert w.plot.layer.c_data.getData()[1].max() == pytest.approx(5.0)
    assert w.state.color_of(pv) not in (w.state.color_of(a),) and pv.id not in w.state.view.colors
    # opening files from the Explorer straight into the overlay
    w.explorer.overlayRequested.emit([str(folder / "one.csv"), str(folder / "two.csv")])
    wait_for(app, lambda: len(w.state.project.datasets) == 5 and not w._io_tasks, "open")
    one, two = w.state.project.datasets[3:]
    assert w.state.preview is None and {one.id, two.id} <= set(w.state.view.shown)
    assert set(w.plot.layer.overlay_items) == {a.id, two.id} and w.state.current() is one


# ============================================================================ zoom regions
def panel_pos(w, plot, x, y):
    return w.plot.glw.mapFromScene(plot.getViewBox().mapViewToScene(QtCore.QPointF(x, y)))


def test_zoom_tool_shows_a_region_next_to_the_overview(app, win):
    w = win
    a, b, c = w.state.project.datasets
    w.state.set_shown([a.id, b.id])
    w.state.set_current(c.id)                                          # peak of height 3 at 160
    w.state.undo.setClean()
    pump(app)
    assert not w.plot._details and not w.plot.act_clear_zoom.isEnabled()

    # drag with the zoom tool over 130 … 145 (the peak of a, height 1)
    w.plot.set_mode("zoom")
    view = w.plot.glw.viewport()
    start, end = panel_pos(w, w.plot.p_main, 130.0, 1.0), panel_pos(w, w.plot.p_main, 145.0, 1.5)
    QtTest.QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, start)
    for k in range(1, 6):
        QtTest.QTest.mouseMove(view, start + (end - start) * k / 5)
        pump(app, 1)
    QtTest.QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, end)
    pump(app)
    assert len(w.state.view.regions) == 1
    assert w.state.view.regions[0] == pytest.approx([130.0, 145.0], abs=1.0)      # pixel grid
    assert w.state.undo.isClean()                                      # a view, not an edit
    d = w.plot._details[0]
    lo, hi = w.state.view.regions[0]
    assert d.plot.getViewBox().viewRange()[0] == pytest.approx([lo, hi])
    assert d.band.getRegion() == pytest.approx((lo, hi))
    # the panel draws the same scene and scales y to what is inside the range (a: 1, not c: 3)
    assert d.layer.c_data.yData.max() == pytest.approx(3.0)
    assert d.layer.c_data.getData()[0].max() < 150.0                   # … drawn only where it is looked at
    assert set(d.layer.overlay_items) == {a.id, b.id}
    pump(app, 20)
    assert 1.0 <= d.plot.getViewBox().viewRange()[1][1] < 2.0
    assert w.plot.p_main.getViewBox().viewRange()[1][1] > 3.0
    assert w.plot.act_clear_zoom.isEnabled()


def test_zoom_panels_have_working_tools(app, win):
    w = win
    a, b, c = w.state.project.datasets
    w.plot.regionsEdited.emit([[130.0, 150.0], [155.0, 165.0]])
    pump(app)
    assert w.state.view.regions == [[130.0, 150.0], [155.0, 165.0]] and len(w.plot._details) == 2
    first, second = w.plot._details
    assert first.color != second.color

    # the peak tool works inside an enlarged panel
    w.plot.set_mode("peak")
    pos = panel_pos(w, first.plot, 140.0, 1.0)
    QtTest.QTest.mouseMove(w.plot.glw.viewport(), pos)
    QtTest.QTest.mouseClick(w.plot.glw.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pos)
    pump(app)
    assert len(a.model.peaks) == 1
    assert a.model.peaks[0].derived(a.model.initial_values())["center"] == pytest.approx(140.0, abs=0.5)
    # every panel has its own handles (top and width), and dragging one of a zoom panel edits the model
    assert [it.panel for it in w.plot._peak_items] == [w.plot.p_main, w.plot.p_main, first.plot, first.plot,
                                                       second.plot, second.plot]
    handle = w.plot._peak_items[2]
    handle.setPos(141.0, 0.8)
    handle.sigPositionChangeFinished.emit(handle)
    pump(app)
    d = a.model.peaks[0].derived(a.model.initial_values())
    assert d["center"] == pytest.approx(141.0) and d["height"] == pytest.approx(0.8, abs=1e-6)
    assert w.plot._peak_items[0].pos().x() == pytest.approx(141.0)      # the overview follows

    # bands belong to the zoom tool only
    assert not first.band.movable
    w.plot.set_mode("zoom")
    assert first.band.movable
    first.band.setRegion((120.0, 135.0))                               # dragged in the overview
    assert first.plot.getViewBox().viewRange()[0] == pytest.approx([120.0, 135.0])     # live
    first.band.sigRegionChangeFinished.emit(first.band)
    pump(app)
    assert w.state.view.regions == [[120.0, 135.0], [155.0, 165.0]]
    # panning inside the enlarged panel moves its band
    vb = second.plot.getViewBox()
    vb.setXRange(157.0, 167.0, padding=0)
    vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"][:])
    assert second.band.getRegion() == pytest.approx((157.0, 167.0))
    wait_for(app, lambda: w.state.view.regions[1] == pytest.approx([157.0, 167.0]), "region update")
    # a right-click on a band (a real one) removes its region; the others keep their place
    right_click(app, w, w.plot.p_main, 127.0, 0.5)
    assert w.state.view.regions == [pytest.approx([157.0, 167.0])] and len(w.plot._details) == 1
    assert [it.panel for it in w.plot._peak_items] == [w.plot.p_main] * 2 + [w.plot._details[0].plot] * 2


def right_click(app, w, plot, x, y=None, choose=None):
    """A real right-click at the data position (x, y) of a panel (``y=None``: at half height).
    ``choose``: index of the entry to pick in the context menu that opens; returns the texts of
    its entries."""
    seen = []
    if y is None:
        y = sum(plot.getViewBox().viewRange()[1]) / 2

    def pick():
        menus = [m for m in QtWidgets.QApplication.topLevelWidgets()
                 if isinstance(m, QtWidgets.QMenu) and m.isVisible()]
        for menu in menus:
            seen.extend(a.text() for a in menu.actions())
            if choose is not None:
                menu.actions()[choose].trigger()
            menu.close()
    pos = panel_pos(w, plot, x, y)
    QtTest.QTest.mouseMove(w.plot.glw.viewport(), pos)
    pump(app, 2)
    QtCore.QTimer.singleShot(150, pick)
    QtTest.QTest.mouseClick(w.plot.glw.viewport(), QtCore.Qt.RightButton, QtCore.Qt.NoModifier, pos)
    for _ in range(60):
        pump(app, 2)
        if seen:
            break
    return seen


def test_zoom_regions_are_removed_with_the_mouse(app, win):
    w = win
    a, b, c = w.state.project.datasets
    regions = [[130.0, 150.0], [155.0, 165.0], [170.0, 180.0]]
    w.plot.regionsEdited.emit(regions)
    pump(app)
    # zoom tool: a right-click on the band or on its panel removes the region at once
    w.plot.set_mode("zoom")
    right_click(app, w, w.plot.p_main, 140.0, 0.5)
    assert w.state.view.regions == regions[1:]
    right_click(app, w, w.plot._details[0].plot, 160.0)
    assert w.state.view.regions == regions[2:]
    # outside a band the plot keeps its own menu, nothing is removed
    w.plot.regionsEdited.emit(regions)
    pump(app)
    entries = right_click(app, w, w.plot.p_main, 110.0, 0.5)
    assert entries and not any("zoom" in e.lower() for e in entries) and w.state.view.regions == regions

    # with another tool the band and the panel offer the removal in a menu
    for mode in ("navigate", "peak"):
        w.plot.regionsEdited.emit(regions)
        w.plot.set_mode(mode)
        pump(app)
        entries = right_click(app, w, w.plot.p_main, 140.0, 0.5, choose=0)
        assert entries == ["Remove zoom region 130 – 150", "Remove all zoom regions"]
        assert w.state.view.regions == regions[1:] and not a.model.peaks
        right_click(app, w, w.plot._details[1].plot, 175.0, choose=1)
        assert w.state.view.regions == [] and not w.plot._details
    # the × of a panel
    w.plot.regionsEdited.emit(regions[:2])
    pump(app)
    d = w.plot._details[0]
    centre = w.plot.glw.mapFromScene(d.close.mapToScene(d.close.boundingRect().center()))
    QtTest.QTest.mouseMove(w.plot.glw.viewport(), centre)
    QtTest.QTest.mouseClick(w.plot.glw.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, centre)
    pump(app, 12)
    assert w.state.view.regions == regions[1:2] and not a.model.peaks       # … and no peak was placed

    # the same routing makes the range tool's right-click work (fit range, masks)
    w.plot.regionsEdited.emit([])
    w.plot.rangeEdited.emit("fit_range", [120.0, 180.0])
    pump(app)
    assert a.fit_options.x_range == [120.0, 180.0]
    w.plot.set_mode("navigate")
    assert right_click(app, w, w.plot.p_main, 150.0, 0.5)                   # the plot's own menu
    assert a.fit_options.x_range == [120.0, 180.0]                          # not while navigating
    w.plot.set_mode("region")
    right_click(app, w, w.plot.p_main, 150.0, 0.5)
    assert a.fit_options.x_range is None


def test_zoom_regions_are_kept_with_the_project(app, win, tmp_path):
    w = win
    w.plot.regionsEdited.emit([[130.0, 150.0], [155.0, 165.0], [170.0, 180.0], [110.0, 120.0]])
    pump(app)
    assert len(w.plot._details) == 4
    w.state.project.path = tmp_path / "z.ezspec"
    assert w.save_project()
    w.plot.act_clear_zoom.trigger()
    pump(app)
    assert w.state.view.regions == [] and not w.plot._details and not w.plot.act_clear_zoom.isEnabled()
    w.open_project(str(tmp_path / "z.ezspec"))
    pump(app)
    assert len(w.state.view.regions) == 4 and len(w.plot._details) == 4
    assert w.plot._details[2].plot.getViewBox().viewRange()[0] == pytest.approx([170.0, 180.0])
    w.new_project()
    pump(app)
    assert not w.plot._details
