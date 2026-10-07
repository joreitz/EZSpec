import numpy as np
import pytest

from ezspec import Model, Spectrum
from ezspec.export.figure import curves_for
from ezspec.fit import fit
from ezspec.io import read_spectra
from ezspec.ops import average_sweeps, select_sweeps
from ezspec.spectrum import SigmaSource
from ezspec.sweeps import count_sweeps, display_order, find_sweeps, is_multivalued


def triangle(periods=3, n_ramp=150, x_noise=0.0, shift=0.0, noise=0.01, seed=0):
    """Current ramp up and down; the absorption line appears shifted by `shift`
    on the down ramp (thermal hysteresis). Returns (Spectrum sorted by x, raw arrays)."""
    rng = np.random.default_rng(seed)
    up = np.linspace(10.0, 50.0, n_ramp)
    xs, ys, dirs = [], [], []
    for _ in range(periods):
        for d, ramp in ((1, up), (-1, up[::-1])):
            x = ramp + rng.normal(scale=x_noise, size=n_ramp) if x_noise else ramp.copy()
            c = 30.0 + (shift if d < 0 else 0.0)
            y = 1.0 - 0.5 * np.exp(-0.5 * ((x - c) / 2.0) ** 2) + rng.normal(scale=noise, size=n_ramp)
            xs.append(x)
            ys.append(y)
            dirs.append(np.full(n_ramp, d))
    x, y, d = np.concatenate(xs), np.concatenate(ys), np.concatenate(dirs)
    s = Spectrum(x, y, aux={"acq_index": np.arange(len(x), dtype=float)}).sorted()
    return s, (x, y, d)


def test_sweep_detection_robust_to_x_noise():
    s, _ = triangle(periods=3, x_noise=0.15)
    c = count_sweeps(s)
    assert (c["up"], c["down"]) == (3, 3)
    assert is_multivalued(s)
    for w in find_sweeps(s):
        if w.direction:
            assert w.n >= 140          # turning points within a few samples


def test_single_ramp_is_not_multivalued():
    s, _ = triangle(periods=1)
    up = select_sweeps(s, direction="up", sweep=-1)
    assert not is_multivalued(up) and display_order(up) is None
    plain = Spectrum(np.arange(10.0), np.arange(10.0))
    assert not is_multivalued(plain)


def test_display_order_is_acquisition_order():
    s, (x, y, _) = triangle(periods=2)
    o = display_order(s)
    np.testing.assert_array_equal(s.x[o], x)
    np.testing.assert_array_equal(s.y[o], y)
    c = curves_for(s)
    np.testing.assert_array_equal(c["processed"][0], x)


def test_select_sweeps_keeps_directions_apart():
    s, (x, y, d) = triangle(periods=3, shift=1.5)
    up = select_sweeps(s, direction="up", sweep=-1)
    assert up.n == np.sum(d == 1)
    np.testing.assert_array_equal(np.sort(up.y), np.sort(y[d == 1]))
    assert any("sweeps selected" in w for w in up.meta["_step_warnings"])
    second_down = select_sweeps(s, direction="down", sweep=1)
    assert second_down.n == 150 and not is_multivalued(second_down)
    np.testing.assert_array_equal(np.sort(second_down.y), np.sort(y[3 * 150:4 * 150]))
    with pytest.raises(ValueError, match="no matching"):
        select_sweeps(s, direction="down", sweep=7)


def test_average_sweeps_identical_grid_standard_error():
    s, (x, y, d) = triangle(periods=4, noise=0.02, seed=3)
    out = average_sweeps(s, direction="down", sigma="repeats")
    Y = np.vstack([y[(2 * k + 1) * 150:(2 * k + 2) * 150][::-1] for k in range(4)])
    np.testing.assert_allclose(out.x, np.linspace(10, 50, 150))
    np.testing.assert_allclose(out.y, Y.mean(axis=0))
    np.testing.assert_allclose(out.sigma, Y.std(axis=0, ddof=1) / 2.0)
    assert out.sigma_source is SigmaSource.REPEATS and out.sigma_source.is_known
    assert "interpolated" not in out.flags and "acq_index" not in out.aux
    assert any("repeats" in w for w in out.meta["_step_warnings"])    # n = 4 < 5


def test_average_sweeps_interpolates_noisy_grids():
    s, _ = triangle(periods=2, x_noise=0.02)
    out = average_sweeps(s, direction="up", sigma="repeats")
    assert "interpolated" in out.flags and out.is_strictly_increasing


def test_reader_stores_acquisition_order(tmp_path):
    _, (x, y, _) = triangle(periods=2)
    p = tmp_path / "ramp.txt"
    np.savetxt(p, np.column_stack([x, y]), header="I_mA signal", comments="")
    s = read_spectra(p)[0]
    assert "acq_index" in s.aux and np.all(np.diff(s.x) >= 0)
    assert is_multivalued(s)
    np.testing.assert_allclose(s.x[display_order(s)], x)


def test_fit_warns_on_multiple_sweeps():
    s, _ = triangle(periods=2, shift=1.0)
    m = Model()
    m.add("formula", options={"expression": "b - a*exp(-0.5*((x-c)/w)**2)",
                              "defaults": {"b": 1, "a": 0.5, "c": 30, "w": 2}})
    r = fit(s, m)
    assert "MULTI_SWEEP" in [w.code for w in r.warnings]
    r1 = fit(select_sweeps(s, direction="up", sweep=0), m)
    assert "MULTI_SWEEP" not in [w.code for w in r1.warnings]
    assert abs(r1.value("c") - 30.0) < 0.1


def test_pipeline_warns_before_smoothing_mixed_ramps():
    from ezspec.pipeline import Pipeline
    s, _ = triangle(periods=2)
    p = Pipeline()
    sm = p.add("smooth_moving_average", {"half_window": 2})
    run = p.run(s)
    assert any("multiple sweeps" in w for w in run.result(sm.id).warnings)
    q = Pipeline()
    q.add("select_sweeps", {"direction": "up", "sweep": 0})
    sm2 = q.add("smooth_moving_average", {"half_window": 2})
    run2 = q.run(s)
    assert not any("multiple sweeps" in w for w in run2.result(sm2.id).warnings)
    assert run2.final.n == 150
