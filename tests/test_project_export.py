import json
import subprocess
import sys

import numpy as np
import pytest

from ezspec import add_peak, lineshapes as ls
from ezspec.export.figure import (curves_for, default_spec, make_resolver, new_spec, param_box_lines,
                                  render_figure, save_figure, set_path)
from ezspec.export.results import write_all
from ezspec.export.script import generate_script
from ezspec.project import Dataset, Project


@pytest.fixture
def raw_file(tmp_path):
    rng = np.random.default_rng(5)
    x = np.linspace(400, 700, 900)
    y = 50 + 0.05 * (x - 400) + ls.gaussian(x, 900, 520, 12) + ls.lorentzian(x, 500, 580, 9)
    y = y + rng.normal(scale=0.8, size=x.size)
    p = tmp_path / "spectrum.csv"
    p.write_text("Wavelength (nm);Intensity\n" + "\n".join(f"{a:.4f};{b:.6f}".replace(".", ",")
                                                         for a, b in zip(x, y)))
    return p


def build_dataset(raw_file):
    ds = Dataset.from_file(raw_file)
    ds.pipeline.add("crop", {"xmin": 420, "xmax": 680})
    ds.pipeline.add("baseline_polynomial", {"order": 1, "ranges": [[420, 470], [640, 680]]})
    ds.pipeline.add("estimate_noise", {"method": "region", "region": [425, 465]})
    add_peak(ds.model, "gaussian", 520, 30, 15)
    add_peak(ds.model, "lorentzian", 580, 30, 10)
    ds.run_fit()
    return ds


def test_decimal_comma_import_and_units(raw_file):
    ds = Dataset.from_file(raw_file)
    assert ds.raw.x_unit == "nm" and ds.raw.n == 900 and ds.raw_bytes == raw_file.read_bytes()


def test_project_roundtrip(raw_file, tmp_path):
    ds = build_dataset(raw_file)
    proj = Project()
    proj.add(ds)
    spec = default_spec(ds.id, curves_for(ds.processed, ds.raw, ds.fit_result), "λ (nm)", "I")
    proj.figures["fig1"] = spec
    path = tmp_path / "p.ezspec"
    proj.save(path)
    back = Project.load(path)
    d2 = back.datasets[0]
    np.testing.assert_array_equal(d2.raw.x, ds.raw.x)
    np.testing.assert_array_equal(d2.raw.y, ds.raw.y)
    assert d2.raw_bytes == raw_file.read_bytes()
    assert [s.op for s in d2.pipeline] == ["crop", "baseline_polynomial", "estimate_noise"]
    assert d2.fit_record["statistics"]["sigma_source"] == "region"
    r2 = d2.run_fit()
    assert r2.values == ds.fit_result.values           # bit-identical recomputation
    assert back.figures["fig1"]["panels"][0]["traces"][0]["curve"] == "processed"


def test_figure_rendering_is_deterministic(raw_file, tmp_path):
    ds = build_dataset(raw_file)
    curves = curves_for(ds.processed, ds.raw, ds.fit_result)
    spec = default_spec(ds.id, curves, r"$\lambda$ (nm)", "Intensität", preset="acs_1")
    spec["panels"][0]["param_box"] = {"lines": param_box_lines(ds.fit_result, ["p1_center", "p1.height"])}
    spec["panels"][0]["secondary_x"] = {"from": "nm", "to": "eV"}
    set_path(spec, "panels.0.legend", "upper left")
    fig = render_figure(spec, make_resolver({ds.id: curves}))
    w, h = fig.get_size_inches()
    assert w == pytest.approx(82.55 / 25.4)
    outs = []
    for i in range(2):
        for ext in ("pdf", "svg", "png"):
            p = tmp_path / f"f{i}.{ext}"
            save_figure(spec, make_resolver({ds.id: curves}), p)
            outs.append(p.read_bytes())
    assert outs[0] == outs[3] and outs[1] == outs[4]           # PDF and SVG byte-identical
    assert b"DejaVu" in outs[0]                                 # TrueType font embedded (fonttype 42)


def test_result_tables(raw_file, tmp_path):
    ds = build_dataset(raw_file)
    files = write_all(ds.fit_result, tmp_path / "res")
    assert len(files) == 5 and all(f.exists() for f in files)
    text = (tmp_path / "res_parameters.csv").read_text()
    assert "p1_area" in text and "p1.height" in text
    json.loads((tmp_path / "res_fit.json").read_text())


def test_generated_script_reproduces_fit_bit_identically(raw_file, tmp_path):
    ds = build_dataset(raw_file)
    curves = curves_for(ds.processed, ds.raw, ds.fit_result)
    spec = default_spec(ds.id, curves, "λ (nm)", "I")
    code = generate_script(ds, raw_path=str(raw_file), figure_spec=spec, out_stem=str(tmp_path / "out"))
    script = tmp_path / "analysis.py"
    script.write_text(code)
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path, timeout=300)
    assert res.returncode == 0, res.stderr
    out = json.loads((tmp_path / "out_fit.json").read_text())
    for name, p in ds.fit_result.params.items():
        assert out["params"][name]["value"] == p.value               # exact equality
        assert out["params"][name]["stderr"] == p.stderr
    assert out["statistics"]["chi2"] == ds.fit_result.stats.chi2
    # the script's figure equals a figure rendered in-process
    save_figure(spec, make_resolver({ds.id: curves}), tmp_path / "ref.svg")
    assert (tmp_path / "out.svg").read_bytes() == (tmp_path / "ref.svg").read_bytes()
    # tampered raw data are refused
    bad = tmp_path / "bad.csv"
    bad.write_bytes(raw_file.read_bytes() + b"\n")
    res = subprocess.run([sys.executable, str(script), str(bad)], capture_output=True, text=True, timeout=300)
    assert res.returncode != 0 and "SHA-256" in res.stderr


def test_new_spec_presets():
    s = new_spec("nature_2")
    assert s["width_mm"] == 183 and s["font_size"] == 7
