"""Generate a standalone Python script that reproduces an analysis from the raw data.

The script re-reads the raw file (checked by SHA-256), applies the recorded
pipeline step by step with the same operation calls the GUI made, fits the
same model with the same options, prints the report, writes the result JSON
and renders the publication figure. No GUI is required.
"""

from __future__ import annotations

import datetime as _dt
import json
import textwrap
from pathlib import Path

from .. import __version__


def _reader_call(raw) -> str:
    src = raw.meta.get("source", {})
    reader = src.get("reader")
    opts = src.get("options", {}) or {}
    if reader == "text":
        kw = {"x_col": opts.get("x_col", 0), "y_cols": [opts.get("y_col", 1)],
              "sigma_col": opts.get("sigma_col"), "delimiter": opts.get("delimiter"),
              "decimal": opts.get("decimal", ".")}
        if opts.get("extra_cols"):
            kw["extra_cols"] = opts["extra_cols"]
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        return f"read_file(RAW, {args})"
    return "read_file(RAW)"


def generate_script(dataset, raw_path=None, figure_spec: dict | None = None, out_stem: str | None = None,
                    extra_figures: dict | None = None) -> str:
    """Python source reproducing ``dataset`` (pipeline + model fit + figures).

    ``figure_spec`` is the main figure (written as ``<out_stem>.pdf/.svg``);
    ``extra_figures`` maps a file suffix to further figures, e.g. of the raw
    data or of a single processing step (``<out_stem>_<suffix>.pdf/.svg``)."""
    raw = dataset.raw
    src = raw.meta.get("source", {})
    if raw_path is None:
        raw_path = src.get("path") or src.get("filename")
    if not raw_path:
        raise ValueError("dataset has no raw data file – export it as a file first")
    sha = src.get("sha256", "")
    stem = out_stem or Path(str(raw_path)).stem + "_ezspec"
    lines = dataset.pipeline.script_lines("s")
    model_json = json.dumps(dataset.model.to_dict(), indent=1, ensure_ascii=False)
    opts = dataset.fit_options.to_dict()
    has_model = bool(dataset.model.components)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out = [
        "#!/usr/bin/env python3",
        f'"""EZSpec analysis “{dataset.name}” – generated {now} with EZSpec {__version__}.',
        "",
        "Reproduces processing, fit and figure from the raw data without the GUI.",
        "Usage:  python <this_script>.py [raw_data_file]",
        '"""',
        "import hashlib",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "from ezspec import ops",
        "from ezspec.io import read_file",
        "",
        f"RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path({str(raw_path)!r})",
        f"RAW_SHA256 = {sha!r}",
        f"OUT = Path({stem!r})",
        "",
        "if RAW_SHA256 and hashlib.sha256(RAW.read_bytes()).hexdigest() != RAW_SHA256:",
        '    raise SystemExit("Raw data differ from the analysed file (SHA-256 mismatch)")',
        "",
        f"raw = {_reader_call(raw)}",
        "s = raw",
        "",
        "# --- Processing (pipeline) " + "-" * 41,
    ]
    out += lines or ["# (no processing steps)"]
    out += ["", "print(s)"]
    if has_model:
        out += [
            "",
            "# --- Fit " + "-" * 58,
            "from ezspec.fit import FitOptions, fit",
            "from ezspec.models import Model",
            "",
            "model = Model.from_dict(json.loads(r'''",
            model_json,
            "'''))",
            f"options = FitOptions.from_dict({opts!r})",
            "result = fit(s, model, options)",
            "print(result.report())",
            'OUT.with_name(OUT.name + "_fit.json").write_text(result.to_json(), encoding="utf-8")',
        ]
    figures = {}
    if figure_spec is not None:
        figures[""] = figure_spec
    figures.update(extra_figures or {})
    if figures:
        # figures of intermediate stages (raw data, single steps) need every step's input and output
        stages = any(str(t.get("curve", "")).startswith(("raw:", "step:"))
                     for f in figures.values() for p in f.get("panels", []) for t in p.get("traces", []))
        fit_args = f"{'result' if has_model else 'None'}, {'model' if has_model else 'None'}"
        out += [
            "",
            "# --- Figures " + "-" * 54,
        ]
        if stages:
            out += [
                "from ezspec.export.figure import dataset_curves, make_resolver, save_figure",
                "from ezspec.pipeline import Pipeline",
                "",
                "# figures of intermediate stages: re-run the same pipeline, keeping the output of every step",
                "run = Pipeline.from_dict(json.loads(r'''",
                json.dumps(dataset.pipeline.to_dict(), indent=1, ensure_ascii=False),
                "''')).run(raw)",
            ]
        else:
            out += ["from ezspec.export.figure import curves_for, make_resolver, save_figure"]
        out += [
            "",
            "# file suffix -> figure specification ('' = main figure)",
            "figures = json.loads(r'''",
            json.dumps(figures, indent=1, ensure_ascii=False),
            "''')",
            (f"curves = {{{dataset.id!r}: dataset_curves(raw, run, {fit_args})}}" if stages else
             f"curves = {{{dataset.id!r}: curves_for(s, raw, {fit_args})}}"),
            "for suffix, spec in figures.items():",
            '    stem = OUT.name + ("_" + suffix if suffix else "")',
            '    save_figure(spec, make_resolver(curves), OUT.with_name(stem + ".pdf"))',
            '    save_figure(spec, make_resolver(curves), OUT.with_name(stem + ".svg"))',
        ]
    return "\n".join(out) + "\n"


def write_script(dataset, path, **kw) -> Path:
    path = Path(path)
    path.write_text(generate_script(dataset, **kw), encoding="utf-8")
    return path


def indent(code: str) -> str:
    return textwrap.indent(code, "    ")
