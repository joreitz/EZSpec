"""Projects: datasets with their processing pipelines, models, fit results and figures.

Single-file format ``*.ezspec`` (a zip archive)::

    manifest.json            format version, app and package versions, dataset index
    raw/<sha256>.<ext>       original files, byte-identical
    data/<id>.npz            parsed raw arrays (x, y, sigma, exclude, aux)
    data/<id>.json           spectrum metadata (units, sigma source, flags, meta)
    pipelines/<id>.json      processing steps {id, op, op_version, params, inputs, created_at}
    models/<id>.json         model and fit options
    results/<id>.json        last fit result (statistics, parameters, warnings, provenance)
    figures/<name>.json      figure specifications

Everything except ``raw/`` and ``data/*.npz`` is plain JSON, so projects can
be inspected and diffed without the program.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fit import FitOptions, FitResult, fit
from .models import Model
from .pipeline import Pipeline, PipelineRun
from .spectrum import SigmaSource, Spectrum

FORMAT = "ezspec.project"
FORMAT_VERSION = 1


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else str(f)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def spectrum_to_bytes(s: Spectrum) -> tuple:
    arrays = {"x": s.x, "y": s.y}
    if s.sigma is not None:
        arrays["sigma"] = s.sigma
    if s.exclude is not None:
        arrays["exclude"] = s.exclude
    for k, v in s.aux.items():
        arrays[f"aux__{k}"] = v
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    meta = {"sigma_source": s.sigma_source.value, "x_unit": s.x_unit, "y_unit": s.y_unit,
            "x_label": s.x_label, "y_label": s.y_label, "meta": _jsonable(dict(s.meta)),
            "flags": sorted(s.flags)}
    return buf.getvalue(), meta


def spectrum_from_bytes(data: bytes, meta: dict) -> Spectrum:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    aux = {k[5:]: v for k, v in arrays.items() if k.startswith("aux__")}
    return Spectrum(arrays["x"], arrays["y"], arrays.get("sigma"), SigmaSource(meta["sigma_source"]),
                    meta.get("x_unit", ""), meta.get("y_unit", ""), meta.get("x_label", "x"),
                    meta.get("y_label", "y"), arrays.get("exclude"), meta.get("meta", {}),
                    frozenset(meta.get("flags", [])), aux)


@dataclass
class Dataset:
    name: str
    raw: Spectrum
    pipeline: Pipeline = field(default_factory=Pipeline)
    model: Model = field(default_factory=Model)
    fit_options: FitOptions = field(default_factory=FitOptions)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    raw_bytes: bytes | None = None
    raw_ext: str = ""
    notes: str = ""
    fit_result: FitResult | None = None
    fit_record: dict | None = None

    _run: PipelineRun | None = field(default=None, repr=False)

    @property
    def source(self) -> dict:
        return dict(self.raw.meta.get("source", {}))

    def run_pipeline(self) -> PipelineRun:
        self._run = self.pipeline.run(self.raw)
        return self._run

    @property
    def processed(self) -> Spectrum:
        return (self._run or self.run_pipeline()).final

    def run_fit(self) -> FitResult:
        result = fit(self.run_pipeline().final, self.model, self.fit_options)
        self.fit_result = result
        self.fit_record = result.to_dict(include_curves=False)
        return result

    @classmethod
    def from_bytes(cls, data: bytes, filename: str, index: int = 0, **reader_kw) -> "Dataset":
        """Create a dataset from file contents (e.g. generated example data)."""
        import tempfile
        from .io import read_spectra
        suffix = Path(filename).suffix
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / filename
            p.write_bytes(data)
            s = read_spectra(p, **reader_kw)[index]
        meta = dict(s.meta)
        src = dict(meta.get("source", {}))
        src["path"] = filename
        meta["source"] = src
        s = s.replace(meta=meta)
        return cls(name=s.meta.get("name", Path(filename).stem), raw=s, raw_bytes=data, raw_ext=suffix.lower())

    @classmethod
    def from_file(cls, path, index: int = 0, **reader_kw) -> "Dataset":
        from .io import read_spectra
        path = Path(path)
        specs = read_spectra(path, **reader_kw)
        s = specs[index]
        return cls(name=s.meta.get("name", path.stem), raw=s, raw_bytes=path.read_bytes(),
                   raw_ext=path.suffix.lower())


@dataclass
class Project:
    datasets: list = field(default_factory=list)
    figures: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    def add(self, ds: Dataset) -> Dataset:
        self.datasets.append(ds)
        return ds

    def get(self, ds_id: str) -> Dataset:
        for d in self.datasets:
            if d.id == ds_id:
                return d
        raise KeyError(ds_id)

    def remove(self, ds_id: str) -> Dataset:
        d = self.get(ds_id)
        self.datasets.remove(d)
        return d

    # ------------------------------------------------------------------ save / load
    def save(self, path) -> None:
        import lmfit
        import scipy

        from . import __version__
        from ._backend import backend_name
        path = Path(path)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self.meta.setdefault("created", now)
        self.meta["modified"] = now
        manifest = {
            "format": FORMAT, "format_version": FORMAT_VERSION, "meta": _jsonable(self.meta),
            "versions": {"ezspec": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
                         "lmfit": lmfit.__version__, "backend": backend_name()},
            "datasets": [], "figures": sorted(self.figures),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for d in self.datasets:
                entry = {"id": d.id, "name": d.name, "notes": d.notes}
                if d.raw_bytes is not None:
                    sha = d.source.get("sha256") or __import__("hashlib").sha256(d.raw_bytes).hexdigest()
                    name = f"raw/{sha}{d.raw_ext}"
                    if name not in z.namelist():
                        z.writestr(name, d.raw_bytes)
                    entry["raw_file"] = name
                data, smeta = spectrum_to_bytes(d.raw)
                z.writestr(f"data/{d.id}.npz", data)
                z.writestr(f"data/{d.id}.json", json.dumps(smeta, indent=1, ensure_ascii=False))
                z.writestr(f"pipelines/{d.id}.json", json.dumps(d.pipeline.to_dict(), indent=1,
                                                               ensure_ascii=False))
                z.writestr(f"models/{d.id}.json", json.dumps({"model": d.model.to_dict(),
                                                              "fit_options": d.fit_options.to_dict()},
                                                             indent=1, ensure_ascii=False))
                if d.fit_record is not None:
                    z.writestr(f"results/{d.id}.json", json.dumps(_jsonable(d.fit_record), indent=1,
                                                                 ensure_ascii=False))
                manifest["datasets"].append(entry)
            for name, spec in self.figures.items():
                z.writestr(f"figures/{name}.json", json.dumps(_jsonable(spec), indent=1, ensure_ascii=False))
            z.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
        tmp.replace(path)
        self.path = path

    @classmethod
    def load(cls, path) -> "Project":
        path = Path(path)
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
            if manifest.get("format") != FORMAT:
                raise ValueError("keine EZSpec-Projektdatei")
            if manifest.get("format_version", 0) > FORMAT_VERSION:
                raise ValueError("Projektdatei stammt aus einer neueren EZSpec-Version")
            proj = cls(meta=manifest.get("meta", {}), path=path)
            for e in manifest["datasets"]:
                i = e["id"]
                smeta = json.loads(z.read(f"data/{i}.json"))
                raw = spectrum_from_bytes(z.read(f"data/{i}.npz"), smeta)
                pipe = Pipeline.from_dict(json.loads(z.read(f"pipelines/{i}.json")))
                md = json.loads(z.read(f"models/{i}.json"))
                ds = Dataset(name=e["name"], raw=raw, pipeline=pipe, model=Model.from_dict(md["model"]),
                             fit_options=FitOptions.from_dict(md.get("fit_options")), id=i,
                             notes=e.get("notes", ""))
                if "raw_file" in e:
                    ds.raw_bytes = z.read(e["raw_file"])
                    ds.raw_ext = Path(e["raw_file"]).suffix
                if f"results/{i}.json" in z.namelist():
                    ds.fit_record = json.loads(z.read(f"results/{i}.json"))
                proj.datasets.append(ds)
            for name in manifest.get("figures", []):
                proj.figures[name] = json.loads(z.read(f"figures/{name}.json"))
        return proj


TEMPLATE_FORMAT = "ezspec.template"


def template_of(ds: Dataset) -> dict:
    """Pipeline, model and fit options of a dataset as a reusable template (x in physical units)."""
    return {"format": TEMPLATE_FORMAT, "version": 1, "pipeline": ds.pipeline.to_dict(),
            "model": ds.model.to_dict(), "fit_options": ds.fit_options.to_dict()}


def save_template(ds: Dataset, path) -> None:
    Path(path).write_text(json.dumps(_jsonable(template_of(ds)), indent=1, ensure_ascii=False), encoding="utf-8")


def load_template(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("format") != TEMPLATE_FORMAT:
        raise ValueError("keine EZSpec-Vorlage")
    Pipeline.from_dict(d["pipeline"])        # validate
    Model.from_dict(d["model"])
    return d
