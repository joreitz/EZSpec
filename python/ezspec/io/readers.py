"""Built-in file readers and the format-independent entry points.

The formats themselves live in the reader registry (:mod:`ezspec.io.registry`);
this module registers text/CSV and JCAMP-DX and dispatches to whichever
reader fits a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spectrum import SigmaSource, Spectrum
from .jcamp import map_x_unit, parse_jcamp
from .registry import Reader, find_reader, get_reader, list_readers, register_reader, source_meta
from .text import decode_bytes, guess_unit, parse_table

_TEXT_EXT = (".txt", ".csv", ".dat", ".tsv", ".xy", ".asc", ".prn")
_JCAMP_EXT = (".jdx", ".dx", ".jcm", ".jcamp")


def read_text(path: Path, raw: bytes, x_col: int = 0, y_cols=None, sigma_col: int | None = None,
              delimiter="auto", decimal="auto", x_unit: str | None = None,
              extra_cols: dict | None = None) -> list:
    """Delimited text. Every column in ``y_cols`` (default: all columns except
    x and sigma) becomes one spectrum. ``extra_cols`` maps a variable name to
    a column used as an additional independent variable (stored as
    ``aux['var:<name>']``)."""
    text, _ = decode_bytes(raw)
    info = parse_table(text, delimiter=delimiter, decimal=decimal)
    data = info.data
    ncol = data.shape[1]
    if ncol == 1:
        data = np.column_stack([np.arange(len(data), dtype=float), data[:, 0]])
        info.header = ["Index"] + info.header
        ncol = 2
    if y_cols is None:
        skip = {x_col} | ({sigma_col} if sigma_col is not None else set()) | set((extra_cols or {}).values())
        y_cols = [c for c in range(ncol) if c not in skip]
    out = []
    xname = info.header[x_col]
    for c in y_cols:
        opts = {"x_col": x_col, "y_col": c, "sigma_col": sigma_col, "delimiter": info.delimiter,
                "decimal": info.decimal, "extra_cols": extra_cols}
        sigma = data[:, sigma_col] if sigma_col is not None else None
        aux = {f"var:{k}": data[:, v] for k, v in (extra_cols or {}).items()}
        aux["acq_index"] = np.arange(len(data), dtype=float)     # acquisition order (rows of the file)
        good = np.isfinite(data[:, x_col]) & np.isfinite(data[:, c])
        if sigma is not None:
            good &= np.isfinite(sigma) & (sigma > 0)
        name = info.header[c] if len(y_cols) > 1 else path.stem
        meta = {"name": name, "source": source_meta(path, raw, "text", opts), "columns": info.header}
        s = Spectrum(data[good, x_col], data[good, c], None if sigma is None else sigma[good],
                     SigmaSource.KNOWN if sigma is not None else SigmaSource.UNKNOWN,
                     x_unit=x_unit if x_unit is not None else guess_unit(xname), x_label=xname,
                     y_label=info.header[c], meta=meta,
                     aux={k: v[good] for k, v in aux.items()}).sorted()
        out.append(s)
    return out


def read_jcamp(path: Path, raw: bytes, x_unit: str | None = None) -> list:
    """JCAMP-DX (AFFN and ASDF compression, one XYDATA/XYPOINTS block)."""
    text, _ = decode_bytes(raw)
    d = parse_jcamp(text)
    unit = map_x_unit(d["x_units"])
    x = d["x"]
    if unit == "um":
        x, unit = x * 1000.0, "nm"
    meta = {"name": d["title"] or path.stem, "source": source_meta(path, raw, "jcamp", {}),
            "jcamp": {k: v for k, v in d["labels"].items() if len(v) < 200},
            "import_warnings": d["warnings"]}
    s = Spectrum(x, d["y"], x_unit=x_unit or unit, y_unit=d["y_units"].lower(),
                 x_label=d["x_units"] or "x", y_label=d["y_units"] or "y", meta=meta,
                 aux={"acq_index": np.arange(len(x), dtype=float)}).sorted()
    return [s]


def _looks_like_jcamp(_path: Path, raw: bytes) -> bool:
    return decode_bytes(raw[:4096])[0].lstrip().startswith("##")


register_reader(Reader("text", "Text/CSV", _TEXT_EXT, read_text, priority=-100,
                       options=("x_col", "y_cols", "sigma_col", "delimiter", "decimal", "x_unit", "extra_cols")),
                replace=True)
register_reader(Reader("jcamp", "JCAMP-DX", _JCAMP_EXT, read_jcamp, detect=_looks_like_jcamp,
                       options=("x_unit",)), replace=True)

# kept for backwards compatibility: name -> file-dialog filter of the built-in formats
FORMATS = {
    "text": "Text/CSV (*.txt *.csv *.dat *.tsv *.xy *.asc *.prn)",
    "jcamp": "JCAMP-DX (*.jdx *.dx *.jcm)",
}


def sniff(path) -> dict:
    """Detect the format and, for text, the table layout (used by the import dialog)."""
    path = Path(path)
    raw = path.read_bytes()
    reader = find_reader(path, raw)
    text, enc = decode_bytes(raw)
    info = {"format": reader.name, "title": reader.title, "encoding": enc}
    if reader.name == "text":
        info["table"] = parse_table(text)
    return info


def read_bytes(path, raw: bytes, reader: str | None = None, **options) -> list:
    """Read the spectra of a file whose content ``raw`` is already in memory.

    ``reader`` forces a format (default: detected from extension and content).
    Options another format understands are ignored, so one set of import
    settings can be applied to a mixed folder; an option no reader knows is
    an error."""
    path = Path(path)
    rd = get_reader(reader) if reader else find_reader(path, raw)
    known = {o for r in list_readers() for o in r.options}
    unknown = sorted(set(options) - known)
    if unknown:
        raise TypeError(f"unknown import option(s): {', '.join(unknown)}")
    specs = list(rd.read(path, raw, **{k: v for k, v in options.items() if k in rd.options}))
    if not specs:
        raise ValueError(f"{path.name}: the {rd.title} reader found no spectrum")
    return [_with_index(s, i) for i, s in enumerate(specs)]


def _with_index(s: Spectrum, i: int) -> Spectrum:
    """Record the position of a spectrum within its file (``read_file(path, index=i)``)."""
    src = s.meta.get("source")
    if not isinstance(src, dict) or "index" in src:
        return s
    return s.replace(meta={**s.meta, "source": {**src, "index": i}})


def read_spectra(path, reader: str | None = None, **options) -> list:
    """Read one or several spectra from a file of any registered format.

    For text tables the options are ``x_col``, ``y_cols``, ``sigma_col``,
    ``delimiter``, ``decimal``, ``x_unit`` and ``extra_cols`` (see
    :func:`read_text`); every y column becomes one spectrum."""
    path = Path(path)
    return read_bytes(path, path.read_bytes(), reader, **options)


def read_file(path, index: int = 0, **kw) -> Spectrum:
    """Read a single spectrum (``index`` selects among several, e.g. several y columns)."""
    return read_spectra(path, **kw)[index]
