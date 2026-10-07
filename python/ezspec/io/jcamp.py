"""JCAMP-DX reader (versions 4.24 / 5.x, single spectrum blocks).

Supports ``##XYDATA=(X++(Y..Y))`` in AFFN and ASDF compression (SQZ, DIF,
DUP, including the DIF y-check value repeated at the start of the following
line), ``##XYPOINTS=(XY..XY)`` and ``##PEAK TABLE=(XY..XY)``, with
XFACTOR/YFACTOR scaling. Abscissae of XYDATA are reconstructed from
FIRSTX, LASTX and NPOINTS (equidistant by definition of the format).

Reference: McDonald & Wilks, Appl. Spectrosc. 42 (1988) 151.
"""

from __future__ import annotations

import re

import numpy as np

SQZ = {"@": 0, **{c: i + 1 for i, c in enumerate("ABCDEFGHI")}, **{c: -(i + 1) for i, c in enumerate("abcdefghi")}}
DIF = {"%": 0, **{c: i + 1 for i, c in enumerate("JKLMNOPQR")}, **{c: -(i + 1) for i, c in enumerate("jklmnopqr")}}
DUP = {c: i + 1 for i, c in enumerate("STUVWXYZ")}
DUP["s"] = 9


class JcampError(ValueError):
    pass


_ASDF_CHARS = set("@ABCDFGHIabcdfghi%JKLMNOPQRjklmnopqrSTUVWXYZs")


def _tokenize(line: str, asdf: bool = False):
    """Split one data line into (kind, text) tokens; kind in AFFN, SQZ, DIF, DUP.

    ``E``/``e`` are SQZ digits (+5/-5) in compressed data but exponent markers
    in AFFN numbers. In ASDF mode an ``E`` directly followed by a digit is SQZ;
    followed by a sign it is an exponent.
    """
    tokens = []
    cur_kind, cur = None, ""

    def flush():
        nonlocal cur_kind, cur
        if cur_kind is not None and cur not in ("", "+", "-"):
            tokens.append((cur_kind, cur))
        cur_kind, cur = None, ""

    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in " \t,;":
            flush()
        elif c in "+-":
            # exponent sign inside an AFFN number
            if cur_kind == "AFFN" and cur and cur[-1] in "eE":
                cur += c
            else:
                flush()
                cur_kind, cur = "AFFN", c
        elif c.isdigit() or c == ".":
            if cur_kind is None:
                cur_kind, cur = "AFFN", ""
            cur += c
        elif (c in "eE" and cur_kind == "AFFN" and cur and i + 1 < n
              and (line[i + 1] in "+-" or (line[i + 1].isdigit() and not asdf))):
            cur += c
        elif c in SQZ:
            flush()
            cur_kind, cur = "SQZ", c
        elif c in DIF:
            flush()
            cur_kind, cur = "DIF", c
        elif c in DUP:
            flush()
            cur_kind, cur = "DUP", c
        elif c == "?":
            flush()
            tokens.append(("AFFN", "nan"))
        else:
            raise JcampError(f"unexpected character {c!r} in data line: {line[:60]!r}")
        i += 1
    flush()
    return tokens


def _value(kind, text):
    if kind == "AFFN":
        return float(text)
    table = SQZ if kind == "SQZ" else DIF
    head = table[text[0]]
    rest = text[1:]
    mag = float(f"{abs(head)}{rest}") if rest else float(abs(head))
    return -mag if head < 0 else mag


def decode_xydata(lines, warnings: list | None = None):
    """Decode (X++(Y..Y)) lines; returns (x_first_per_line, array of y)."""
    warnings = [] if warnings is None else warnings
    lines = [ln for ln in lines if ln.strip()]
    ys = []
    line_x = []
    prev_ended_dif = False
    asdf = any(ch in _ASDF_CHARS for line in lines for ch in line)
    for k, line in enumerate(lines):
        toks = _tokenize(line, asdf)
        if not toks:
            continue
        line_x.append(_value(*toks[0]) if toks[0][0] in ("AFFN", "SQZ") else np.nan)
        vals = []
        last_dif = False
        last_delta = 0.0
        for kind, text in toks[1:]:
            if kind in ("AFFN", "SQZ"):
                vals.append(_value(kind, text))
                last_dif = False
            elif kind == "DIF":
                d = _value(kind, text)
                base = vals[-1] if vals else (ys[-1] if ys else 0.0)
                vals.append(base + d)
                last_dif, last_delta = True, d
            else:  # DUP: total count of occurrences of the previous token
                count = DUP[text[0]]
                if len(text) > 1:
                    count = int(f"{count}{text[1:]}")
                for _ in range(count - 1):
                    base = vals[-1]
                    vals.append(base + last_delta if last_dif else base)
        if prev_ended_dif and vals:
            # y-check: first ordinate repeats the last ordinate of the previous line
            if ys and not np.isclose(vals[0], ys[-1], rtol=1e-9, atol=1e-9):
                if k == len(lines) - 1 and len(vals) == 1:
                    warnings.append(f"Last line with mismatching y-check value ({vals[0]:g} instead of "
                                    f"{ys[-1]:g}) ignored")
                else:
                    raise JcampError(f"y-check value mismatch (line {k + 1}): {vals[0]} != {ys[-1]}")
            vals = vals[1:]
        ys.extend(vals)
        prev_ended_dif = last_dif
    return np.array(line_x), np.array(ys, float)


def _decode_pairs(lines):
    nums = []
    for line in lines:
        for kind, text in _tokenize(line):
            nums.append(_value(kind, text))
    if len(nums) % 2:
        raise JcampError("odd number of values in (XY..XY) data")
    a = np.array(nums, float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


_LABEL = re.compile(r"^##\s*([^=]+)=(.*)$")


def parse_jcamp(text: str) -> dict:
    """Parse the first spectrum block. Returns dict with x, y, labels (dict), units."""
    labels = {}
    data_kind = None
    data_lines = []
    in_data = False
    extra_blocks = 0
    done = False
    for raw in text.splitlines():
        line = raw.split("$$", 1)[0].rstrip()
        m = _LABEL.match(line.strip())
        if m:
            key = re.sub(r"[\s\-_/]", "", m.group(1)).upper()
            val = m.group(2).strip()
            if in_data:
                in_data = False
                if key == "END":
                    break
            if key in ("XYDATA", "XYPOINTS", "PEAKTABLE"):
                if data_kind is not None:
                    extra_blocks += 1     # only the first data table is read
                    done = True
                    continue
                data_kind = (key, val.replace(" ", ""))
                in_data = True
                continue
            if key == "END" and data_kind is not None:
                done = True
            if not done:
                labels.setdefault(key, val)
            continue
        if in_data and line.strip() and not done:
            data_lines.append(line.strip())
    if data_kind is None:
        if "NTUPLES" in labels:
            raise JcampError("JCAMP NTUPLES (e.g. NMR FIDs, multiple spectra) are not supported yet")
        raise JcampError("no ##XYDATA, ##XYPOINTS or ##PEAK TABLE data found")

    def num(key, default=None):
        if key not in labels:
            return default
        try:
            return float(labels[key].replace(",", ".").split()[0])
        except (ValueError, IndexError):
            return default

    warnings = []
    if extra_blocks:
        warnings.append(f"File contains {extra_blocks + 1} data blocks; only the first was read")
    xf = num("XFACTOR", 1.0)
    yf = num("YFACTOR", 1.0)
    key, form = data_kind
    if key == "XYDATA" and form.upper().startswith("(XY..XY)"):
        key = "XYPOINTS"
    if key == "XYDATA":
        if not form.upper().startswith("(X++(Y..Y)"):
            raise JcampError(f"XYDATA form {form} not supported")
        line_x, y = decode_xydata(data_lines, warnings)
        y = y * yf
        npts = num("NPOINTS")
        first = num("FIRSTX")
        last = num("LASTX")
        if npts is not None and int(npts) != len(y):
            raise JcampError(f"NPOINTS = {int(npts)}, but {len(y)} values decoded")
        if first is not None and last is not None and len(y) > 1:
            x = np.linspace(first, last, len(y))
        else:
            dx = num("DELTAX")
            x0 = line_x[0] * xf
            x = x0 + dx * np.arange(len(y)) if dx else np.arange(len(y), dtype=float)
    else:
        x, y = _decode_pairs(data_lines)
        x, y = x * xf, y * yf
    return {"x": x, "y": y, "labels": labels, "x_units": labels.get("XUNITS", ""),
            "y_units": labels.get("YUNITS", ""), "title": labels.get("TITLE", ""), "warnings": warnings}


_UNIT_MAP = {"1/CM": "cm-1", "CM-1": "cm-1", "CM^-1": "cm-1", "NANOMETERS": "nm", "NM": "nm",
             "MICROMETERS": "um", "EV": "eV"}


def map_x_unit(units: str) -> str:
    return _UNIT_MAP.get(units.strip().upper(), units.strip())
