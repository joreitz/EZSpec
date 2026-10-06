import numpy as np
import pytest

from ezspec.io import parse_table, read_file, read_spectra
from ezspec.io.jcamp import JcampError, decode_xydata, parse_jcamp
from ezspec.spectrum import SigmaSource

# ----------------------------------------------------------------------------- text


@pytest.mark.parametrize("content,delim,dec", [
    ("x,y\n1.5,2.25\n2.5,3.5\n3.5,4.75\n", ",", "."),
    ("Wellenlänge;Intensität\n1,5;2,25\n2,5;3,5\n3,5;4,75\n", ";", ","),
    ("# comment\n1.5\t2.25\n2.5\t3.5\n3.5\t4.75\n", "\t", "."),
    ("  1.5   2.25\n  2.5   3.5\n  3.5   4.75\nend of data\n", None, "."),
    ("1,5 2,25\n2,5 3,5\n3,5 4,75\n", None, ","),
    ("x y\n1.5e0 2.25E+00\n2.5 3.5\n3.5 4.75\n", None, "."),
])
def test_table_detection(content, delim, dec):
    info = parse_table(content)
    assert info.delimiter == delim and info.decimal == dec
    np.testing.assert_allclose(info.data, [[1.5, 2.25], [2.5, 3.5], [3.5, 4.75]])


def test_read_multi_column_with_sigma_and_units(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("Raman shift (cm-1),I1,I2,sigma\n300,1,10,0.5\n100,2,20,0.5\n200,3,30,0.5\n")
    specs = read_spectra(p, sigma_col=3)
    assert len(specs) == 2
    s = specs[1]
    assert list(s.x) == [100, 200, 300] and list(s.y) == [20, 30, 10]
    assert s.sigma_source is SigmaSource.KNOWN and s.x_unit == "raman"
    assert s.meta["source"]["sha256"] and s.meta["name"] == "I2"
    one = read_file(p, y_cols=[1])
    assert one.sigma is None


def test_extra_independent_columns(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("x1 x2 y\n1 10 5\n2 20 7\n3 30 9\n")
    s = read_file(p, x_col=0, y_cols=[2], extra_cols={"x2": 1})
    assert list(s.aux["var:x2"]) == [10, 20, 30]


# ----------------------------------------------------------------------------- JCAMP
SQZ_POS = "@ABCDEFGHI"
SQZ_NEG = "@abcdefghi"
DIF_POS = "%JKLMNOPQR"
DIF_NEG = "%jklmnopqr"
DUP = "STUVWXYZs"


def sqz(v: int) -> str:
    s = str(abs(v))
    return (SQZ_POS if v >= 0 else SQZ_NEG)[int(s[0])] + s[1:]


def dif(v: int) -> str:
    s = str(abs(v))
    return (DIF_POS if v >= 0 else DIF_NEG)[int(s[0])] + s[1:]


def dup(n: int) -> str:
    s = str(n)
    return DUP[int(s[0]) - 1] + s[1:]


def encode_difdup(xs, ys, per_line=10):
    """Reference encoder following JCAMP-DX 4.24 (DIF with DUP and y-check)."""
    lines = []
    i = 0
    n = len(ys)
    while i < n:
        chunk = list(range(i, min(i + per_line, n)))
        out = f"{xs[chunk[0]]:g}" + sqz(ys[chunk[0]])
        diffs = [ys[k] - ys[k - 1] for k in chunk[1:]]
        j = 0
        while j < len(diffs):
            run = 1
            while j + run < len(diffs) and diffs[j + run] == diffs[j]:
                run += 1
            out += dif(diffs[j]) + (dup(run) if run > 1 else "")
            j += run
        lines.append(out)
        i = chunk[-1]       # last value repeated as y-check on next line
        if i == n - 1:
            break
    return lines


def jcamp_text(lines, n, first, last, yfactor=1.0, form="(X++(Y..Y))"):
    head = ["##TITLE=synthetic", "##JCAMP-DX=4.24", "##DATA TYPE=INFRARED SPECTRUM", "##XUNITS=1/CM",
            "##YUNITS=ABSORBANCE", f"##FIRSTX={first}", f"##LASTX={last}", "##XFACTOR=1",
            f"##YFACTOR={yfactor}", f"##NPOINTS={n}", f"##XYDATA={form}"]
    return "\n".join(head + lines + ["##END="]) + "\n"


def test_jcamp_difdup_roundtrip_with_runs_and_negative_values():
    rng = np.random.default_rng(3)
    ys = list(rng.integers(-5000, 5000, size=57))
    ys[10:16] = [100] * 6                       # DUP of a value (DIF 0)
    ys[20:26] = [7, 10, 13, 16, 19, 22]         # DUP of a DIF
    xs = list(range(1000, 1057))
    text = jcamp_text(encode_difdup(xs, ys), len(ys), 1000, 1056, yfactor=0.001)
    d = parse_jcamp(text)
    np.testing.assert_allclose(d["y"], np.array(ys) * 0.001)
    np.testing.assert_allclose(d["x"], xs)


def test_jcamp_affn_sqz_and_exponents():
    ys = [12, -3, 450, 0, 7]
    affn = jcamp_text(["1 12 -3 450", "4 0 7"], 5, 1, 5)
    sqz_text = jcamp_text(["1" + "".join(sqz(v) for v in ys[:3]), "4" + "".join(sqz(v) for v in ys[3:])], 5, 1, 5)
    expo = jcamp_text(["1 1.2E+01 -3.0E0 4.5e2", "4 0.0 7"], 5, 1, 5)
    for t in (affn, sqz_text, expo):
        np.testing.assert_allclose(parse_jcamp(t)["y"], ys)
    # SQZ 'E' directly after the abscissa must not be read as an exponent
    t = jcamp_text(["630.7E2" + sqz(-1) + sqz(3)], 3, 630.7, 632.7)
    np.testing.assert_allclose(parse_jcamp(t)["y"], [52, -1, 3])


def test_jcamp_hand_decoded_example():
    # 1000: A0=10, J=+1 -> 11, T: repeat DIF once more -> 12, k=-2 -> 10 ; next line y-check 10
    _, y = decode_xydata(["1000A0JTk", "1004A0j"])
    np.testing.assert_allclose(y, [10, 11, 12, 10, 9])


def test_jcamp_xy_pairs_and_errors(tmp_path):
    t = jcamp_text(["1,5 2,7", "3,9"], 3, 1, 3, form="(XY..XY)")
    d = parse_jcamp(t.replace("##XYDATA", "##XYPOINTS"))
    np.testing.assert_allclose(d["x"], [1, 2, 3])
    np.testing.assert_allclose(d["y"], [5, 7, 9])
    with pytest.raises(JcampError):
        parse_jcamp(jcamp_text(["1 2 3"], 5, 1, 5))     # NPOINTS mismatch
    with pytest.raises(JcampError):
        decode_xydata(["1000A0J", "1002B0J", "1004A0"])  # wrong y-check in the middle
    p = tmp_path / "s.jdx"
    p.write_text(jcamp_text(["1 12 -3 450", "4 0 7"], 5, 1, 5))
    s = read_file(p)
    assert s.x_unit == "cm-1" and s.n == 5 and s.meta["source"]["reader"] == "jcamp"
