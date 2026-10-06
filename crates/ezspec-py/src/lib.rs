//! Python bindings for `ezspec-core`, importable as `ezspec._core`.
//!
//! All functions take contiguous float64 NumPy arrays (the Python wrapper in
//! `ezspec._backend` guarantees this) and return new arrays. Long-running
//! iterative routines copy their inputs and release the GIL (`py.detach`).

use ezspec_core::{
    decimate, interp, lineshapes as ls, noise, rubberband, smooth, snip as snip_mod, whittaker,
    CoreError,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py(e: CoreError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn slice<'a>(a: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("'{name}' must be a contiguous float64 array")))
}

fn map_shape<'py>(
    py: Python<'py>,
    x: &PyReadonlyArray1<'py, f64>,
    f: impl Fn(f64) -> f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = slice(x, "x")?;
    let mut out = vec![0.0; xs.len()];
    ls::eval_into(xs, &mut out, f);
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn gaussian<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    center: f64,
    fwhm: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| ls::gaussian(t, area, center, fwhm))
}

#[pyfunction]
fn lorentzian<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    center: f64,
    fwhm: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| ls::lorentzian(t, area, center, fwhm))
}

#[pyfunction]
fn voigt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    center: f64,
    fwhm_g: f64,
    fwhm_l: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| ls::voigt(t, area, center, fwhm_g, fwhm_l))
}

#[pyfunction]
fn pseudo_voigt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    center: f64,
    fwhm: f64,
    eta: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| ls::pseudo_voigt(t, area, center, fwhm, eta))
}

#[pyfunction]
fn tch_pseudo_voigt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    center: f64,
    fwhm_g: f64,
    fwhm_l: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| {
        ls::tch_pseudo_voigt(t, area, center, fwhm_g, fwhm_l)
    })
}

#[pyfunction]
fn pearson7<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    center: f64,
    fwhm: f64,
    m: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| ls::pearson7(t, area, center, fwhm, m))
}

#[pyfunction]
fn emg<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    area: f64,
    mu: f64,
    fwhm_g: f64,
    tau: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_shape(py, &x, |t| ls::emg(t, area, mu, fwhm_g, tau))
}

#[pyfunction]
fn voigt_fwhm(fwhm_g: f64, fwhm_l: f64) -> f64 {
    ls::voigt_fwhm(fwhm_g, fwhm_l)
}

#[pyfunction]
fn tch_width_eta(fwhm_g: f64, fwhm_l: f64) -> (f64, f64) {
    ls::tch_width_eta(fwhm_g, fwhm_l)
}

#[pyfunction]
fn whittaker_smooth<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    lam: f64,
    diff_order: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = slice(&y, "y")?.to_vec();
    let w = slice(&weights, "weights")?.to_vec();
    let z = py
        .detach(move || whittaker::whittaker_smooth(&y, &w, lam, diff_order))
        .map_err(to_py)?;
    Ok(z.into_pyarray(py))
}

type BaselineOut<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    bool,
);

fn baseline_out(py: Python<'_>, r: whittaker::BaselineFit) -> BaselineOut<'_> {
    (
        r.baseline.into_pyarray(py),
        r.weights.into_pyarray(py),
        r.tol_history.into_pyarray(py),
        r.converged,
    )
}

fn opt_vec(a: &Option<PyReadonlyArray1<'_, f64>>, name: &str) -> PyResult<Option<Vec<f64>>> {
    match a {
        Some(a) => Ok(Some(slice(a, name)?.to_vec())),
        None => Ok(None),
    }
}

/// Returns (baseline, weights, tol_history, converged).
#[pyfunction]
#[pyo3(signature = (y, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None, fixed=None))]
#[allow(clippy::too_many_arguments)]
fn asls<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    lam: f64,
    p: f64,
    diff_order: usize,
    max_iter: usize,
    tol: f64,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fixed: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<BaselineOut<'py>> {
    let y = slice(&y, "y")?.to_vec();
    let w0 = opt_vec(&weights, "weights")?;
    let fx = opt_vec(&fixed, "fixed")?;
    let r = py
        .detach(move || {
            whittaker::asls(
                &y,
                lam,
                p,
                diff_order,
                max_iter,
                tol,
                w0.as_deref(),
                fx.as_deref(),
            )
        })
        .map_err(to_py)?;
    Ok(baseline_out(py, r))
}

/// Returns (baseline, weights, tol_history, converged).
#[pyfunction]
#[pyo3(signature = (y, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None, fixed=None))]
#[allow(clippy::too_many_arguments)]
fn arpls<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    lam: f64,
    diff_order: usize,
    max_iter: usize,
    tol: f64,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fixed: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<BaselineOut<'py>> {
    let y = slice(&y, "y")?.to_vec();
    let w0 = opt_vec(&weights, "weights")?;
    let fx = opt_vec(&fixed, "fixed")?;
    let r = py
        .detach(move || {
            whittaker::arpls(
                &y,
                lam,
                diff_order,
                max_iter,
                tol,
                w0.as_deref(),
                fx.as_deref(),
            )
        })
        .map_err(to_py)?;
    Ok(baseline_out(py, r))
}

#[pyfunction]
#[pyo3(signature = (y, max_half_window, decreasing=false, filter_order=2))]
fn snip<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    max_half_window: usize,
    decreasing: bool,
    filter_order: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = slice(&y, "y")?.to_vec();
    let b = py
        .detach(move || snip_mod::snip(&y, max_half_window, decreasing, filter_order))
        .map_err(to_py)?;
    Ok(b.into_pyarray(py))
}

#[pyfunction(name = "rubberband")]
fn rubberband_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let b = rubberband::rubberband(slice(&x, "x")?, slice(&y, "y")?).map_err(to_py)?;
    Ok(b.into_pyarray(py))
}

fn interp_common<'py>(
    py: Python<'py>,
    kind: &str,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    xq: PyReadonlyArray1<'py, f64>,
    extrapolation: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ext = interp::Extrapolation::parse(extrapolation).map_err(to_py)?;
    let (x, y, xq) = (slice(&x, "x")?, slice(&y, "y")?, slice(&xq, "xq")?);
    let r = match kind {
        "pchip" => interp::pchip(x, y, xq, ext),
        "cubic" => interp::natural_cubic(x, y, xq, ext),
        _ => interp::linear(x, y, xq, ext),
    }
    .map_err(to_py)?;
    Ok(r.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (x, y, xq, extrapolation="constant"))]
fn pchip<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    xq: PyReadonlyArray1<'py, f64>,
    extrapolation: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    interp_common(py, "pchip", x, y, xq, extrapolation)
}

#[pyfunction]
#[pyo3(signature = (x, y, xq, extrapolation="constant"))]
fn natural_cubic<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    xq: PyReadonlyArray1<'py, f64>,
    extrapolation: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    interp_common(py, "cubic", x, y, xq, extrapolation)
}

#[pyfunction]
#[pyo3(signature = (x, y, xq, extrapolation="constant"))]
fn linear_interp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    xq: PyReadonlyArray1<'py, f64>,
    extrapolation: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    interp_common(py, "linear", x, y, xq, extrapolation)
}

#[pyfunction]
#[pyo3(signature = (y, half_window, mode="shrink"))]
fn moving_average<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    half_window: usize,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = smooth::EdgeMode::parse(mode).map_err(to_py)?;
    Ok(smooth::moving_average(slice(&y, "y")?, half_window, m).into_pyarray(py))
}

#[pyfunction]
fn der_snr(y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(noise::der_snr(slice(&y, "y")?))
}

type ArrayPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

#[pyfunction]
fn minmax_decimate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    n_bins: usize,
) -> PyResult<ArrayPair<'py>> {
    let (xo, yo) = decimate::minmax(slice(&x, "x")?, slice(&y, "y")?, n_bins);
    Ok((xo.into_pyarray(py), yo.into_pyarray(py)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(lorentzian, m)?)?;
    m.add_function(wrap_pyfunction!(voigt, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_voigt, m)?)?;
    m.add_function(wrap_pyfunction!(tch_pseudo_voigt, m)?)?;
    m.add_function(wrap_pyfunction!(pearson7, m)?)?;
    m.add_function(wrap_pyfunction!(emg, m)?)?;
    m.add_function(wrap_pyfunction!(voigt_fwhm, m)?)?;
    m.add_function(wrap_pyfunction!(tch_width_eta, m)?)?;
    m.add_function(wrap_pyfunction!(whittaker_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(asls, m)?)?;
    m.add_function(wrap_pyfunction!(arpls, m)?)?;
    m.add_function(wrap_pyfunction!(snip, m)?)?;
    m.add_function(wrap_pyfunction!(rubberband_py, m)?)?;
    m.add_function(wrap_pyfunction!(pchip, m)?)?;
    m.add_function(wrap_pyfunction!(natural_cubic, m)?)?;
    m.add_function(wrap_pyfunction!(linear_interp, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(der_snr, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_decimate, m)?)?;
    Ok(())
}
