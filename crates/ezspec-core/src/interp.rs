//! 1-D interpolants used for anchor-point baselines.
//!
//! * `pchip`: shape-preserving piecewise cubic Hermite (Fritsch-Carlson /
//!   Fritsch-Butland), derivative rule identical to SciPy's `PchipInterpolator`.
//! * `natural_cubic`: cubic spline with zero second derivative at both ends
//!   (SciPy `CubicSpline(bc_type="natural")`).
//! * `linear`.
//!
//! Outside the node range the behaviour is selected by [`Extrapolation`].

use crate::error::{check_increasing, check_len, CoreError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extrapolation {
    /// Hold the end values.
    Constant,
    /// Continue with the end tangent of the interpolant.
    Linear,
    /// Continue the polynomial of the first/last interval (SciPy default).
    Polynomial,
}

impl Extrapolation {
    pub fn parse(s: &str) -> Result<Self, CoreError> {
        match s {
            "constant" => Ok(Extrapolation::Constant),
            "linear" => Ok(Extrapolation::Linear),
            "polynomial" | "extrapolate" => Ok(Extrapolation::Polynomial),
            other => Err(CoreError::InvalidParameter(format!("unknown extrapolation '{other}'"))),
        }
    }
}

fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn pchip_edge(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign(d) != sign(m0) {
        0.0
    } else if sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs() {
        3.0 * m0
    } else {
        d
    }
}

/// Node derivatives of the PCHIP interpolant (SciPy `_find_derivatives`).
pub fn pchip_derivatives(x: &[f64], y: &[f64]) -> Result<Vec<f64>, CoreError> {
    let n = x.len();
    check_len(n, y.len())?;
    if n < 2 {
        return Err(CoreError::TooFewPoints { needed: 2, got: n });
    }
    check_increasing(x)?;
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let m: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
    if n == 2 {
        return Ok(vec![m[0], m[0]]);
    }
    let mut d = vec![0.0; n];
    for k in 1..n - 1 {
        let (m0, m1) = (m[k - 1], m[k]);
        if sign(m0) != sign(m1) || m0 == 0.0 || m1 == 0.0 {
            d[k] = 0.0;
        } else {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            // weighted harmonic mean of the adjacent slopes
            d[k] = (w1 + w2) / (w1 / m0 + w2 / m1);
        }
    }
    d[0] = pchip_edge(h[0], h[1], m[0], m[1]);
    d[n - 1] = pchip_edge(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
    Ok(d)
}

/// Second derivatives of the natural cubic spline through (x, y).
pub fn natural_cubic_second_derivatives(x: &[f64], y: &[f64]) -> Result<Vec<f64>, CoreError> {
    let n = x.len();
    check_len(n, y.len())?;
    if n < 2 {
        return Err(CoreError::TooFewPoints { needed: 2, got: n });
    }
    check_increasing(x)?;
    let mut m2 = vec![0.0; n];
    if n == 2 {
        return Ok(m2);
    }
    // tridiagonal system for interior second derivatives (Thomas algorithm)
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let k = n - 2;
    let mut diag = vec![0.0; k];
    let mut upper = vec![0.0; k];
    let mut rhs = vec![0.0; k];
    for i in 0..k {
        diag[i] = 2.0 * (h[i] + h[i + 1]);
        upper[i] = h[i + 1];
        rhs[i] = 6.0 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i]);
    }
    for i in 1..k {
        let f = h[i] / diag[i - 1];
        diag[i] -= f * upper[i - 1];
        rhs[i] -= f * rhs[i - 1];
    }
    let mut sol = vec![0.0; k];
    sol[k - 1] = rhs[k - 1] / diag[k - 1];
    for i in (0..k - 1).rev() {
        sol[i] = (rhs[i] - upper[i] * sol[i + 1]) / diag[i];
    }
    m2[1..n - 1].copy_from_slice(&sol);
    Ok(m2)
}

#[inline]
fn locate(x: &[f64], t: f64) -> usize {
    // interval index i with x[i] <= t < x[i+1], clamped to [0, n-2]
    let n = x.len();
    match x.partition_point(|&v| v <= t) {
        0 => 0,
        p if p >= n => n - 2,
        p => p - 1,
    }
}

#[inline]
fn hermite(x0: f64, x1: f64, y0: f64, y1: f64, d0: f64, d1: f64, t: f64) -> f64 {
    let h = x1 - x0;
    let s = (t - x0) / h;
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
}

/// Evaluate a piecewise cubic Hermite interpolant with node derivatives `d`.
pub fn hermite_eval(x: &[f64], y: &[f64], d: &[f64], xq: &[f64], ext: Extrapolation) -> Vec<f64> {
    let n = x.len();
    xq.iter()
        .map(|&t| {
            if t < x[0] && ext != Extrapolation::Polynomial {
                return match ext {
                    Extrapolation::Constant => y[0],
                    _ => y[0] + d[0] * (t - x[0]),
                };
            }
            if t > x[n - 1] && ext != Extrapolation::Polynomial {
                return match ext {
                    Extrapolation::Constant => y[n - 1],
                    _ => y[n - 1] + d[n - 1] * (t - x[n - 1]),
                };
            }
            let i = locate(x, t);
            hermite(x[i], x[i + 1], y[i], y[i + 1], d[i], d[i + 1], t)
        })
        .collect()
}

pub fn pchip(x: &[f64], y: &[f64], xq: &[f64], ext: Extrapolation) -> Result<Vec<f64>, CoreError> {
    let d = pchip_derivatives(x, y)?;
    Ok(hermite_eval(x, y, &d, xq, ext))
}

pub fn natural_cubic(x: &[f64], y: &[f64], xq: &[f64], ext: Extrapolation) -> Result<Vec<f64>, CoreError> {
    let m2 = natural_cubic_second_derivatives(x, y)?;
    let n = x.len();
    let eval = |i: usize, t: f64| {
        let h = x[i + 1] - x[i];
        let a = (x[i + 1] - t) / h;
        let b = (t - x[i]) / h;
        a * y[i] + b * y[i + 1] + ((a * a * a - a) * m2[i] + (b * b * b - b) * m2[i + 1]) * h * h / 6.0
    };
    let slope = |i: usize, t: f64| {
        let h = x[i + 1] - x[i];
        let a = (x[i + 1] - t) / h;
        let b = (t - x[i]) / h;
        (y[i + 1] - y[i]) / h - (3.0 * a * a - 1.0) * h * m2[i] / 6.0 + (3.0 * b * b - 1.0) * h * m2[i + 1] / 6.0
    };
    Ok(xq
        .iter()
        .map(|&t| {
            if t < x[0] && ext != Extrapolation::Polynomial {
                return match ext {
                    Extrapolation::Constant => y[0],
                    _ => y[0] + slope(0, x[0]) * (t - x[0]),
                };
            }
            if t > x[n - 1] && ext != Extrapolation::Polynomial {
                return match ext {
                    Extrapolation::Constant => y[n - 1],
                    _ => y[n - 1] + slope(n - 2, x[n - 1]) * (t - x[n - 1]),
                };
            }
            eval(locate(x, t), t)
        })
        .collect())
}

pub fn linear(x: &[f64], y: &[f64], xq: &[f64], ext: Extrapolation) -> Result<Vec<f64>, CoreError> {
    let n = x.len();
    check_len(n, y.len())?;
    if n < 2 {
        return Err(CoreError::TooFewPoints { needed: 2, got: n });
    }
    check_increasing(x)?;
    Ok(xq
        .iter()
        .map(|&t| {
            if ext == Extrapolation::Constant {
                if t <= x[0] {
                    return y[0];
                }
                if t >= x[n - 1] {
                    return y[n - 1];
                }
            }
            let i = locate(x, t);
            let s = (t - x[i]) / (x[i + 1] - x[i]);
            y[i] + s * (y[i + 1] - y[i])
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::linspace;

    #[test]
    fn pchip_is_monotone_on_monotone_data() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.0, 0.1, 5.0, 5.05, 5.1];
        let xq = linspace(0.0, 5.0, 501);
        let v = pchip(&x, &y, &xq, Extrapolation::Constant).unwrap();
        assert!(v.windows(2).all(|w| w[1] >= w[0] - 1e-14));
        // passes through nodes
        let at = pchip(&x, &y, &x, Extrapolation::Constant).unwrap();
        for (a, b) in at.iter().zip(&y) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn natural_spline_reproduces_lines_and_overshoots_steps() {
        let x = vec![0.0, 0.7, 2.0, 3.1, 5.0];
        let y: Vec<f64> = x.iter().map(|t| 1.0 - 2.0 * t).collect();
        let xq = linspace(-1.0, 6.0, 71);
        let v = natural_cubic(&x, &y, &xq, Extrapolation::Linear).unwrap();
        for (t, vi) in xq.iter().zip(&v) {
            assert!((vi - (1.0 - 2.0 * t)).abs() < 1e-12);
        }
        let ys = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let v = natural_cubic(&x, &ys, &linspace(0.0, 5.0, 501), Extrapolation::Constant).unwrap();
        assert!(v.iter().any(|&a| a < -1e-3), "cubic spline should undershoot near a step");
        let p = pchip(&x, &ys, &linspace(0.0, 5.0, 501), Extrapolation::Constant).unwrap();
        assert!(p.iter().all(|&a| (-1e-14..=1.0 + 1e-14).contains(&a)));
    }

    #[test]
    fn extrapolation_modes() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0, 4.0];
        let c = linear(&x, &y, &[-1.0, 3.0], Extrapolation::Constant).unwrap();
        assert_eq!(c, vec![1.0, 4.0]);
        let l = linear(&x, &y, &[-1.0, 3.0], Extrapolation::Linear).unwrap();
        assert_eq!(l, vec![0.0, 6.0]);
    }
}
