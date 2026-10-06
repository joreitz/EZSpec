//! Whittaker smoother and the asymmetric baseline family built on it.
//!
//! All methods solve the banded symmetric positive definite system
//! `(W + lam * D^T D) z = W y`, where `D` is the difference matrix of order
//! `d`. The matrix has half bandwidth `d` and is factorised by a banded
//! Cholesky decomposition in `O(N d^2)`.
//!
//! The iteration schemes reproduce pybaselines 1.2.1 (`asls`, `arpls`):
//! same defaults, same weighting functions, same convergence criterion
//! `||w_new - w_old|| / max(||w_old||, eps) < tol` evaluated *before* the
//! weights are replaced, and at most `max_iter + 1` solves.
//!
//! Extension (not in pybaselines): an optional `fixed` array. `NaN` marks a
//! free point; any other value is a weight that is held fixed through the
//! iteration (0 = excluded region, e.g. a user mask over a peak; 1 = point
//! forced onto the baseline). Statistics of the arPLS weighting are computed
//! from free points only. With no fixed points the result is identical to
//! pybaselines.

use crate::error::{check_len, CoreError};

/// Lower-band storage of a symmetric banded matrix:
/// `band[k][i] = A[i + k][i]` for `k = 0..=d`, `i = 0..n-k`.
#[derive(Clone, Debug)]
pub struct SymBand {
    pub n: usize,
    pub d: usize,
    pub band: Vec<Vec<f64>>,
}

/// Binomial coefficients of the forward difference of order `d` with
/// alternating sign, e.g. d = 2 -> [1, -2, 1].
pub fn difference_coefficients(d: usize) -> Vec<f64> {
    let mut c = vec![1.0f64];
    for _ in 0..d {
        let mut next = vec![0.0; c.len() + 1];
        for (i, &v) in c.iter().enumerate() {
            next[i] -= v;
            next[i + 1] += v;
        }
        c = next;
    }
    // the sign convention is irrelevant for D^T D
    c
}

/// Banded representation of `D^T D` for `n` points and difference order `d`.
pub fn penalty(n: usize, d: usize) -> SymBand {
    let mut band: Vec<Vec<f64>> = (0..=d).map(|k| vec![0.0; n.saturating_sub(k)]).collect();
    if n > d {
        let c = difference_coefficients(d);
        for row in 0..(n - d) {
            for a in 0..=d {
                for b in 0..=a {
                    // contributes c_a c_b to A[row + a][row + b] (a >= b: lower triangle)
                    band[a - b][row + b] += c[a] * c[b];
                }
            }
        }
    }
    SymBand { n, d, band }
}

/// In-place banded Cholesky factorisation `A = L L^T` (lower band storage).
pub fn cholesky_banded(a: &mut SymBand) -> Result<(), CoreError> {
    let n = a.n;
    let d = a.d;
    for j in 0..n {
        let k0 = j.saturating_sub(d);
        let mut s = a.band[0][j];
        for k in k0..j {
            let l = a.band[j - k][k];
            s -= l * l;
        }
        if !(s > 0.0) || !s.is_finite() {
            return Err(CoreError::NotPositiveDefinite { row: j });
        }
        let ljj = s.sqrt();
        a.band[0][j] = ljj;
        let imax = (j + d).min(n - 1);
        for i in (j + 1)..=imax {
            let k0 = i.saturating_sub(d);
            let mut s = a.band[i - j][j];
            for k in k0..j {
                s -= a.band[i - k][k] * a.band[j - k][k];
            }
            a.band[i - j][j] = s / ljj;
        }
    }
    Ok(())
}

/// Solve `L L^T x = b` for a factor produced by [`cholesky_banded`].
pub fn cholesky_solve(l: &SymBand, b: &mut [f64]) {
    let n = l.n;
    let d = l.d;
    // forward: L y = b
    for i in 0..n {
        let mut s = b[i];
        for k in i.saturating_sub(d)..i {
            s -= l.band[i - k][k] * b[k];
        }
        b[i] = s / l.band[0][i];
    }
    // backward: L^T x = y
    for i in (0..n).rev() {
        let mut s = b[i];
        let kmax = (i + d).min(n - 1);
        for k in (i + 1)..=kmax {
            s -= l.band[k - i][i] * b[k];
        }
        b[i] = s / l.band[0][i];
    }
}

/// Precomputed `lam * D^T D` for repeated solves with changing weights.
#[derive(Clone, Debug)]
pub struct WhittakerSystem {
    pen: SymBand,
}

impl WhittakerSystem {
    pub fn new(n: usize, lam: f64, d: usize) -> Result<Self, CoreError> {
        if !(lam >= 0.0) || !lam.is_finite() {
            return Err(CoreError::InvalidParameter(format!(
                "lam must be finite and >= 0, got {lam}"
            )));
        }
        if d == 0 {
            return Err(CoreError::InvalidParameter(
                "diff_order must be >= 1".into(),
            ));
        }
        if n <= d {
            return Err(CoreError::TooFewPoints {
                needed: d + 1,
                got: n,
            });
        }
        let mut pen = penalty(n, d);
        for band in pen.band.iter_mut() {
            for v in band.iter_mut() {
                *v *= lam;
            }
        }
        Ok(WhittakerSystem { pen })
    }

    pub fn n(&self) -> usize {
        self.pen.n
    }

    /// Solve `(W + lam D^T D) z = W y`.
    pub fn solve(&self, y: &[f64], w: &[f64]) -> Result<Vec<f64>, CoreError> {
        let n = self.pen.n;
        check_len(n, y.len())?;
        check_len(n, w.len())?;
        // W + lam D^T D is positive definite iff at least d points carry a
        // positive weight (the null space of D are polynomials of degree < d).
        let n_pos = w.iter().filter(|&&v| v > 0.0).count();
        if n_pos < self.pen.d {
            return Err(CoreError::NotPositiveDefinite { row: 0 });
        }
        if w.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(CoreError::InvalidParameter(
                "weights must be finite and >= 0".into(),
            ));
        }
        let mut a = self.pen.clone();
        for i in 0..n {
            a.band[0][i] += w[i];
        }
        cholesky_banded(&mut a)?;
        let mut rhs: Vec<f64> = y.iter().zip(w).map(|(yi, wi)| yi * wi).collect();
        cholesky_solve(&a, &mut rhs);
        Ok(rhs)
    }
}

/// Plain Whittaker smoother with weights `w` (all ones for ordinary smoothing).
pub fn whittaker_smooth(y: &[f64], w: &[f64], lam: f64, d: usize) -> Result<Vec<f64>, CoreError> {
    WhittakerSystem::new(y.len(), lam, d)?.solve(y, w)
}

#[derive(Clone, Debug)]
pub struct BaselineFit {
    pub baseline: Vec<f64>,
    /// Weights used in the final solve.
    pub weights: Vec<f64>,
    /// Relative weight change per iteration (pybaselines `tol_history`).
    pub tol_history: Vec<f64>,
    pub converged: bool,
}

const EPS: f64 = f64::EPSILON;

fn norm(v: impl Iterator<Item = f64>) -> f64 {
    v.map(|a| a * a).sum::<f64>().sqrt()
}

fn relative_difference(old: &[f64], new: &[f64]) -> f64 {
    let num = norm(old.iter().zip(new).map(|(o, n)| n - o));
    let den = norm(old.iter().copied()).max(EPS);
    num / den
}

fn apply_fixed(w: &mut [f64], fixed: Option<&[f64]>) {
    if let Some(f) = fixed {
        for (wi, fi) in w.iter_mut().zip(f) {
            if !fi.is_nan() {
                *wi = *fi;
            }
        }
    }
}

fn initial_weights(
    n: usize,
    w0: Option<&[f64]>,
    fixed: Option<&[f64]>,
) -> Result<Vec<f64>, CoreError> {
    let mut w = match w0 {
        Some(w0) => {
            check_len(n, w0.len())?;
            w0.to_vec()
        }
        None => vec![1.0; n],
    };
    if let Some(f) = fixed {
        check_len(n, f.len())?;
    }
    apply_fixed(&mut w, fixed);
    Ok(w)
}

/// Asymmetric least squares (Eilers & Boelens 2005), pybaselines `asls`.
/// Defaults there: lam = 1e6, p = 0.01, d = 2, max_iter = 50, tol = 1e-3.
#[allow(clippy::too_many_arguments)]
pub fn asls(
    y: &[f64],
    lam: f64,
    p: f64,
    d: usize,
    max_iter: usize,
    tol: f64,
    w0: Option<&[f64]>,
    fixed: Option<&[f64]>,
) -> Result<BaselineFit, CoreError> {
    if !(p > 0.0 && p < 1.0) {
        return Err(CoreError::InvalidParameter(
            "p must be between 0 and 1".into(),
        ));
    }
    let n = y.len();
    let system = WhittakerSystem::new(n, lam, d)?;
    let mut w = initial_weights(n, w0, fixed)?;
    let mut history = Vec::with_capacity(max_iter + 1);
    let mut baseline = Vec::new();
    let mut converged = false;
    for _ in 0..=max_iter {
        baseline = system.solve(y, &w)?;
        let mut new_w: Vec<f64> = y
            .iter()
            .zip(&baseline)
            .map(|(&yi, &zi)| if yi > zi { p } else { 1.0 - p })
            .collect();
        apply_fixed(&mut new_w, fixed);
        let diff = relative_difference(&w, &new_w);
        history.push(diff);
        if diff < tol {
            converged = true;
            break;
        }
        w = new_w;
    }
    Ok(BaselineFit {
        baseline,
        weights: w,
        tol_history: history,
        converged,
    })
}

/// Logistic function evaluated without overflow (scipy.special.expit).
#[inline]
fn expit(t: f64) -> f64 {
    if t >= 0.0 {
        1.0 / (1.0 + (-t).exp())
    } else {
        let e = t.exp();
        e / (1.0 + e)
    }
}

/// Asymmetrically reweighted penalized least squares (Baek et al. 2015), pybaselines `arpls`.
/// Defaults there: lam = 1e5, d = 2, max_iter = 50, tol = 1e-3.
pub fn arpls(
    y: &[f64],
    lam: f64,
    d: usize,
    max_iter: usize,
    tol: f64,
    w0: Option<&[f64]>,
    fixed: Option<&[f64]>,
) -> Result<BaselineFit, CoreError> {
    let n = y.len();
    let system = WhittakerSystem::new(n, lam, d)?;
    let mut w = initial_weights(n, w0, fixed)?;
    let is_free = |i: usize| fixed.map_or(true, |f| f[i].is_nan());
    let mut history = Vec::with_capacity(max_iter + 1);
    let mut baseline = Vec::new();
    let mut converged = false;
    for _ in 0..=max_iter {
        baseline = system.solve(y, &w)?;
        let neg: Vec<f64> = (0..n)
            .filter(|&i| is_free(i))
            .map(|i| y[i] - baseline[i])
            .filter(|&r| r < 0.0)
            .collect();
        if neg.len() < 2 {
            // pybaselines: "almost all baseline points are below the data"; stop
            // and keep the current baseline and weights
            break;
        }
        let m = neg.iter().sum::<f64>() / neg.len() as f64;
        let var = neg.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / (neg.len() - 1) as f64;
        let mut std = var.sqrt();
        if std == 0.0 {
            std = EPS;
        }
        let mut new_w: Vec<f64> = y
            .iter()
            .zip(&baseline)
            .map(|(&yi, &zi)| expit(-(2.0 / std) * ((yi - zi) - (2.0 * std - m))))
            .collect();
        apply_fixed(&mut new_w, fixed);
        let diff = relative_difference(&w, &new_w);
        history.push(diff);
        if diff < tol {
            converged = true;
            break;
        }
        w = new_w;
    }
    Ok(BaselineFit {
        baseline,
        weights: w,
        tol_history: history,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{linspace, Rng};

    fn dense_from_band(a: &SymBand) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; a.n]; a.n];
        for k in 0..=a.d {
            for i in 0..a.n - k {
                m[i + k][i] = a.band[k][i];
                m[i][i + k] = a.band[k][i];
            }
        }
        m
    }

    #[test]
    fn difference_coefficients_are_binomial() {
        assert_eq!(difference_coefficients(1), vec![-1.0, 1.0]);
        assert_eq!(difference_coefficients(2), vec![1.0, -2.0, 1.0]);
        assert_eq!(difference_coefficients(3), vec![-1.0, 3.0, -3.0, 1.0]);
    }

    #[test]
    fn penalty_matches_known_second_order_pattern() {
        let p = penalty(6, 2);
        assert_eq!(p.band[0], vec![1.0, 5.0, 6.0, 6.0, 5.0, 1.0]);
        assert_eq!(p.band[1], vec![-2.0, -4.0, -4.0, -4.0, -2.0]);
        assert_eq!(p.band[2], vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn banded_cholesky_solves_the_dense_system() {
        let n = 40;
        let mut rng = Rng::new(7);
        for d in 1..=3 {
            let sys = WhittakerSystem::new(n, 12.5, d).unwrap();
            let w: Vec<f64> = (0..n).map(|_| 0.1 + rng.uniform()).collect();
            let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let z = sys.solve(&y, &w).unwrap();
            let mut a = sys.pen.clone();
            for i in 0..n {
                a.band[0][i] += w[i];
            }
            let m = dense_from_band(&a);
            for i in 0..n {
                let lhs: f64 = (0..n).map(|j| m[i][j] * z[j]).sum();
                assert!((lhs - w[i] * y[i]).abs() < 1e-10, "d={d} row {i}");
            }
        }
    }

    #[test]
    fn huge_lambda_gives_polynomial_least_squares() {
        // null space of D (order 2) are straight lines -> lam -> inf gives the LS line
        let x = linspace(0.0, 1.0, 101);
        let y: Vec<f64> = x
            .iter()
            .map(|&t| 2.0 + 3.0 * t + 0.1 * (20.0 * t).sin())
            .collect();
        let w = vec![1.0; x.len()];
        let z = whittaker_smooth(&y, &w, 1e9, 2).unwrap();
        // closed-form LS line
        let n = x.len() as f64;
        let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
        let sxx: f64 = x.iter().map(|a| a * a).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let icpt = (sy - slope * sx) / n;
        let dev = x
            .iter()
            .zip(&z)
            .map(|(xi, zi)| (zi - (icpt + slope * xi)).abs())
            .fold(0.0, f64::max);
        assert!(dev < 1e-4, "{dev}");
    }

    #[test]
    fn zero_weights_interpolate() {
        let y = vec![0.0, 1.0, 2.0, 100.0, 4.0, 5.0, 6.0];
        let w = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let z = whittaker_smooth(&y, &w, 1e-6, 2).unwrap();
        assert!((z[3] - 3.0).abs() < 1e-4, "{}", z[3]);
    }

    #[test]
    fn too_few_weighted_points_is_an_error() {
        let y = vec![1.0; 10];
        let mut w = vec![0.0; 10];
        w[3] = 1.0; // only one point carries weight; order-2 null space is 2-dim
        assert!(matches!(
            whittaker_smooth(&y, &w, 1.0, 2),
            Err(CoreError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn asls_and_arpls_recover_a_smooth_baseline_under_peaks() {
        let x = linspace(0.0, 100.0, 1000);
        let truth: Vec<f64> = x
            .iter()
            .map(|&t| 5.0 + 0.02 * t + 2.0 * (t / 30.0).sin())
            .collect();
        let mut rng = Rng::new(3);
        let y: Vec<f64> = x
            .iter()
            .zip(&truth)
            .map(|(&t, &b)| {
                b + 20.0 * (-((t - 30.0) / 1.5f64).powi(2)).exp()
                    + 15.0 * (-((t - 70.0) / 2.0f64).powi(2)).exp()
                    + 0.05 * rng.normal()
            })
            .collect();
        let a = asls(&y, 1e6, 1e-3, 2, 50, 1e-3, None, None).unwrap();
        let r = arpls(&y, 1e6, 2, 50, 1e-3, None, None).unwrap();
        assert!(a.converged && r.converged);
        let rmse = |b: &[f64]| {
            (b.iter()
                .zip(&truth)
                .map(|(p, q)| (p - q).powi(2))
                .sum::<f64>()
                / b.len() as f64)
                .sqrt()
        };
        assert!(rmse(&a.baseline) < 0.25, "asls rmse {}", rmse(&a.baseline));
        assert!(rmse(&r.baseline) < 0.25, "arpls rmse {}", rmse(&r.baseline));
    }

    #[test]
    fn fixed_zero_weights_exclude_regions() {
        let n = 200;
        let x = linspace(0.0, 1.0, n);
        let inside = |t: f64| (0.2..0.8).contains(&t);
        // straight line plus a large band confined to the masked region
        let y: Vec<f64> = x
            .iter()
            .map(|&t| {
                1.0 + t
                    + if inside(t) {
                        3.0 * (std::f64::consts::PI * (t - 0.2) / 0.6).sin()
                    } else {
                        0.0
                    }
            })
            .collect();
        let fixed: Vec<f64> = x
            .iter()
            .map(|&t| if inside(t) { 0.0 } else { f64::NAN })
            .collect();
        let r = asls(&y, 1e3, 0.5, 2, 3, 1e-3, None, Some(&fixed)).unwrap();
        for (t, b) in x.iter().zip(&r.baseline) {
            assert!((b - (1.0 + t)).abs() < 1e-8, "{t}: {b}");
        }
        assert!(r
            .weights
            .iter()
            .zip(&x)
            .all(|(w, &t)| !inside(t) || *w == 0.0));
    }
}
