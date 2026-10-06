//! Area-normalised peak profiles.
//!
//! Conventions (identical in the Python reference `ezspec.lineshapes`):
//! * `area` is the integral over the real line, so it is directly reportable.
//! * widths are given as FWHM; widths are floored at [`TINY`] to keep the
//!   functions finite at a bound of zero (bounds `fwhm >= 0` are set by the
//!   model layer).
//! * Voigt: `V(x) = Re[w(z)] / (sigma*sqrt(2*pi))`, `z = (x - c + i*gamma)/(sigma*sqrt(2))`,
//!   `sigma = fwhm_g / (2*sqrt(2 ln 2))`, `gamma = fwhm_l / 2`, with the
//!   Faddeeva function `w` from S. G. Johnson's Faddeeva package (the same
//!   algorithm SciPy's `wofz` uses).

use errorfunctions::{ComplexErrorFunctions, RealErrorFunctions};
use num_complex::Complex64;
use std::f64::consts::{LN_2, PI, SQRT_2};

/// Lower floor for widths.
pub const TINY: f64 = 1.0e-15;
/// sigma = FWHM * FWHM_TO_SIGMA for a Gaussian.
pub const FWHM_TO_SIGMA: f64 = 0.424_660_900_144_009_5; // 1 / (2 sqrt(2 ln 2))
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

#[inline]
fn floor_width(w: f64) -> f64 {
    if w > TINY {
        w
    } else {
        TINY
    }
}

/// Gaussian with integral `area`, centre `center` and full width at half maximum `fwhm`.
#[inline]
pub fn gaussian(x: f64, area: f64, center: f64, fwhm: f64) -> f64 {
    let sigma = floor_width(fwhm) * FWHM_TO_SIGMA;
    let t = (x - center) / sigma;
    area * (-0.5 * t * t).exp() / (sigma * SQRT_2PI)
}

/// Lorentzian (Cauchy) with integral `area` and FWHM `fwhm`.
#[inline]
pub fn lorentzian(x: f64, area: f64, center: f64, fwhm: f64) -> f64 {
    let gamma = 0.5 * floor_width(fwhm);
    let dx = x - center;
    area * gamma / (PI * (dx * dx + gamma * gamma))
}

/// Exact Voigt profile (convolution of Gaussian FWHM `fwhm_g` and Lorentzian FWHM `fwhm_l`).
#[inline]
pub fn voigt(x: f64, area: f64, center: f64, fwhm_g: f64, fwhm_l: f64) -> f64 {
    if !(fwhm_g > TINY) {
        return lorentzian(x, area, center, fwhm_l);
    }
    let sigma = fwhm_g * FWHM_TO_SIGMA;
    let gamma = if fwhm_l > 0.0 { 0.5 * fwhm_l } else { 0.0 };
    let z = Complex64::new(x - center, gamma) / (sigma * SQRT_2);
    area * z.w().re / (sigma * SQRT_2PI)
}

/// Olivero & Longbothum (1977) approximation of the Voigt FWHM (~0.02 % accuracy).
#[inline]
pub fn voigt_fwhm(fwhm_g: f64, fwhm_l: f64) -> f64 {
    0.5346 * fwhm_l + (0.2166 * fwhm_l * fwhm_l + fwhm_g * fwhm_g).sqrt()
}

/// Linear combination `eta * L + (1 - eta) * G` with a common FWHM.
#[inline]
pub fn pseudo_voigt(x: f64, area: f64, center: f64, fwhm: f64, eta: f64) -> f64 {
    eta * lorentzian(x, area, center, fwhm) + (1.0 - eta) * gaussian(x, area, center, fwhm)
}

/// Thompson-Cox-Hastings (1987) total width and mixing parameter.
#[inline]
pub fn tch_width_eta(fwhm_g: f64, fwhm_l: f64) -> (f64, f64) {
    let g = fwhm_g.max(0.0);
    let l = fwhm_l.max(0.0);
    let g2 = g * g;
    let l2 = l * l;
    let s = g2 * g2 * g
        + 2.69269 * g2 * g2 * l
        + 2.42843 * g2 * g * l2
        + 4.47163 * g2 * l2 * l
        + 0.07842 * g * l2 * l2
        + l2 * l2 * l;
    let width = s.powf(0.2);
    if !(width > 0.0) {
        return (0.0, 0.0);
    }
    let q = l / width;
    let eta = 1.36603 * q - 0.47719 * q * q + 0.11116 * q * q * q;
    (width, eta)
}

/// Thompson-Cox-Hastings pseudo-Voigt parametrised by Gaussian and Lorentzian FWHM.
/// This is an approximation of the Voigt profile (max. deviation ~1.3 % of the
/// peak height, see tests); use [`voigt`] when the exact shape matters.
#[inline]
pub fn tch_pseudo_voigt(x: f64, area: f64, center: f64, fwhm_g: f64, fwhm_l: f64) -> f64 {
    let (w, eta) = tch_width_eta(fwhm_g, fwhm_l);
    pseudo_voigt(x, area, center, w, eta)
}

/// Pearson VII with integral `area`, FWHM `fwhm` and shape exponent `m` (m > 1/2).
/// m = 1 is a Lorentzian, m -> infinity a Gaussian.
#[inline]
pub fn pearson7(x: f64, area: f64, center: f64, fwhm: f64, m: f64) -> f64 {
    if !(m > 0.5) {
        return f64::NAN;
    }
    let a = floor_width(fwhm) / (2.0 * ((LN_2 / m).exp() - 1.0).sqrt());
    let norm = (libm::lgamma(m) - libm::lgamma(m - 0.5)).exp() / (PI.sqrt() * a);
    let t = (x - center) / a;
    area * norm * (1.0 + t * t).powf(-m)
}

/// Exponentially modified Gaussian: Gaussian (centre `mu`, FWHM `fwhm_g`)
/// convolved with a one-sided exponential of decay length `tau > 0` (tail to +x).
/// Evaluated in a numerically stable piecewise form using `erfcx`.
#[inline]
pub fn emg(x: f64, area: f64, mu: f64, fwhm_g: f64, tau: f64) -> f64 {
    let sigma = floor_width(fwhm_g) * FWHM_TO_SIGMA;
    if !(tau > TINY) {
        return gaussian(x, area, mu, fwhm_g);
    }
    let u = (x - mu) / sigma;
    let b = (sigma / tau - u) / SQRT_2;
    let pref = area / (2.0 * tau);
    if b >= 0.0 {
        pref * (-0.5 * u * u).exp() * RealErrorFunctions::erfcx(b)
    } else {
        let a = 0.5 * (sigma / tau).powi(2) - (x - mu) / tau;
        pref * a.exp() * RealErrorFunctions::erfc(b)
    }
}

/// Evaluate a scalar profile over a slice.
pub fn eval_into<F: Fn(f64) -> f64>(x: &[f64], out: &mut [f64], f: F) {
    for (o, &xi) in out.iter_mut().zip(x) {
        *o = f(xi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{linspace, trapz};

    fn integral<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        let x = linspace(a, b, n);
        let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
        trapz(&x, &y)
    }

    fn half_max_width<F: Fn(f64) -> f64>(f: F, center: f64, guess: f64) -> f64 {
        // bisection for the right half-maximum point of a symmetric peak
        let peak = f(center);
        let (mut lo, mut hi) = (center, center + 20.0 * guess);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.5 * peak {
                lo = mid
            } else {
                hi = mid
            }
        }
        2.0 * (0.5 * (lo + hi) - center)
    }

    #[test]
    fn profiles_are_area_normalised() {
        let gauss = integral(|x| gaussian(x, 2.5, 1.0, 0.7), -20.0, 20.0, 200_001);
        assert!((gauss - 2.5).abs() < 1e-9, "{gauss}");
        // heavy tails: compare with analytic tail correction of the Lorentzian
        let (a, b) = (-2000.0, 2000.0);
        let lor = integral(|x| lorentzian(x, 1.0, 0.0, 1.0), a, b, 4_000_001);
        let expected = ((b / 0.5f64).atan() - (a / 0.5f64).atan()) / PI;
        assert!((lor - expected).abs() < 1e-8, "{lor} vs {expected}");
        let p7 = integral(|x| pearson7(x, 1.0, 0.0, 1.0, 3.0), -200.0, 200.0, 400_001);
        assert!((p7 - 1.0).abs() < 1e-6, "{p7}");
        let e = integral(|x| emg(x, 1.0, 0.0, 1.0, 0.8), -30.0, 60.0, 400_001);
        assert!((e - 1.0).abs() < 1e-8, "{e}");
        let v = integral(|x| voigt(x, 1.0, 0.0, 1.0, 0.0), -30.0, 30.0, 200_001);
        assert!((v - 1.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn fwhm_parameters_are_fwhm() {
        for &(w, m) in &[(1.0, 1.5), (2.3, 4.0), (0.5, 50.0)] {
            assert!((half_max_width(|x| gaussian(x, 1.0, 0.0, w), 0.0, w) - w).abs() < 1e-9);
            assert!((half_max_width(|x| lorentzian(x, 1.0, 0.0, w), 0.0, w) - w).abs() < 1e-9);
            assert!((half_max_width(|x| pearson7(x, 1.0, 0.0, w, m), 0.0, w) - w).abs() < 1e-9);
            assert!(
                (half_max_width(|x| pseudo_voigt(x, 1.0, 0.0, w, 0.3), 0.0, w) - w).abs() < 1e-9
            );
        }
    }

    #[test]
    fn voigt_limits() {
        for &x in &[-3.0, -0.4, 0.0, 0.25, 2.0] {
            let v = voigt(x, 1.0, 0.1, 0.8, 0.0);
            let g = gaussian(x, 1.0, 0.1, 0.8);
            assert!((v - g).abs() < 1e-14 * g.max(1e-300) + 1e-300, "{v} {g}");
            let v = voigt(x, 1.0, 0.1, 0.0, 0.8);
            let l = lorentzian(x, 1.0, 0.1, 0.8);
            assert!((v - l).abs() < 1e-15);
            // tiny Gaussian width converges to the Lorentzian
            let v = voigt(x, 1.0, 0.1, 1e-7, 0.8);
            assert!((v - l).abs() / l < 1e-6, "{v} {l}");
        }
    }

    #[test]
    fn olivero_longbothum_accuracy() {
        // The approximation is quoted with ~0.02 % accuracy against the exact Voigt FWHM.
        for &(g, l) in &[(1.0, 0.0), (1.0, 0.3), (1.0, 1.0), (0.4, 1.0), (0.05, 1.0)] {
            let exact = half_max_width(|x| voigt(x, 1.0, 0.0, g, l), 0.0, g + l);
            let approx = voigt_fwhm(g, l);
            assert!(((approx - exact) / exact).abs() < 2.5e-4, "g={g} l={l}: {approx} vs {exact}");
        }
    }

    #[test]
    fn tch_approximates_the_voigt_to_known_accuracy() {
        // TCH is an approximation. Measured against the exact Voigt (Faddeeva)
        // its width deviates by at most ~0.43 % and the profile by at most
        // ~1.27 % of the peak height over 1e-3 <= fwhm_l/fwhm_g <= 1e3.
        // A typo in one of the coefficients pushes these bounds far out.
        let x = linspace(-30.0, 30.0, 120_001);
        for &(g, l) in &[(1.0, 0.01), (1.0, 0.1), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.1, 1.0), (0.01, 1.0)] {
            let exact = half_max_width(|t| voigt(t, 1.0, 0.0, g, l), 0.0, g + l);
            let (w, eta) = tch_width_eta(g, l);
            assert!(((w - exact) / exact).abs() < 4.5e-3, "g={g} l={l}: {w} vs {exact}");
            assert!((0.0..=1.0).contains(&eta));
            let peak = voigt(0.0, 1.0, 0.0, g, l);
            let worst = x
                .iter()
                .map(|&t| (tch_pseudo_voigt(t, 1.0, 0.0, g, l) - voigt(t, 1.0, 0.0, g, l)).abs())
                .fold(0.0, f64::max);
            assert!(worst / peak < 1.3e-2, "g={g} l={l}: {}", worst / peak);
        }
        let (w, eta) = tch_width_eta(1.0, 0.0);
        assert!((w - 1.0).abs() < 1e-12 && eta.abs() < 1e-12);
        let (w, eta) = tch_width_eta(0.0, 1.0);
        assert!((w - 1.0).abs() < 1e-12 && (eta - 1.0).abs() < 1e-12);
    }

    #[test]
    fn emg_branches_are_continuous_and_reduce_to_gaussian() {
        let (mu, w, tau) = (0.0, 1.0, 0.3);
        let sigma = w * FWHM_TO_SIGMA;
        let xc = mu + sigma * sigma / tau; // b == 0
        let left = emg(xc - 1e-9, 1.0, mu, w, tau);
        let right = emg(xc + 1e-9, 1.0, mu, w, tau);
        assert!((left - right).abs() < 1e-8);
        // far right tail must not overflow
        assert!(emg(1e4, 1.0, 0.0, 0.01, 1e-3).is_finite());
        let g = gaussian(0.2, 1.0, 0.0, 1.0);
        let e = emg(0.2, 1.0, 0.0, 1.0, 1e-9);
        assert!((g - e).abs() / g < 1e-6, "{g} {e}");
    }
}
