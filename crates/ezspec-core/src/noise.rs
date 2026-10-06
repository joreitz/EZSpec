//! Noise estimation.

use crate::median;

/// DER_SNR noise estimate (Stoehr et al. 2008):
/// `sigma = 0.6052697 * median(|2 f_i - f_{i-2} - f_{i+2}|)`.
///
/// Assumes Gaussian noise uncorrelated over 2 pixels and a signal that is
/// approximately linear over 5 pixels. Unlike the astronomical reference
/// script, zero values are *not* discarded (baseline-corrected spectra
/// legitimately contain zeros). Returns NaN for fewer than 5 points.
pub fn der_snr(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 5 {
        return f64::NAN;
    }
    let d: Vec<f64> = (2..n - 2).map(|i| (2.0 * y[i] - y[i - 2] - y[i + 2]).abs()).collect();
    0.605_269_7 * median(&d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    #[test]
    fn recovers_gaussian_noise_level_on_a_linear_signal() {
        let mut rng = Rng::new(11);
        let n = 200_000;
        let sigma = 0.37;
        let y: Vec<f64> = (0..n).map(|i| 4.0 + 1e-3 * i as f64 + sigma * rng.normal()).collect();
        let est = der_snr(&y);
        // relative standard error of a median-based estimate at this n is < 0.5 %
        assert!((est / sigma - 1.0).abs() < 0.01, "{est}");
    }
}
