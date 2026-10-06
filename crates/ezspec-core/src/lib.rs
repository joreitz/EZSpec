//! Numerical core of EZSpec.
//!
//! This crate is deliberately free of any GUI or Python dependency. Every
//! routine here has a reference implementation in the Python layer
//! (NumPy/SciPy/pybaselines); the Python test-suite checks that both agree.
//! The Python side defines what is *correct*, this crate makes it *fast*.

// `!(a > b)` is used deliberately: unlike `a <= b` it is also true for NaN,
// which is how invalid parameters are rejected.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Index loops mirror the formulas of the numerical algorithms.
#![allow(clippy::needless_range_loop)]

pub mod decimate;
pub mod error;
pub mod interp;
pub mod lineshapes;
pub mod noise;
pub mod rubberband;
pub mod smooth;
pub mod snip;
pub mod whittaker;

pub use error::CoreError;

/// Median of a slice (NumPy convention: mean of the two central values for
/// even length). Returns NaN for an empty slice.
pub fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|a| !a.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, &mut upper, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        upper
    } else {
        // largest element of the lower half
        let lower = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    /// Small deterministic RNG (SplitMix64 + Box-Muller) so that tests need
    /// no external crates.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Rng(seed)
        }
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        pub fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    pub fn trapz(x: &[f64], y: &[f64]) -> f64 {
        x.windows(2)
            .zip(y.windows(2))
            .map(|(xw, yw)| 0.5 * (xw[1] - xw[0]) * (yw[0] + yw[1]))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_matches_numpy_convention() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&[]).is_nan());
    }
}
