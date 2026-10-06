//! Moving average.
//!
//! Smoothing correlates neighbouring points and biases peak heights/widths;
//! in EZSpec it is meant for display and peak finding, not as input to a fit
//! (the Python layer warns when a fit depends on smoothed data).

use crate::error::CoreError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMode {
    /// Shrink the window symmetrically near the ends (no bias for linear trends).
    Shrink,
    /// Mirror the data at the ends (`d c b | a b c d | c b a`, NumPy "reflect").
    Reflect,
    /// Repeat the end values.
    Nearest,
}

impl EdgeMode {
    pub fn parse(s: &str) -> Result<Self, CoreError> {
        match s {
            "shrink" => Ok(EdgeMode::Shrink),
            "reflect" => Ok(EdgeMode::Reflect),
            "nearest" => Ok(EdgeMode::Nearest),
            o => Err(CoreError::InvalidParameter(format!(
                "unknown edge mode '{o}'"
            ))),
        }
    }
}

fn reflect_index(i: isize, n: usize) -> usize {
    // NumPy "reflect" (edge sample not repeated); valid for |overshoot| < n
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - j;
    }
    j as usize
}

/// Centred moving average with window `2 * half_window + 1`.
pub fn moving_average(y: &[f64], half_window: usize, mode: EdgeMode) -> Vec<f64> {
    let n = y.len();
    if n == 0 || half_window == 0 {
        return y.to_vec();
    }
    let k = half_window as isize;
    (0..n)
        .map(|i| match mode {
            EdgeMode::Shrink => {
                let h = half_window.min(i).min(n - 1 - i);
                let s: f64 = y[i - h..=i + h].iter().sum();
                s / (2 * h + 1) as f64
            }
            EdgeMode::Reflect | EdgeMode::Nearest => {
                let mut s = 0.0;
                for o in -k..=k {
                    let j = i as isize + o;
                    let idx = if mode == EdgeMode::Reflect {
                        reflect_index(j, n)
                    } else {
                        j.clamp(0, n as isize - 1) as usize
                    };
                    s += y[idx];
                }
                s / (2 * half_window + 1) as f64
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_linear_trends_in_shrink_mode() {
        let y: Vec<f64> = (0..20).map(|i| 3.0 + 0.5 * i as f64).collect();
        let s = moving_average(&y, 3, EdgeMode::Shrink);
        for (a, b) in s.iter().zip(&y) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn reflect_and_nearest_edges() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let r = moving_average(&y, 1, EdgeMode::Reflect);
        assert!((r[0] - (2.0 + 1.0 + 2.0) / 3.0).abs() < 1e-15);
        let n = moving_average(&y, 1, EdgeMode::Nearest);
        assert!((n[0] - (1.0 + 1.0 + 2.0) / 3.0).abs() < 1e-15);
        assert!((n[3] - (3.0 + 4.0 + 4.0) / 3.0).abs() < 1e-15);
    }
}
