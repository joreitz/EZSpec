//! SNIP baseline (statistics-sensitive non-linear iterative peak clipping;
//! Ryan et al. 1988, Morhac et al. 1997).
//!
//! Matches pybaselines 1.2.1 `snip` for symmetric windows without smoothing:
//! the data are padded at both ends by linear extrapolation of the outer
//! `max_half_window` points, and each clipping pass uses the values of the
//! previous pass (Jacobi-type update). Higher `filter_order` (4, 6, 8) uses
//! the corresponding higher-order clipping filters.

use crate::error::CoreError;

fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - slope * mx, slope)
}

/// Pad by linear extrapolation of the outer `pad` points (pybaselines "extrapolate").
pub fn pad_extrapolate(y: &[f64], pad: usize) -> Vec<f64> {
    let n = y.len();
    if pad == 0 || n == 0 {
        return y.to_vec();
    }
    let w = pad.min(n);
    let mut out = Vec::with_capacity(n + 2 * pad);
    if w == 1 {
        out.extend(std::iter::repeat(y[0]).take(pad));
        out.extend_from_slice(y);
        out.extend(std::iter::repeat(y[n - 1]).take(pad));
        return out;
    }
    let xl: Vec<f64> = (0..w).map(|i| (pad + i) as f64).collect();
    let (a, b) = linear_fit(&xl, &y[..w]);
    out.extend((0..pad).map(|i| a + b * i as f64));
    out.extend_from_slice(y);
    let xr: Vec<f64> = (0..w).map(|i| (pad + n - w + i) as f64).collect();
    let (a, b) = linear_fit(&xr, &y[n - w..]);
    out.extend((0..pad).map(|i| a + b * (pad + n + i) as f64));
    out
}

pub fn snip(y: &[f64], max_half_window: usize, decreasing: bool, filter_order: usize) -> Result<Vec<f64>, CoreError> {
    if ![2, 4, 6, 8].contains(&filter_order) {
        return Err(CoreError::InvalidParameter("filter_order must be 2, 4, 6 or 8".into()));
    }
    let n = y.len();
    if n < 3 {
        return Err(CoreError::TooFewPoints { needed: 3, got: n });
    }
    let hw = max_half_window.clamp(1, (n - 1) / 2);
    let mut b = pad_extrapolate(y, hw);
    let m = b.len();
    let windows: Vec<usize> = if decreasing { (1..=hw).rev().collect() } else { (1..=hw).collect() };
    let mut filt = vec![0.0; m];
    for i in windows {
        for j in i..m - i {
            let pair = |k: usize| b[j - k] + b[j + k];
            let mut f = 0.5 * pair(i);
            if filter_order > 2 {
                f = f.max((-pair(i) + 4.0 * pair(i / 2)) / 6.0);
            }
            if filter_order > 4 {
                f = f.max((pair(i) - 6.0 * pair(2 * i / 3) + 15.0 * pair(i / 3)) / 20.0);
            }
            if filter_order > 6 {
                f = f.max(
                    (-pair(i) + 8.0 * pair(3 * i / 4) - 28.0 * pair(i / 2) + 56.0 * pair(i / 4)) / 70.0,
                );
            }
            filt[j] = f;
        }
        for j in i..m - i {
            if b[j] > filt[j] {
                b[j] = filt[j];
            }
        }
    }
    Ok(b[hw..hw + n].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::linspace;

    #[test]
    fn linear_data_is_left_unchanged() {
        let y: Vec<f64> = (0..50).map(|i| 1.0 + 0.3 * i as f64).collect();
        for order in [2, 4, 6, 8] {
            let b = snip(&y, 10, false, order).unwrap();
            for (a, c) in b.iter().zip(&y) {
                assert!((a - c).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn clips_narrow_peaks() {
        let x = linspace(0.0, 100.0, 501);
        let y: Vec<f64> = x.iter().map(|&t| 2.0 + 10.0 * (-((t - 50.0) / 1.0f64).powi(2)).exp()).collect();
        let b = snip(&y, 20, false, 2).unwrap();
        assert!((b[250] - 2.0).abs() < 0.05, "{}", b[250]);
    }
}
