//! Rubberband baseline: lower convex hull of the data, linearly interpolated.

use crate::error::{check_increasing, check_len, CoreError};

/// Indices of the vertices of the lower convex hull (Andrew's monotone chain).
pub fn lower_hull(x: &[f64], y: &[f64]) -> Result<Vec<usize>, CoreError> {
    let n = x.len();
    check_len(n, y.len())?;
    if n < 2 {
        return Err(CoreError::TooFewPoints { needed: 2, got: n });
    }
    check_increasing(x)?;
    let mut hull: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        while hull.len() >= 2 {
            let o = hull[hull.len() - 2];
            let a = hull[hull.len() - 1];
            let cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(i);
    }
    Ok(hull)
}

pub fn rubberband(x: &[f64], y: &[f64]) -> Result<Vec<f64>, CoreError> {
    let hull = lower_hull(x, y)?;
    let mut out = vec![0.0; x.len()];
    for seg in hull.windows(2) {
        let (i0, i1) = (seg[0], seg[1]);
        let slope = (y[i1] - y[i0]) / (x[i1] - x[i0]);
        for k in i0..=i1 {
            out[k] = y[i0] + slope * (x[k] - x[i0]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_is_below_data_and_touches_hull_points() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&t| 0.01 * (t - 50.0) * (t - 50.0) + 5.0 * (-((t - 30.0) / 3.0f64).powi(2)).exp())
            .collect();
        let b = rubberband(&x, &y).unwrap();
        assert!(b.iter().zip(&y).all(|(bi, yi)| *bi <= yi + 1e-12));
        assert_eq!(b[0], y[0]);
        assert_eq!(b[99], y[99]);
        assert!((b[50] - y[50]).abs() < 1e-12);
    }
}
