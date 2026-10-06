//! Min/max decimation for display of very long traces. Each bin keeps its
//! minimum and maximum in their original order, so spikes stay visible.

pub fn minmax(x: &[f64], y: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().min(y.len());
    if n_bins == 0 || n <= 2 * n_bins {
        return (x[..n].to_vec(), y[..n].to_vec());
    }
    let mut xo = Vec::with_capacity(2 * n_bins);
    let mut yo = Vec::with_capacity(2 * n_bins);
    for b in 0..n_bins {
        let lo = b * n / n_bins;
        let hi = ((b + 1) * n / n_bins).max(lo + 1);
        let (mut imin, mut imax) = (lo, lo);
        for i in lo..hi {
            if y[i] < y[imin] {
                imin = i;
            }
            if y[i] > y[imax] {
                imax = i;
            }
        }
        let (first, second) = if imin <= imax { (imin, imax) } else { (imax, imin) };
        xo.push(x[first]);
        yo.push(y[first]);
        if second != first {
            xo.push(x[second]);
            yo.push(y[second]);
        }
    }
    (xo, yo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_extrema() {
        let x: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let mut y = vec![0.0; 10_000];
        y[1234] = 50.0;
        y[8000] = -7.0;
        let (_, yo) = minmax(&x, &y, 100);
        assert!(yo.len() <= 200);
        assert!(yo.contains(&50.0) && yo.contains(&-7.0));
    }
}
