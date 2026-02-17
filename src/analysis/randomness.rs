/// Market randomness analysis: 2D KDE, entropy, Hurst exponent, autocorrelation.

/// Compute 2D kernel density estimation on a grid.
/// Returns (x_grid, y_grid, density_matrix) where density_matrix[ix][iy] is the
/// joint density at (x_grid[ix], y_grid[iy]).
pub fn kde_2d_grid(
    x_data: &[f64],
    y_data: &[f64],
    grid_size: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = x_data.len().min(y_data.len());
    if n < 4 {
        return (vec![], vec![], vec![]);
    }

    let mean_x = x_data[..n].iter().sum::<f64>() / n as f64;
    let mean_y = y_data[..n].iter().sum::<f64>() / n as f64;
    let std_x = (x_data[..n].iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / n as f64).sqrt();
    let std_y = (y_data[..n].iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64).sqrt();

    if std_x <= 0.0 || std_y <= 0.0 {
        return (vec![], vec![], vec![]);
    }

    // Silverman bandwidth
    let nf = n as f64;
    let bw_x = 1.06 * std_x * nf.powf(-0.2);
    let bw_y = 1.06 * std_y * nf.powf(-0.2);

    // Grid range: mean +/- 3.5 std
    let x_min = mean_x - 3.5 * std_x;
    let x_max = mean_x + 3.5 * std_x;
    let y_min = mean_y - 3.5 * std_y;
    let y_max = mean_y + 3.5 * std_y;

    let x_step = (x_max - x_min) / (grid_size - 1) as f64;
    let y_step = (y_max - y_min) / (grid_size - 1) as f64;

    let x_grid: Vec<f64> = (0..grid_size).map(|i| x_min + i as f64 * x_step).collect();
    let y_grid: Vec<f64> = (0..grid_size).map(|j| y_min + j as f64 * y_step).collect();

    let inv_2pi = 1.0 / (2.0 * std::f64::consts::PI);
    let norm = 1.0 / (nf * bw_x * bw_y);

    let mut density = vec![vec![0.0f64; grid_size]; grid_size];

    for ix in 0..grid_size {
        let gx = x_grid[ix];
        for iy in 0..grid_size {
            let gy = y_grid[iy];
            let mut sum = 0.0;
            for k in 0..n {
                let ux = (gx - x_data[k]) / bw_x;
                let uy = (gy - y_data[k]) / bw_y;
                sum += inv_2pi * (-0.5 * (ux * ux + uy * uy)).exp();
            }
            density[ix][iy] = sum * norm;
        }
    }

    (x_grid, y_grid, density)
}

/// Shannon entropy of a return series (discretized into bins).
/// Higher entropy = more random / less predictable.
pub fn shannon_entropy(data: &[f64], bins: usize) -> f64 {
    if data.len() < 2 || bins < 2 {
        return 0.0;
    }

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range <= 0.0 {
        return 0.0;
    }

    let mut counts = vec![0usize; bins];
    for &v in data {
        let idx = ((v - min_val) / range * (bins - 1) as f64).round() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    let n = data.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Autocorrelation at a given lag.
/// For a random walk, autocorrelation should be near zero at all lags.
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag || lag == 0 {
        return 0.0;
    }

    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var <= 0.0 {
        return 0.0;
    }

    let cov: f64 = (0..n - lag)
        .map(|i| (data[i] - mean) * (data[i + lag] - mean))
        .sum::<f64>()
        / n as f64;

    cov / var
}

/// Hurst exponent via rescaled range (R/S) analysis.
/// H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
pub fn hurst_exponent(data: &[f64]) -> f64 {
    if data.len() < 20 {
        return 0.5;
    }

    let sizes: Vec<usize> = vec![8, 16, 32, 64, 128, 256]
        .into_iter()
        .filter(|&s| s <= data.len() / 2)
        .collect();

    if sizes.len() < 2 {
        return 0.5;
    }

    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();

    for &size in &sizes {
        let mut rs_values = Vec::new();
        let n_segments = data.len() / size;

        for seg in 0..n_segments {
            let segment = &data[seg * size..(seg + 1) * size];
            let mean = segment.iter().sum::<f64>() / size as f64;

            // Cumulative deviations from mean
            let mut cumdev = vec![0.0; size];
            cumdev[0] = segment[0] - mean;
            for i in 1..size {
                cumdev[i] = cumdev[i - 1] + (segment[i] - mean);
            }

            let r = cumdev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cumdev.iter().cloned().fold(f64::INFINITY, f64::min);

            let s = (segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / size as f64).sqrt();

            if s > 0.0 {
                rs_values.push(r / s);
            }
        }

        if !rs_values.is_empty() {
            let avg_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            log_n.push((size as f64).ln());
            log_rs.push(avg_rs.ln());
        }
    }

    if log_n.len() < 2 {
        return 0.5;
    }

    // Linear regression: log(R/S) = H * log(n) + c
    let n = log_n.len() as f64;
    let sum_x: f64 = log_n.iter().sum();
    let sum_y: f64 = log_rs.iter().sum();
    let sum_xy: f64 = log_n.iter().zip(log_rs.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = log_n.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.5;
    }

    let h = (n * sum_xy - sum_x * sum_y) / denom;
    h.clamp(0.0, 1.0)
}

/// Compute all randomness metrics for a sector
pub fn compute_sector_randomness(symbol: &str, log_returns: &[f64]) -> SectorRandomness {
    SectorRandomness {
        symbol: symbol.to_string(),
        entropy: shannon_entropy(log_returns, 50),
        hurst_exponent: hurst_exponent(log_returns),
        autocorrelation_lag1: autocorrelation(log_returns, 1),
        autocorrelation_lag5: autocorrelation(log_returns, 5),
    }
}

/// Per-sector randomness metrics
#[derive(Debug, Clone)]
pub struct SectorRandomness {
    pub symbol: String,
    pub entropy: f64,
    pub hurst_exponent: f64,
    pub autocorrelation_lag1: f64,
    pub autocorrelation_lag5: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_positive() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let h = shannon_entropy(&data, 20);
        assert!(h > 0.0, "Entropy should be positive, got {}", h);
    }

    #[test]
    fn test_autocorrelation_self() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ac = autocorrelation(&data, 0);
        // lag 0 is edge case, should handle gracefully
        assert_eq!(ac, 0.0);
    }

    #[test]
    fn test_hurst_random() {
        // Hurst exponent for random data should be near 0.5
        let h = hurst_exponent(&vec![0.5; 200]);
        // constant data is degenerate, just check it doesn't panic
        assert!(h >= 0.0 && h <= 1.0);
    }
}
