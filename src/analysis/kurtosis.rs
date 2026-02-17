use chrono::NaiveDate;
use crate::data::models::KurtosisMetrics;

/// Compute the mean of a slice
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the standard deviation (population) of a slice
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Compute excess kurtosis (Fisher's definition: normal = 0)
pub fn excess_kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s == 0.0 {
        return 0.0;
    }
    let m4 = data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    m4 - 3.0
}

/// Compute skewness
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s == 0.0 {
        return 0.0;
    }
    data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

/// Jarque-Bera test statistic for normality
/// Large values indicate departure from normality
pub fn jarque_bera(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return 0.0;
    }
    let s = skewness(data);
    let k = excess_kurtosis(data);
    (n / 6.0) * (s.powi(2) + k.powi(2) / 4.0)
}

/// Compute rolling excess kurtosis over a given window
pub fn rolling_kurtosis(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window || window < 4 {
        return vec![];
    }
    (0..=(data.len() - window))
        .map(|i| excess_kurtosis(&data[i..i + window]))
        .collect()
}

/// Compute rolling skewness over a given window
pub fn rolling_skewness(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window || window < 3 {
        return vec![];
    }
    (0..=(data.len() - window))
        .map(|i| skewness(&data[i..i + window]))
        .collect()
}

/// Gaussian kernel density estimation
/// Returns a vector of (x, density) points suitable for plotting
pub fn kde(data: &[f64], n_points: usize) -> Vec<[f64; 2]> {
    if data.is_empty() {
        return vec![];
    }

    let n = data.len() as f64;
    let s = std_dev(data);
    let m = mean(data);

    // Silverman's rule of thumb for bandwidth
    let bandwidth = 1.06 * s * n.powf(-0.2);
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return vec![];
    }

    // Evaluation range: mean +/- 4 standard deviations
    let x_min = m - 4.0 * s;
    let x_max = m + 4.0 * s;
    let step = (x_max - x_min) / (n_points - 1) as f64;

    (0..n_points)
        .map(|i| {
            let x = x_min + i as f64 * step;
            let density = data.iter()
                .map(|xi| {
                    let u = (x - xi) / bandwidth;
                    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
                })
                .sum::<f64>()
                / (n * bandwidth);
            [x, density]
        })
        .collect()
}

/// Normal distribution PDF evaluated over the same range as the data
pub fn normal_pdf_curve(data: &[f64], n_points: usize) -> Vec<[f64; 2]> {
    if data.is_empty() {
        return vec![];
    }

    let m = mean(data);
    let s = std_dev(data);
    if s <= 0.0 {
        return vec![];
    }

    let x_min = m - 4.0 * s;
    let x_max = m + 4.0 * s;
    let step = (x_max - x_min) / (n_points - 1) as f64;

    (0..n_points)
        .map(|i| {
            let x = x_min + i as f64 * step;
            let z = (x - m) / s;
            let density = (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt());
            [x, density]
        })
        .collect()
}

/// Compute full kurtosis metrics for a sector
pub fn compute_sector_kurtosis(
    symbol: &str,
    dates: &[NaiveDate],
    log_returns: &[f64],
    rolling_window: usize,
) -> KurtosisMetrics {
    let m = mean(log_returns);
    let s = std_dev(log_returns);
    let kurt = excess_kurtosis(log_returns);
    let skew = skewness(log_returns);
    let jb = jarque_bera(log_returns);

    let roll_kurt = rolling_kurtosis(log_returns, rolling_window);
    let roll_skew = rolling_skewness(log_returns, rolling_window);

    // Dates aligned to rolling output (starts at index `window - 1`)
    let roll_dates = if dates.len() >= rolling_window {
        dates[(rolling_window - 1)..].to_vec()
    } else {
        vec![]
    };
    // Trim to match rolling output length
    let roll_dates = roll_dates[..roll_dates.len().min(roll_kurt.len())].to_vec();

    let empirical_density = kde(log_returns, 200);
    let normal_density = normal_pdf_curve(log_returns, 200);

    KurtosisMetrics {
        symbol: symbol.to_string(),
        mean: m,
        std_dev: s,
        excess_kurtosis: kurt,
        skewness: skew,
        jarque_bera: jb,
        rolling_dates: roll_dates,
        rolling_kurtosis: roll_kurt,
        rolling_skewness: roll_skew,
        empirical_density,
        normal_density,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_kurtosis_near_zero() {
        // For a large uniform-ish sample, kurtosis should be roughly -1.2
        let data: Vec<f64> = (0..10000).map(|i| (i as f64) / 10000.0).collect();
        let k = excess_kurtosis(&data);
        assert!((k - (-1.2)).abs() < 0.1, "Uniform kurtosis should be ~-1.2, got {}", k);
    }

    #[test]
    fn test_skewness_symmetric() {
        let data: Vec<f64> = (-500..=500).map(|i| i as f64 / 100.0).collect();
        let s = skewness(&data);
        assert!(s.abs() < 0.01, "Symmetric data should have ~0 skewness, got {}", s);
    }

    #[test]
    fn test_kde_integrates_to_one() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let density = kde(&data, 500);
        if density.len() > 1 {
            let dx = density[1][0] - density[0][0];
            let integral: f64 = density.iter().map(|p| p[1] * dx).sum();
            assert!((integral - 1.0).abs() < 0.05, "KDE should integrate to ~1, got {}", integral);
        }
    }
}
