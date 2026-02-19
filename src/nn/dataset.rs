use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Tensor},
};

use crate::analysis;
use crate::config;
use crate::data::models::MarketData;

/// A single training sample: a window of features and targets
#[derive(Debug, Clone)]
pub struct VolSample {
    /// Feature matrix: [seq_length, num_features]
    pub features: Vec<Vec<f64>>,
    /// Target: forward realized volatility
    pub target_vol: f64,
    /// Target: per-sector entropy (5-day forward), length 11
    pub target_randomness: Vec<f64>,
    /// Target: per-sector (kurtosis, skewness) interleaved, length 22
    pub target_kurtosis: Vec<f64>,
}

/// Dataset of volatility prediction samples
#[derive(Debug, Clone)]
pub struct VolDataset {
    pub samples: Vec<VolSample>,
}

impl Dataset<VolSample> for VolDataset {
    fn get(&self, index: usize) -> Option<VolSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Build a dataset from market data by engineering features and creating sliding windows
pub fn build_dataset(data: &MarketData, lookback: usize, forward: usize) -> VolDataset {
    // Compute log returns for each sector
    let sector_returns: Vec<Vec<f64>> = data.sectors.iter().map(|s| s.log_returns()).collect();
    let n_sectors = sector_returns.len();

    if n_sectors == 0 {
        return VolDataset { samples: vec![] };
    }

    // Align all to same length (shortest)
    let min_len = sector_returns.iter().map(|r| r.len()).min().unwrap_or(0);
    if min_len < lookback + forward + config::LONG_VOL_WINDOW {
        return VolDataset { samples: vec![] };
    }

    let aligned_returns: Vec<Vec<f64>> = sector_returns
        .iter()
        .map(|r| r[r.len() - min_len..].to_vec())
        .collect();

    // Compute rolling volatilities for each sector
    let sector_vols: Vec<Vec<f64>> = aligned_returns
        .iter()
        .map(|r| analysis::volatility::rolling_volatility(r, config::SHORT_VOL_WINDOW))
        .collect();

    let vol_len = sector_vols.iter().map(|v| v.len()).min().unwrap_or(0);
    if vol_len < lookback + forward {
        return VolDataset { samples: vec![] };
    }

    // Compute bond spreads
    let bond_spreads = analysis::bond_spreads::compute_term_spreads(&data.treasury_rates);

    // Compute cross-sector correlation (over entire period as a scalar)
    let symbols: Vec<String> = data.sectors.iter().map(|s| s.symbol.clone()).collect();
    let returns_for_corr: Vec<Vec<f64>> = aligned_returns.clone();
    let corr_matrix =
        analysis::cross_sector::compute_correlation_matrix(&symbols, &returns_for_corr);
    let avg_corr = analysis::cross_sector::average_cross_correlation(&corr_matrix);

    // Benchmark (SPY) vol as VIX proxy
    let bench_vol = data.benchmark.as_ref().map(|b| {
        let ret = b.log_returns();
        analysis::volatility::rolling_volatility(&ret, config::SHORT_VOL_WINDOW)
    });

    // Align everything to vol_len
    let aligned_vols: Vec<Vec<f64>> = sector_vols
        .iter()
        .map(|v| v[v.len() - vol_len..].to_vec())
        .collect();

    // Trim returns to match vol length (vol starts SHORT_VOL_WINDOW into returns)
    let aligned_rets: Vec<Vec<f64>> = aligned_returns
        .iter()
        .map(|r| r[r.len() - vol_len..].to_vec())
        .collect();

    // Rolling randomness per sector (entropy, hurst) - 2 values per sector, window 21
    let randomness_window = config::SHORT_VOL_WINDOW;
    let sector_randomness: Vec<Vec<(f64, f64, f64, f64)>> = aligned_rets
        .iter()
        .map(|r| analysis::randomness::rolling_sector_randomness(r, randomness_window))
        .collect();

    // Rolling kurtosis and skewness per sector - use LONG_VOL_WINDOW
    let sector_rolling_kurt: Vec<Vec<f64>> = aligned_rets
        .iter()
        .map(|r| analysis::kurtosis::rolling_kurtosis(r, config::LONG_VOL_WINDOW))
        .collect();
    let sector_rolling_skew: Vec<Vec<f64>> = aligned_rets
        .iter()
        .map(|r| analysis::kurtosis::rolling_skewness(r, config::LONG_VOL_WINDOW))
        .collect();

    let bench_v = bench_vol.map(|bv| {
        if bv.len() >= vol_len {
            bv[bv.len() - vol_len..].to_vec()
        } else {
            vec![0.0; vol_len]
        }
    });

    // Get spread values aligned to the data
    let spread_vals: Vec<f64> = if bond_spreads.len() >= vol_len {
        bond_spreads[..vol_len]
            .iter()
            .rev()
            .map(|s| s.spread_10y_2y)
            .collect()
    } else {
        vec![0.0; vol_len]
    };

    let slope_vals: Vec<f64> = if bond_spreads.len() >= vol_len {
        bond_spreads[..vol_len]
            .iter()
            .rev()
            .map(|s| s.curve_slope)
            .collect()
    } else {
        vec![0.0; vol_len]
    };

    // Build sliding windows
    let mut samples = Vec::new();
    let effective_len = vol_len.saturating_sub(forward);
    if effective_len <= lookback {
        return VolDataset { samples: vec![] };
    }

    for start in 0..(effective_len - lookback) {
        let end = start + lookback;

        // Build feature matrix for this window
        let mut window_features = Vec::with_capacity(lookback);
        for t in start..end {
            let mut features = Vec::with_capacity(crate::nn::model::NUM_FEATURES);

            // 11 sector volatilities
            for sv in &aligned_vols {
                features.push(sv.get(t).copied().unwrap_or(0.0));
            }
            // Pad if fewer sectors
            for _ in n_sectors..11 {
                features.push(0.0);
            }

            // 11 sector returns
            for sr in &aligned_rets {
                features.push(sr.get(t).copied().unwrap_or(0.0));
            }
            for _ in n_sectors..11 {
                features.push(0.0);
            }

            // Average cross-sector correlation
            features.push(avg_corr);

            // Bond spread (10Y-2Y)
            features.push(spread_vals.get(t).copied().unwrap_or(0.0));

            // Curve slope
            features.push(slope_vals.get(t).copied().unwrap_or(0.0));

            // VIX proxy (benchmark vol)
            features.push(
                bench_v
                    .as_ref()
                    .and_then(|bv| bv.get(t).copied())
                    .unwrap_or(0.0),
            );

            // Randomness: entropy, hurst per sector (2 × 11 = 22)
            // rolling_randomness[t-20] aligns with vol[t]; pad with idx 0 when t < 20
            let rr_len = sector_randomness.first().map(|v| v.len()).unwrap_or(0);
            let rr_idx = if t >= randomness_window - 1 && rr_len > 0 {
                (t - (randomness_window - 1)).min(rr_len - 1)
            } else {
                0
            };
            for sr in &sector_randomness {
                if let Some(&(entropy, hurst, _ac1, _ac5)) = sr.get(rr_idx) {
                    features.push(entropy);
                    features.push(hurst);
                } else {
                    features.push(0.0);
                    features.push(0.0);
                }
            }
            for _ in n_sectors..11 {
                features.push(0.0);
                features.push(0.0);
            }

            // Kurtosis: rolling_kurtosis, rolling_skewness per sector (2 × 11 = 22)
            // rolling_kurt at [t-62] aligns with vol[t]; pad with 0 when t < 62
            let k_idx = t.saturating_sub(config::LONG_VOL_WINDOW - 1);
            for (rk, rs) in sector_rolling_kurt.iter().zip(sector_rolling_skew.iter()) {
                features.push(rk.get(k_idx).copied().unwrap_or(0.0));
                features.push(rs.get(k_idx).copied().unwrap_or(0.0));
            }
            for _ in n_sectors..11 {
                features.push(0.0);
                features.push(0.0);
            }

            window_features.push(features);
        }

        // Target vol: average realized vol over [end, end+forward)
        let target_start = end;
        let target_end = (end + forward).min(vol_len);
        let mut target_sum = 0.0;
        let mut target_count = 0;
        for sv in &aligned_vols {
            for tt in target_start..target_end {
                if let Some(v) = sv.get(tt) {
                    target_sum += v;
                    target_count += 1;
                }
            }
        }
        let target_vol = if target_count > 0 {
            target_sum / target_count as f64
        } else {
            0.0
        };

        // Target randomness: 5-day forward entropy per sector (11 values)
        let rr_len = sector_randomness.first().map(|v| v.len()).unwrap_or(0);
        let fwd_rr_idx = if end + forward >= randomness_window && rr_len > 0 {
            (end + forward - randomness_window).min(rr_len - 1)
        } else {
            0
        };
        let mut target_randomness = Vec::with_capacity(11);
        for i in 0..11 {
            target_randomness.push(
                sector_randomness
                    .get(i)
                    .and_then(|sr| sr.get(fwd_rr_idx).map(|(e, _, _, _)| *e))
                    .unwrap_or(0.0),
            );
        }

        // Target kurtosis: 5-day forward (kurtosis, skew) per sector (22 values)
        let fwd_k_idx = (end + forward).saturating_sub(config::LONG_VOL_WINDOW);
        let mut target_kurtosis = Vec::with_capacity(22);
        for i in 0..11 {
            let k = sector_rolling_kurt
                .get(i)
                .and_then(|v| v.get(fwd_k_idx).copied())
                .unwrap_or(0.0);
            let s = sector_rolling_skew
                .get(i)
                .and_then(|v| v.get(fwd_k_idx).copied())
                .unwrap_or(0.0);
            target_kurtosis.push(k);
            target_kurtosis.push(s);
        }

        samples.push(VolSample {
            features: window_features,
            target_vol,
            target_randomness,
            target_kurtosis,
        });
    }

    VolDataset { samples }
}

/// Batcher that converts VolSample slices into tensors for training
#[derive(Clone, Debug)]
pub struct VolBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> VolBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

/// Batched data for training
#[derive(Debug, Clone)]
pub struct VolBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,  // [batch_size, seq_length, num_features]
    pub targets: Tensor<B, 2>, // [batch_size, OUTPUT_SIZE]
}

impl<B: Backend> Batcher<VolSample, VolBatch<B>> for VolBatcher<B> {
    fn batch(&self, items: Vec<VolSample>) -> VolBatch<B> {
        let batch_size = items.len();
        let seq_len = items.first().map(|s| s.features.len()).unwrap_or(0);
        let num_features = items
            .first()
            .and_then(|s| s.features.first().map(|f| f.len()))
            .unwrap_or(0);
        let output_size = crate::nn::model::OUTPUT_SIZE;

        // Flatten features into a single vec for tensor creation
        let mut input_data = Vec::with_capacity(batch_size * seq_len * num_features);
        let mut target_data = Vec::with_capacity(batch_size * output_size);

        for sample in &items {
            for step in &sample.features {
                for &f in step {
                    input_data.push(f as f32);
                }
            }
            target_data.push(sample.target_vol as f32);
            for &v in &sample.target_randomness {
                target_data.push(v as f32);
            }
            for &v in &sample.target_kurtosis {
                target_data.push(v as f32);
            }
        }

        let inputs = Tensor::<B, 1>::from_floats(input_data.as_slice(), &self.device)
            .reshape([batch_size, seq_len, num_features]);

        let targets = Tensor::<B, 1>::from_floats(target_data.as_slice(), &self.device)
            .reshape([batch_size, output_size]);

        VolBatch { inputs, targets }
    }
}
