use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Single OHLCV bar for a given date
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

/// Time series of OHLCV data for a single symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorTimeSeries {
    pub symbol: String,
    pub name: String,
    pub bars: Vec<OhlcvBar>,
}

impl SectorTimeSeries {
    pub fn close_prices(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.close).collect()
    }

    pub fn dates(&self) -> Vec<NaiveDate> {
        self.bars.iter().map(|b| b.date).collect()
    }

    pub fn highs(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.high).collect()
    }

    pub fn lows(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.low).collect()
    }

    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }
}

/// Treasury rate data from FMP API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreasuryRate {
    pub date: String,
    #[serde(default)]
    pub month1: Option<f64>,
    #[serde(default)]
    pub month2: Option<f64>,
    #[serde(default)]
    pub month3: Option<f64>,
    #[serde(default)]
    pub month6: Option<f64>,
    #[serde(default)]
    pub year1: Option<f64>,
    #[serde(default)]
    pub year2: Option<f64>,
    #[serde(default)]
    pub year3: Option<f64>,
    #[serde(default)]
    pub year5: Option<f64>,
    #[serde(default)]
    pub year7: Option<f64>,
    #[serde(default)]
    pub year10: Option<f64>,
    #[serde(default)]
    pub year20: Option<f64>,
    #[serde(default)]
    pub year30: Option<f64>,
}

impl TreasuryRate {
    pub fn parsed_date(&self) -> Option<NaiveDate> {
        NaiveDate::parse_from_str(&self.date, "%Y-%m-%d").ok()
    }
}

/// Computed bond spread for a given date
#[derive(Debug, Clone)]
pub struct BondSpread {
    pub date: NaiveDate,
    pub spread_10y_2y: f64,
    pub curve_slope: f64,
}

/// Sector performance snapshot from FMP stable API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorPerformance {
    pub sector: String,
    #[serde(alias = "averageChange", alias = "changesPercentage")]
    pub changes_percentage: f64,
    #[serde(default)]
    pub exchange: Option<String>,
    #[serde(default)]
    pub date: Option<String>,
}

/// Volatility metrics for a sector over time
#[derive(Debug, Clone)]
pub struct VolatilityMetrics {
    pub symbol: String,
    pub dates: Vec<NaiveDate>,
    pub short_window_vol: Vec<f64>,
    pub long_window_vol: Vec<f64>,
    pub parkinson_vol: Vec<f64>,
    pub vol_ratio: Vec<f64>,
}

/// Correlation matrix result
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub symbols: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

/// Neural network training status
#[derive(Debug, Clone)]
pub enum TrainingStatus {
    Idle,
    Training { epoch: usize, total_epochs: usize, loss: f64 },
    Paused { epoch: usize, total_epochs: usize, loss: f64 },
    Complete { final_loss: f64 },
    Error(String),
}

/// Compute/resource statistics collected during training
#[derive(Debug, Clone, Default)]
pub struct ComputeStats {
    pub backend_name: String,
    pub cpu_usage_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub epoch_duration_ms: u64,
    pub samples_per_sec: f64,
    pub total_params: usize,
    // GPU fields
    pub gpu_detected: bool,
    pub gpu_name: Option<String>,
    pub gpu_vram_total_mb: Option<u64>,
    pub gpu_vram_used_mb: Option<u64>,
    pub gpu_utilization_percent: Option<f32>,
    pub gpu_temperature_c: Option<f32>,
    pub using_gpu: bool,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// Application-wide market data state
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    pub sectors: Vec<SectorTimeSeries>,
    pub benchmark: Option<SectorTimeSeries>,
    pub treasury_rates: Vec<TreasuryRate>,
    pub sector_performance: Vec<SectorPerformance>,
    pub last_refresh: Option<String>,
}
