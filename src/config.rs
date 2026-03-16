/// Sector ETF ticker symbols (SPDR S&P 500 sector ETFs)
pub const SECTOR_ETFS: &[(&str, &str)] = &[
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLE", "Energy"),
    ("XLV", "Healthcare"),
    ("XLI", "Industrials"),
    ("XLP", "Consumer Staples"),
    ("XLY", "Consumer Discretionary"),
    ("XLU", "Utilities"),
    ("XLRE", "Real Estate"),
    ("XLC", "Communication Services"),
    ("XLB", "Materials"),
];

/// Market benchmark
pub const BENCHMARK_SYMBOL: &str = "SPY";

/// Financial Modeling Prep API key.
/// Reads FMP_API_KEY from the environment (set in a gitignored .env file).
/// Call `load_env()` once at startup to populate the environment from .env.
pub fn fmp_api_key() -> String {
    dotenvy::var("FMP_API_KEY").unwrap_or_default()
}

/// Load variables from a `.env` file in the working directory into the process
/// environment. Silent no-op if the file is absent (production builds may rely
/// on the environment being set externally).
pub fn load_env() {
    let Ok(contents) = std::fs::read_to_string(".env") else { return };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            if std::env::var(key).is_err() {
                // Only set if not already present in the environment
                unsafe { std::env::set_var(key, value); }
            }
        }
    }
}

/// Update or insert `key=value` in the `.env` file on disk.
///
/// Existing lines for `key` are replaced in-place. All other lines are
/// preserved. The file is created if it does not yet exist.
pub fn save_env_file(key: &str, value: &str) -> Result<(), String> {
    let path = std::path::Path::new(".env");
    let existing = if path.exists() {
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read .env: {e}"))?
    } else {
        String::new()
    };

    let mut found = false;
    let mut lines: Vec<String> = existing
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if let Some((k, _)) = trimmed.split_once('=') {
                if k.trim() == key {
                    found = true;
                    return format!("{key}={value}");
                }
            }
            line.to_string()
        })
        .collect();

    if !found {
        lines.push(format!("{key}={value}"));
    }

    let contents = lines.join("\n") + "\n";
    std::fs::write(path, contents).map_err(|e| format!("Failed to write .env: {e}"))
}

/// Default historical lookback in calendar days (~2 years)
pub const DEFAULT_LOOKBACK_DAYS: u32 = 730;

/// Rolling volatility window sizes (trading days)
pub const SHORT_VOL_WINDOW: usize = 21;  // ~1 month
pub const LONG_VOL_WINDOW: usize = 63;   // ~3 months

/// Neural network configuration
pub const NN_LOOKBACK_DAYS: usize = 60;
pub const NN_FORWARD_DAYS: usize = 5;
pub const NN_HIDDEN_SIZE: usize = 64;
pub const NN_LEARNING_RATE: f64 = 1e-3;
pub const NN_EPOCHS: usize = 1000;
pub const NN_BATCH_SIZE: usize = 32;

/// Walk-forward backtest configuration
/// Minimum fraction of total samples used for the first training window
pub const NN_BACKTEST_MIN_TRAIN_FRAC: f64 = 0.6;
/// Number of samples in each out-of-sample test window per fold
pub const NN_BACKTEST_TEST_WINDOW: usize = 50;
/// Reduced epochs per fold (faster than full NN_EPOCHS)
pub const NN_BACKTEST_EPOCHS: usize = 500;
