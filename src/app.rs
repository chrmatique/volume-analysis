use std::sync::{Arc, Mutex};

use eframe::egui;

use crate::analysis;
use crate::config;
use crate::data::models::{
    BondSpread, ComputeStats, CorrelationMatrix, KurtosisMetrics, MarketData, TrainingStatus,
    VolatilityMetrics,
};
use crate::nn::training::TrainingProgress;
use crate::ui;

/// Active tab in the main UI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    SectorVol,
    Correlations,
    Bonds,
    Kurtosis,
    NeuralNet,
}

/// Computed analysis results (derived from MarketData)
#[derive(Debug, Clone, Default)]
pub struct AnalysisResults {
    pub volatility: Vec<VolatilityMetrics>,
    pub correlation: Option<CorrelationMatrix>,
    pub bond_spreads: Vec<BondSpread>,
    pub avg_cross_correlation: f64,
    pub kurtosis: Vec<KurtosisMetrics>,
}

/// Shared application state
pub struct AppState {
    pub active_tab: Tab,
    pub market_data: MarketData,
    pub analysis: AnalysisResults,
    pub selected_sector_idx: usize,
    pub status_message: String,
    pub is_loading: bool,
    pub training_status: TrainingStatus,
    pub training_losses: Vec<f64>,
    pub nn_predictions: Vec<(String, f64)>,
    pub compute_stats: ComputeStats,
    pub use_gpu: bool,
    pub training_progress: Option<TrainingProgress>,
    /// Shared channel for async data loading results
    pub data_receiver: Option<Arc<Mutex<Option<MarketData>>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            active_tab: Tab::Dashboard,
            market_data: MarketData::default(),
            analysis: AnalysisResults::default(),
            selected_sector_idx: 0,
            status_message: "Ready. Click 'Refresh Data' to fetch market data.".to_string(),
            is_loading: false,
            training_status: TrainingStatus::Idle,
            training_losses: vec![],
            nn_predictions: vec![],
            compute_stats: ComputeStats::default(),
            use_gpu: true,
            training_progress: None,
            data_receiver: None,
        }
    }
}

impl AppState {
    /// Recompute all analysis from current market data
    pub fn recompute_analysis(&mut self) {
        let mut vol_metrics = Vec::new();
        for sector in &self.market_data.sectors {
            if sector.bars.len() < config::LONG_VOL_WINDOW + 2 {
                continue;
            }
            let dates = sector.dates();
            let log_ret = sector.log_returns();
            let highs = sector.highs();
            let lows = sector.lows();

            let vm = analysis::volatility::compute_sector_volatility(
                &sector.symbol,
                &dates,
                &log_ret,
                &highs,
                &lows,
                config::SHORT_VOL_WINDOW,
                config::LONG_VOL_WINDOW,
            );
            vol_metrics.push(vm);
        }

        // Correlation matrix
        let symbols: Vec<String> = self
            .market_data
            .sectors
            .iter()
            .map(|s| s.symbol.clone())
            .collect();
        let returns: Vec<Vec<f64>> = self
            .market_data
            .sectors
            .iter()
            .map(|s| s.log_returns())
            .collect();
        let corr = analysis::cross_sector::compute_correlation_matrix(&symbols, &returns);
        let avg_corr = analysis::cross_sector::average_cross_correlation(&corr);

        // Bond spreads
        let spreads = analysis::bond_spreads::compute_term_spreads(&self.market_data.treasury_rates);

        // Kurtosis
        let mut kurtosis_metrics = Vec::new();
        for sector in &self.market_data.sectors {
            let log_ret = sector.log_returns();
            if log_ret.len() < config::LONG_VOL_WINDOW {
                continue;
            }
            let dates = sector.dates();
            // Use dates offset by 1 to align with log returns
            let ret_dates = if dates.len() > 1 { &dates[1..] } else { &dates };
            let km = analysis::kurtosis::compute_sector_kurtosis(
                &sector.symbol,
                ret_dates,
                &log_ret,
                config::LONG_VOL_WINDOW,
            );
            kurtosis_metrics.push(km);
        }

        self.analysis = AnalysisResults {
            volatility: vol_metrics,
            correlation: Some(corr),
            bond_spreads: spreads,
            avg_cross_correlation: avg_corr,
            kurtosis: kurtosis_metrics,
        };
    }
}

/// Main application struct for eframe
pub struct MktNoiseApp {
    pub state: AppState,
    pub tokio_rt: tokio::runtime::Runtime,
}

impl Default for MktNoiseApp {
    fn default() -> Self {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        Self {
            state: AppState::default(),
            tokio_rt: rt,
        }
    }
}

impl MktNoiseApp {
    fn start_data_fetch(&mut self) {
        if self.state.is_loading {
            return;
        }
        self.state.is_loading = true;
        self.state.status_message = "Fetching market data...".to_string();

        let result_slot: Arc<Mutex<Option<MarketData>>> = Arc::new(Mutex::new(None));
        self.state.data_receiver = Some(result_slot.clone());

        self.tokio_rt.spawn(async move {
            let mut market_data = MarketData::default();

            // Fetch sector ETFs
            let results = crate::data::yahoo::fetch_all_sectors(
                config::SECTOR_ETFS,
                config::DEFAULT_LOOKBACK_DAYS,
            )
            .await;

            for (sym, result) in results {
                match result {
                    Ok(series) => market_data.sectors.push(series),
                    Err(e) => tracing::warn!("Failed to fetch {}: {}", sym, e),
                }
            }

            // Fetch benchmark
            match crate::data::yahoo::fetch_symbol_history(
                config::BENCHMARK_SYMBOL,
                "S&P 500",
                config::DEFAULT_LOOKBACK_DAYS,
            )
            .await
            {
                Ok(bench) => market_data.benchmark = Some(bench),
                Err(e) => tracing::warn!("Failed to fetch benchmark: {}", e),
            }

            // Fetch treasury rates
            match crate::data::fmp::fetch_treasury_rates(config::FMP_API_KEY).await {
                Ok(rates) => market_data.treasury_rates = rates,
                Err(e) => tracing::warn!("Failed to fetch treasury rates: {}", e),
            }

            // Fetch sector performance
            match crate::data::fmp::fetch_sector_performance(config::FMP_API_KEY).await {
                Ok(perf) => market_data.sector_performance = perf,
                Err(e) => tracing::warn!("Failed to fetch sector performance: {}", e),
            }

            market_data.last_refresh = Some(chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string());

            if let Ok(mut slot) = result_slot.lock() {
                *slot = Some(market_data);
            }
        });
    }

    fn check_data_ready(&mut self) {
        let maybe_data = self
            .state
            .data_receiver
            .as_ref()
            .and_then(|receiver| receiver.lock().ok()?.take());

        if let Some(data) = maybe_data {
            let n_sectors = data.sectors.len();
            let n_rates = data.treasury_rates.len();
            self.state.market_data = data;
            self.state.recompute_analysis();
            self.state.is_loading = false;
            self.state.status_message = format!(
                "Loaded {} sectors, {} treasury rate entries. Last refresh: {}",
                n_sectors,
                n_rates,
                self.state
                    .market_data
                    .last_refresh
                    .as_deref()
                    .unwrap_or("N/A")
            );
            self.state.data_receiver = None;
        }
    }
}

impl eframe::App for MktNoiseApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll for async data
        self.check_data_ready();
        if self.state.is_loading {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Top panel with tabs
        egui::TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.state.active_tab, Tab::Dashboard, "Dashboard");
                ui.selectable_value(&mut self.state.active_tab, Tab::SectorVol, "Sector Vol");
                ui.selectable_value(
                    &mut self.state.active_tab,
                    Tab::Correlations,
                    "Correlations",
                );
                ui.selectable_value(&mut self.state.active_tab, Tab::Bonds, "Bonds");
                ui.selectable_value(&mut self.state.active_tab, Tab::Kurtosis, "Kurtosis");
                ui.selectable_value(&mut self.state.active_tab, Tab::NeuralNet, "Neural Net");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.state.is_loading {
                        ui.spinner();
                        ui.label("Loading...");
                    } else if ui.button("Refresh Data").clicked() {
                        self.start_data_fetch();
                    }
                });
            });
        });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.state.status_message);
            });
        });

        // Central panel with active tab content
        egui::CentralPanel::default().show(ctx, |ui| match self.state.active_tab {
            Tab::Dashboard => ui::dashboard::render(ui, &mut self.state),
            Tab::SectorVol => ui::sector_view::render(ui, &mut self.state),
            Tab::Correlations => ui::correlation_view::render(ui, &mut self.state),
            Tab::Bonds => ui::bond_view::render(ui, &mut self.state),
            Tab::Kurtosis => ui::kurtosis_view::render(ui, &mut self.state),
            Tab::NeuralNet => ui::nn_view::render(ui, &mut self.state),
        });
    }
}
