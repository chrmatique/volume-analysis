use std::sync::{Arc, Mutex};

use eframe::egui;

use crate::analysis;
use crate::config;
use crate::analysis::randomness::SectorRandomness;
use crate::data::models::{
    BondSpread, ComputeStats, CorrelationMatrix, GpuAdapterInfo, KurtosisMetrics, MarketData,
    NnFeatureFlags, NnPredictions, ScreenshotSettings, TrainingStatus, VolatilityMetrics,
};
use crate::nn::persistence::ModelMetadata;
use crate::nn::training::TrainingProgress;
use crate::nn::LoadedModel;
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
    Settings,
}

/// Computed analysis results (derived from MarketData)
#[derive(Debug, Clone, Default)]
pub struct AnalysisResults {
    pub volatility: Vec<VolatilityMetrics>,
    pub correlation: Option<CorrelationMatrix>,
    pub bond_spreads: Vec<BondSpread>,
    pub avg_cross_correlation: f64,
    pub kurtosis: Vec<KurtosisMetrics>,
    pub randomness: Vec<SectorRandomness>,
}

/// State for the 3D probability distribution plot on the dashboard
pub struct Plot3DState {
    pub pitch: f32,
    pub yaw: f32,
    pub sector_x_idx: usize,
    pub sector_y_idx: usize,
    pub texture: Option<egui::TextureHandle>,
    pub needs_redraw: bool,
}

impl Default for Plot3DState {
    fn default() -> Self {
        Self {
            pitch: 0.35,
            yaw: 0.75,
            sector_x_idx: 0,
            sector_y_idx: 1,
            texture: None,
            needs_redraw: true,
        }
    }
}

/// Per-chart height overrides (pixels), adjustable by the user at runtime
#[derive(Debug, Clone)]
pub struct ChartHeights {
    pub sector_price: f32,
    pub sector_vol: f32,
    pub sector_ratio: f32,
    pub bond_yield_curve: f32,
    pub bond_term_spread: f32,
    pub bond_curve_slope: f32,
    pub nn_loss: f32,
    pub kurtosis_distribution: f32,
    pub kurtosis_rolling_kurtosis: f32,
    pub kurtosis_rolling_skewness: f32,
    pub kurtosis_accel_chart: f32,
    pub put_call_skew: f32,
}

impl Default for ChartHeights {
    fn default() -> Self {
        Self {
            sector_price: 200.0,
            sector_vol: 250.0,
            sector_ratio: 150.0,
            bond_yield_curve: 200.0,
            bond_term_spread: 200.0,
            bond_curve_slope: 180.0,
            nn_loss: 200.0,
            kurtosis_distribution: 280.0,
            kurtosis_rolling_kurtosis: 200.0,
            kurtosis_rolling_skewness: 200.0,
            kurtosis_accel_chart: 220.0,
            put_call_skew: 200.0,
        }
    }
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
    pub nn_predictions: NnPredictions,
    pub compute_stats: ComputeStats,
    pub use_gpu: bool,
    pub training_progress: Option<TrainingProgress>,
    pub plot_3d: Plot3DState,
    pub chart_heights: ChartHeights,
    /// Loaded model from disk (avoids retraining on each launch)
    pub loaded_model: Option<LoadedModel>,
    pub model_metadata: Option<ModelMetadata>,
    /// Feedback message from the last model save/load attempt, shown in the Neural Net tab
    pub persistence_message: Option<String>,
    /// WGPU-capable adapters (NVIDIA, AMD, Intel) detected at startup
    pub available_gpus: Vec<GpuAdapterInfo>,
    /// Shared channel for async data loading results
    pub data_receiver: Option<Arc<Mutex<Option<MarketData>>>>,
    /// NN training feature flags
    pub nn_feature_flags: NnFeatureFlags,
    /// Screenshot capture settings (save path, format, compression)
    pub screenshot_settings: ScreenshotSettings,
    /// Result slot for the async native folder-picker dialog
    pub folder_picker_result: Option<Arc<Mutex<Option<String>>>>,
    /// Rolling window size for kurtosis analysis (30 or 60 trading days)
    pub kurtosis_window: usize,
}

impl Default for AppState {
    fn default() -> Self {
        let (loaded_model, model_metadata) = match crate::nn::persistence::load_model() {
            Some((model, meta)) => {
                tracing::info!("Loaded saved model (trained {})", meta.trained_at);
                (Some(model), Some(meta))
            }
            None => (None, None),
        };

        let available_gpus = crate::nn::gpu::detect_wgpu_adapters();
        let use_gpu = !available_gpus.is_empty();

        Self {
            active_tab: Tab::Dashboard,
            market_data: MarketData::default(),
            analysis: AnalysisResults::default(),
            selected_sector_idx: 0,
            status_message: "Ready. Click 'Refresh Data' to fetch market data.".to_string(),
            is_loading: false,
            training_status: TrainingStatus::Idle,
            training_losses: vec![],
            nn_predictions: NnPredictions::default(),
            compute_stats: ComputeStats::default(),
            use_gpu,
            training_progress: None,
            plot_3d: Plot3DState::default(),
            chart_heights: ChartHeights::default(),
            loaded_model,
            model_metadata,
            persistence_message: None,
            available_gpus,
            data_receiver: None,
            nn_feature_flags: NnFeatureFlags::default(),
            screenshot_settings: crate::data::cache::load_json("screenshot_settings.json")
                .unwrap_or_default(),
            folder_picker_result: None,
            kurtosis_window: 30,
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
            if log_ret.len() < self.kurtosis_window {
                continue;
            }
            let dates = sector.dates();
            // Use dates offset by 1 to align with log returns
            let ret_dates = if dates.len() > 1 { &dates[1..] } else { &dates };
            let km = analysis::kurtosis::compute_sector_kurtosis(
                &sector.symbol,
                ret_dates,
                &log_ret,
                self.kurtosis_window,
            );
            kurtosis_metrics.push(km);
        }

        // Randomness metrics
        let mut randomness_metrics = Vec::new();
        for sector in &self.market_data.sectors {
            let log_ret = sector.log_returns();
            if log_ret.len() >= 20 {
                randomness_metrics.push(
                    analysis::randomness::compute_sector_randomness(&sector.symbol, &log_ret),
                );
            }
        }

        self.analysis = AnalysisResults {
            volatility: vol_metrics,
            correlation: Some(corr),
            bond_spreads: spreads,
            avg_cross_correlation: avg_corr,
            kurtosis: kurtosis_metrics,
            randomness: randomness_metrics,
        };

        // Signal the 3D plot needs a redraw with new data
        self.plot_3d.needs_redraw = true;
    }

    /// Recompute only kurtosis metrics using the current `kurtosis_window`.
    /// Much faster than `recompute_analysis()` — avoids recalculating vol, bonds, correlations.
    pub fn recompute_kurtosis(&mut self) {
        let mut kurtosis_metrics = Vec::new();
        for sector in &self.market_data.sectors {
            let log_ret = sector.log_returns();
            if log_ret.len() < self.kurtosis_window {
                continue;
            }
            let dates = sector.dates();
            let ret_dates = if dates.len() > 1 { &dates[1..] } else { &dates };
            let km = analysis::kurtosis::compute_sector_kurtosis(
                &sector.symbol,
                ret_dates,
                &log_ret,
                self.kurtosis_window,
            );
            kurtosis_metrics.push(km);
        }
        self.analysis.kurtosis = kurtosis_metrics;
    }
}

/// Main application struct for eframe
pub struct MktNoiseApp {
    pub state: AppState,
    pub tokio_rt: tokio::runtime::Runtime,
}

/// Encode and write a screenshot to disk under `settings.save_path`.
///
/// The filename is `YYYYMMDD_HHMMSS.{ext}`. Returns the full path on success.
fn save_screenshot(
    image: &egui::ColorImage,
    settings: &ScreenshotSettings,
) -> Result<String, String> {
    use std::io::BufWriter;
    use crate::data::models::{ScreenshotCompression, ScreenshotFileType};

    std::fs::create_dir_all(&settings.save_path)
        .map_err(|e| format!("Failed to create directory '{}': {}", settings.save_path, e))?;

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let ext = match settings.file_type {
        ScreenshotFileType::Png => "png",
        ScreenshotFileType::Jpeg => "jpg",
        ScreenshotFileType::Tiff => "tif",
    };
    let path = std::path::Path::new(&settings.save_path).join(format!("{timestamp}.{ext}"));

    let width = image.size[0] as u32;
    let height = image.size[1] as u32;
    let pixels: Vec<u8> = image
        .pixels
        .iter()
        .flat_map(|c| [c.r(), c.g(), c.b(), c.a()])
        .collect();
    let rgba = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| "Failed to create image buffer from pixel data".to_string())?;

    let file = std::fs::File::create(&path)
        .map_err(|e| format!("Failed to create file '{}': {}", path.display(), e))?;
    let mut writer = BufWriter::new(file);

    match settings.file_type {
        ScreenshotFileType::Png => {
            use image::codecs::png::{CompressionType, FilterType, PngEncoder};
            let compression = match settings.compression {
                ScreenshotCompression::None => CompressionType::Fast,
                ScreenshotCompression::Low => CompressionType::Default,
                ScreenshotCompression::High => CompressionType::Best,
            };
            let encoder = PngEncoder::new_with_quality(&mut writer, compression, FilterType::Sub);
            image::DynamicImage::ImageRgba8(rgba)
                .write_with_encoder(encoder)
                .map_err(|e| format!("PNG encode failed: {e}"))?;
        }
        ScreenshotFileType::Jpeg => {
            use image::codecs::jpeg::JpegEncoder;
            let quality: u8 = match settings.compression {
                ScreenshotCompression::None => 100,
                ScreenshotCompression::Low => 80,
                ScreenshotCompression::High => 50,
            };
            // JPEG does not support an alpha channel — convert to RGB first
            let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
            let encoder = JpegEncoder::new_with_quality(&mut writer, quality);
            image::DynamicImage::ImageRgb8(rgb)
                .write_with_encoder(encoder)
                .map_err(|e| format!("JPEG encode failed: {e}"))?;
        }
        ScreenshotFileType::Tiff => {
            use image::codecs::tiff::TiffEncoder;
            // TIFF compression is not directly controllable via this encoder; the
            // compression setting is informational (visible in the settings tooltip).
            let encoder = TiffEncoder::new(&mut writer);
            image::DynamicImage::ImageRgba8(rgba)
                .write_with_encoder(encoder)
                .map_err(|e| format!("TIFF encode failed: {e}"))?;
        }
    }

    Ok(path.to_string_lossy().into_owned())
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
            match crate::data::fmp::fetch_treasury_rates(&config::fmp_api_key()).await {
                Ok(rates) => market_data.treasury_rates = rates,
                Err(e) => tracing::warn!("Failed to fetch treasury rates: {:?}", e),
            }

            // Fetch sector performance
            match crate::data::fmp::fetch_sector_performance(&config::fmp_api_key()).await {
                Ok(perf) => market_data.sector_performance = perf,
                Err(e) => tracing::warn!("Failed to fetch sector performance: {}", e),
            }

            // Fetch CBOE put/call ratio and SKEW
            match crate::data::cboe::fetch_put_call_ratio().await {
                Ok(records) => market_data.put_call_ratio = records,
                Err(e) => tracing::warn!("Failed to fetch CBOE put/call ratio: {:?}", e),
            }
            match crate::data::cboe::fetch_skew_history().await {
                Ok(records) => market_data.skew_history = records,
                Err(e) => tracing::warn!("Failed to fetch CBOE SKEW: {:?}", e),
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
            self.state.available_gpus = crate::nn::gpu::detect_wgpu_adapters();
            if self.state.available_gpus.is_empty() {
                self.state.use_gpu = false;
            }
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

            // Run inference with loaded model if available (avoids retraining)
            if let Some(ref model) = self.state.loaded_model {
                let preds = crate::nn::training::run_inference(model, &self.state.market_data, &self.state.nn_feature_flags);
                if !preds.is_empty() {
                    self.state.nn_predictions = preds.clone();
                    if let Some(ref meta) = self.state.model_metadata {
                        self.state.training_status =
                            crate::data::models::TrainingStatus::Complete {
                                final_loss: meta.final_loss,
                            };
                    }
                }
            }
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

        // Drain the folder-picker result (written by background thread after dialog closes)
        let picked = self
            .state
            .folder_picker_result
            .as_ref()
            .and_then(|slot| slot.lock().ok()?.take());
        if let Some(path) = picked {
            self.state.screenshot_settings.save_path = path;
            self.state.folder_picker_result = None;
        }

        // Handle screenshot events from ViewportCommand::Screenshot (arrives on next frame)
        let events: Vec<egui::Event> = ctx.input(|i| i.events.clone());
        for event in &events {
            if let egui::Event::Screenshot { image, .. } = event {
                match save_screenshot(image, &self.state.screenshot_settings) {
                    Ok(path) => {
                        self.state.status_message = format!("Screenshot saved: {}", path);
                    }
                    Err(e) => {
                        self.state.status_message = format!("Screenshot failed: {}", e);
                    }
                }
            }
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
                ui.selectable_value(&mut self.state.active_tab, Tab::Settings, "Settings");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.state.is_loading {
                        ui.spinner();
                        ui.label("Loading...");
                    } else if ui.button("Refresh Data").clicked() {
                        self.start_data_fetch();
                    }

                    ui.separator();

                    if ui.button("📷").on_hover_text("Take screenshot").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                            egui::UserData::default(),
                        ));
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

        // Central panel with active tab content (scrollable when content overflows)
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| match self.state.active_tab {
                    Tab::Dashboard => ui::dashboard::render(ui, &mut self.state),
                    Tab::SectorVol => ui::sector_view::render(ui, &mut self.state),
                    Tab::Correlations => ui::correlation_view::render(ui, &mut self.state),
                    Tab::Bonds => ui::bond_view::render(ui, &mut self.state),
                    Tab::Kurtosis => ui::kurtosis_view::render(ui, &mut self.state),
                    Tab::NeuralNet => ui::nn_view::render(ui, &mut self.state),
                    Tab::Settings => ui::settings_view::render(ui, &mut self.state),
                });
        });
    }
}
