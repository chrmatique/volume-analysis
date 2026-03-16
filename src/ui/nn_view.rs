use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::app::AppState;
use crate::data::models::{BacktestResults, BacktestStatus, TrainingStatus};
use crate::nn::training::TrainingProgress;
use crate::ui::chart_utils::{self, height_control, HoverSeries};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Neural Network - Volatility Regime Prediction");
    ui.add_space(8.0);

    if state.market_data.sectors.is_empty() {
        ui.label("Load market data first before training the neural network.");
        return;
    }

    // Model info
    ui.group(|ui| {
        ui.label("Model Architecture: LSTM (hidden=64) -> Linear");
        ui.label("Input: 70 features (vols, returns, randomness, kurtosis, cross-corr, spread, slope, VIX-proxy)");
        ui.label("Output: 5-day forward vol + entropy + kurtosis/skewness per sector");
        ui.label(format!(
            "Lookback: {} trading days per sample",
            crate::config::NN_LOOKBACK_DAYS
        ));
    });

    ui.add_space(8.0);

    // Sync training progress from background thread
    if let Some(ref progress) = state.training_progress {
        if let Ok(status) = progress.status.lock() {
            state.training_status = status.clone();
        }
        if let Ok(losses) = progress.losses.lock() {
            state.training_losses = losses.clone();
        }
        if let Ok(preds) = progress.predictions.lock() {
            state.nn_predictions = preds.clone();
        }
        if let Ok(stats) = progress.compute_stats.lock() {
            state.compute_stats = stats.clone();
        }
    }

    // Sync backtest progress from its background thread
    if let Some(ref progress) = state.backtest_progress {
        if let Ok(bt_status) = progress.backtest_status.lock() {
            state.backtest_status = bt_status.clone();
        }
        if let Ok(bt_results) = progress.backtest_results.lock() {
            state.backtest_results = bt_results.clone();
        }
    }

    // After training completes, load the saved model so we have it for future inference.
    // persistence_message is only set here (not in Default) so the banner is fresh each session.
    if matches!(state.training_status, TrainingStatus::Complete { .. }) && state.loaded_model.is_none() {
        match crate::nn::persistence::load_model() {
            Some((model, meta)) => {
                state.persistence_message = Some(format!(
                    "Model saved and loaded from disk (trained {}; loss: {:.6}).",
                    meta.trained_at, meta.final_loss
                ));
                state.loaded_model = Some(model);
                state.model_metadata = Some(meta);
            }
            None => {
                state.persistence_message =
                    Some("Warning: model could not be saved or loaded from disk.".to_string());
            }
        }
    }

    // Device selector (only when idle or complete -- can't switch mid-training)
    let is_training = matches!(
        state.training_status,
        TrainingStatus::Training { .. } | TrainingStatus::Paused { .. }
    );
    if !is_training {
        ui.horizontal(|ui| {
            ui.label("Training Device:");

            if !state.available_gpus.is_empty() {
                let gpu_label = format!(
                    "GPU ({})",
                    state.available_gpus.first().map(|a| a.name.as_str()).unwrap_or("WGPU")
                );
                ui.selectable_value(&mut state.use_gpu, true, gpu_label);
                ui.selectable_value(&mut state.use_gpu, false, "CPU (NdArray)");
                ui.colored_label(
                    egui::Color32::from_rgb(50, 180, 50),
                    format!("  Detected: {}", state.available_gpus[0].name),
                );
            } else {
                ui.selectable_value(&mut state.use_gpu, false, "CPU (NdArray)");
                state.use_gpu = false;
                ui.colored_label(
                    egui::Color32::from_rgb(220, 100, 50),
                    "  No GPU detected (WGPU found no adapters)",
                );
            }
        });
        ui.add_space(4.0);
    }

    // Model loaded from disk indicator
    if let Some(ref meta) = state.model_metadata {
        ui.horizontal(|ui| {
            ui.colored_label(
                egui::Color32::from_rgb(100, 180, 255),
                format!("Model loaded from disk (trained {}; loss: {:.6})", meta.trained_at, meta.final_loss),
            );
        });
        ui.add_space(4.0);
    }

    // Persistence feedback (save/load result from the most recent training session)
    if let Some(ref msg) = state.persistence_message.clone() {
        let is_warning = msg.starts_with("Warning");
        let color = if is_warning {
            egui::Color32::from_rgb(220, 120, 50)
        } else {
            egui::Color32::from_rgb(50, 200, 100)
        };
        ui.horizontal(|ui| {
            ui.colored_label(color, msg.as_str());
            if ui.small_button("✕").clicked() {
                state.persistence_message = None;
            }
        });
        ui.add_space(4.0);
    }

    // Training controls -- each arm owns its own layout so ProgressBar never hides buttons
    match state.training_status.clone() {
        TrainingStatus::Idle => {
            ui.horizontal(|ui| {
                if ui.button("Train Model").clicked() {
                    start_training(state);
                }
                if state.loaded_model.is_some() {
                    if ui.button("Run Inference").clicked() {
                        if let Some(ref model) = state.loaded_model {
                            let preds = crate::nn::training::run_inference(model, &state.market_data, &state.nn_feature_flags);
                            if !preds.is_empty() {
                                state.nn_predictions = preds.clone();
                                if let Some(ref meta) = state.model_metadata {
                                    state.training_status = TrainingStatus::Complete {
                                        final_loss: meta.final_loss,
                                    };
                                }
                            }
                        }
                    }
                }
            });
        }
        TrainingStatus::Training { epoch, total_epochs, loss } => {
            // Row 1: spinner + status text + Pause button
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!(
                    "Training... Epoch {}/{} | Loss: {:.6}",
                    epoch, total_epochs, loss
                ));
                if ui.button("Pause").clicked() {
                    if let Some(ref progress) = state.training_progress {
                        progress.request_pause();
                    }
                }
            });
            // Row 2: full-width progress bar
            let frac = epoch as f32 / total_epochs as f32;
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
        }
        TrainingStatus::Paused { epoch, total_epochs, loss } => {
            // Row 1: status text + Resume + Stop buttons
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 180, 50),
                    format!("Paused at Epoch {}/{} | Loss: {:.6}", epoch, total_epochs, loss),
                );
                if ui.button("Resume").clicked() {
                    if let Some(ref progress) = state.training_progress {
                        progress.request_resume();
                    }
                }
                if ui.button("Stop").clicked() {
                    state.training_status = TrainingStatus::Idle;
                    state.training_progress = None;
                }
            });
            // Row 2: full-width progress bar
            let frac = epoch as f32 / total_epochs as f32;
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(300));
        }
        TrainingStatus::Complete { final_loss } => {
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(50, 180, 50),
                    format!("Training complete! Final loss: {:.6}", final_loss),
                );
                if ui.button("Retrain").clicked() {
                    state.training_status = TrainingStatus::Idle;
                    state.training_losses.clear();
                    state.nn_predictions = crate::data::models::NnPredictions::default();
                    state.training_progress = None;
                }
                if state.loaded_model.is_some() {
                    if ui.button("Run Inference").clicked() {
                        if let Some(ref model) = state.loaded_model {
                            let preds = crate::nn::training::run_inference(model, &state.market_data, &state.nn_feature_flags);
                            if !preds.is_empty() {
                                state.nn_predictions = preds.clone();
                            }
                        }
                    }
                }
            });
        }
        TrainingStatus::Error(ref msg) => {
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 50, 50),
                    format!("Error: {}", msg),
                );
                if ui.button("Retry").clicked() {
                    state.training_status = TrainingStatus::Idle;
                    state.training_progress = None;
                }
            });
        }
    }

    ui.add_space(8.0);

    // Compute / Resource Statistics panel
    let show_stats = matches!(
        state.training_status,
        TrainingStatus::Training { .. }
            | TrainingStatus::Paused { .. }
            | TrainingStatus::Complete { .. }
    ) || state.compute_stats.total_params > 0;

    if show_stats {
        render_compute_stats(ui, &state.compute_stats, &state.training_status);
        ui.add_space(8.0);
    }

    // Loss curve
    if !state.training_losses.is_empty() {
        ui.heading("Training Loss");
        let loss_data: Vec<[f64; 2]> = state
            .training_losses
            .iter()
            .enumerate()
            .map(|(i, l)| [i as f64, *l])
            .collect();
        let loss_points: PlotPoints = loss_data.iter().copied().collect();
        let loss_hover = [HoverSeries { name: "MSE Loss", data: &loss_data, decimals: 6, suffix: "" }];

        height_control(ui, &mut state.chart_heights.nn_loss, "Loss Chart Height");
        chart_utils::plot_with_y_drag(
            ui,
            "loss_plot",
            chart_utils::default_plot_interaction(
                Plot::new("loss_plot")
                    .height(state.chart_heights.nn_loss),
            )
                .x_axis_label("Epoch")
                .y_axis_label("MSE Loss")
                .coordinates_formatter(chart_utils::HOVER_CORNER, chart_utils::hover_formatter(&loss_hover))
                .label_formatter(chart_utils::no_hover_label),
            |plot_ui| {
                plot_ui.line(
                    Line::new(loss_points)
                        .name("Training Loss")
                        .color(egui::Color32::from_rgb(255, 100, 100)),
                );
            },
        );
    }

    ui.add_space(8.0);

    // Predictions: only show columns for enabled feature flags (Vol | Randomness | Kurtosis)
    if !state.nn_predictions.is_empty() {
        let flags = state.nn_feature_flags.clone();
        let show_vol = flags.sector_volatility;
        let show_rand = flags.market_randomness;
        let show_kurt = flags.kurtosis;
        let col_count = [show_vol, show_rand, show_kurt]
            .iter()
            .filter(|&&v| v)
            .count();

        ui.heading("5-Day Forward Predictions");
        ui.add_space(4.0);

        if col_count == 0 {
            ui.label("All feature groups are disabled. Enable at least one feature in Settings to see predictions.");
        } else {
            // Build ordered list of enabled column renderers so the column indices stay contiguous.
            // We use a nested columns call whose count matches the number of active flags.
            let vol_data: Vec<_> = state.nn_predictions.vol.clone();
            let rand_data: Vec<_> = state.nn_predictions.randomness.clone();
            let kurt_data: Vec<_> = state.nn_predictions.kurtosis.clone();

            ui.columns(col_count, |cols| {
                let mut col_idx = 0;

                if show_vol {
                    cols[col_idx].group(|ui| {
                        ui.strong("Volatility");
                        ui.add_space(4.0);
                        egui::Grid::new("pred_vol_grid")
                            .striped(true)
                            .min_col_width(80.0)
                            .show(ui, |ui| {
                                ui.strong("Sector");
                                ui.strong("Vol (%)");
                                ui.end_row();
                                for (sector, vol) in &vol_data {
                                    ui.label(sector);
                                    let vol_pct = vol * 100.0;
                                    let color = if vol_pct > 30.0 {
                                        egui::Color32::from_rgb(220, 50, 50)
                                    } else if vol_pct > 20.0 {
                                        egui::Color32::from_rgb(220, 150, 50)
                                    } else {
                                        egui::Color32::from_rgb(50, 180, 50)
                                    };
                                    ui.colored_label(color, format!("{:.2}%", vol_pct));
                                    ui.end_row();
                                }
                            });
                    });
                    col_idx += 1;
                }

                if show_rand {
                    cols[col_idx].group(|ui| {
                        ui.strong("Randomness");
                        ui.add_space(4.0);
                        egui::Grid::new("pred_randomness_grid")
                            .striped(true)
                            .min_col_width(80.0)
                            .show(ui, |ui| {
                                ui.strong("Sector");
                                ui.strong("Entropy");
                                ui.end_row();
                                for (sector, entropy) in &rand_data {
                                    ui.label(sector);
                                    ui.label(format!("{:.3}", entropy));
                                    ui.end_row();
                                }
                            });
                    });
                    col_idx += 1;
                }

                if show_kurt {
                    cols[col_idx].group(|ui| {
                        ui.strong("Kurtosis");
                        ui.add_space(4.0);
                        egui::Grid::new("pred_kurtosis_grid")
                            .striped(true)
                            .min_col_width(70.0)
                            .show(ui, |ui| {
                                ui.strong("Sector");
                                ui.strong("Kurt");
                                ui.strong("Skew");
                                ui.end_row();
                                for (sector, k, s) in &kurt_data {
                                    ui.label(sector);
                                    ui.label(format!("{:.2}", k));
                                    ui.label(format!("{:.2}", s));
                                    ui.end_row();
                                }
                            });
                    });
                    let _ = col_idx + 1; // suppress unused warning
                }
            });
        }
    } else if matches!(state.training_status, TrainingStatus::Idle) {
        ui.add_space(8.0);
        ui.label("No predictions yet. Train the model to generate predictions.");
    }

    // ── Walk-Forward Backtest Section ─────────────────────────────────────
    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);
    ui.heading("Walk-Forward Backtest");
    ui.small("Trains a fresh model on expanding windows and measures out-of-sample MAE per output group.");
    ui.add_space(6.0);

    render_backtest_section(ui, state);

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(4.0);
    ui.small("Neural network powered by the Burn deep learning framework.");
}

fn render_compute_stats(
    ui: &mut egui::Ui,
    stats: &crate::data::models::ComputeStats,
    status: &TrainingStatus,
) {
    ui.group(|ui| {
        ui.heading("Compute Statistics");
        ui.add_space(4.0);

        ui.columns(2, |cols| {
            // Left column: System / Training stats
            cols[0].group(|ui| {
                ui.strong("System & Training");
                ui.add_space(4.0);

                egui::Grid::new("system_stats_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(12.0, 3.0))
                    .show(ui, |ui| {
                        // Backend
                        ui.label("Backend:");
                        if stats.backend_name.contains("fallback") {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 165, 0),
                                &stats.backend_name,
                            );
                        } else {
                            ui.strong(&stats.backend_name);
                        }
                        ui.end_row();

                        // Model parameters
                        if stats.total_params > 0 {
                            ui.label("Parameters:");
                            ui.strong(format_param_count(stats.total_params));
                            ui.end_row();
                        }

                        // CPU usage
                        ui.label("CPU:");
                        let cpu_color = cpu_color(stats.cpu_usage_percent);
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::ProgressBar::new(stats.cpu_usage_percent / 100.0)
                                    .desired_width(100.0),
                            );
                            ui.colored_label(cpu_color, format!("{:.1}%", stats.cpu_usage_percent));
                        });
                        ui.end_row();

                        // RAM
                        if stats.memory_total_mb > 0 {
                            ui.label("RAM:");
                            let mem_frac =
                                stats.memory_used_mb as f32 / stats.memory_total_mb as f32;
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::ProgressBar::new(mem_frac).desired_width(100.0),
                                );
                                ui.label(format!(
                                    "{:.1} / {:.1} GB",
                                    stats.memory_used_mb as f64 / 1024.0,
                                    stats.memory_total_mb as f64 / 1024.0,
                                ));
                            });
                            ui.end_row();
                        }

                        // Epoch duration
                        if stats.epoch_duration_ms > 0 {
                            ui.label("Epoch Time:");
                            if stats.epoch_duration_ms >= 1000 {
                                ui.strong(format!(
                                    "{:.2}s",
                                    stats.epoch_duration_ms as f64 / 1000.0
                                ));
                            } else {
                                ui.strong(format!("{}ms", stats.epoch_duration_ms));
                            }
                            ui.end_row();
                        }

                        // Throughput
                        if stats.samples_per_sec > 0.0 {
                            ui.label("Throughput:");
                            ui.strong(format!("{:.0} samples/sec", stats.samples_per_sec));
                            ui.end_row();
                        }

                        // Status indicator
                        ui.label("Status:");
                        match status {
                            TrainingStatus::Training { .. } => {
                                ui.colored_label(
                                    egui::Color32::from_rgb(50, 180, 50),
                                    "Running",
                                );
                            }
                            TrainingStatus::Paused { .. } => {
                                ui.colored_label(
                                    egui::Color32::from_rgb(220, 180, 50),
                                    "Paused",
                                );
                            }
                            TrainingStatus::Complete { .. } => {
                                ui.colored_label(
                                    egui::Color32::from_rgb(100, 150, 255),
                                    "Complete",
                                );
                            }
                            _ => {
                                ui.label("Idle");
                            }
                        }
                        ui.end_row();
                    });
            });

            // Right column: GPU stats
            cols[1].group(|ui| {
                ui.strong("GPU");
                ui.add_space(4.0);

                if !stats.gpu_detected {
                    ui.colored_label(
                        egui::Color32::from_rgb(150, 150, 150),
                        "No NVIDIA GPU detected",
                    );
                    return;
                }

                egui::Grid::new("gpu_stats_grid")
                    .num_columns(2)
                    .spacing(egui::vec2(12.0, 3.0))
                    .show(ui, |ui| {
                        // GPU name
                        ui.label("Device:");
                        ui.strong(stats.gpu_name.as_deref().unwrap_or("Unknown"));
                        ui.end_row();

                        // Active indicator
                        ui.label("Training on:");
                        if stats.using_gpu {
                            ui.colored_label(
                                egui::Color32::from_rgb(50, 220, 50),
                                "GPU",
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::from_rgb(150, 150, 150),
                                "CPU (GPU idle)",
                            );
                        }
                        ui.end_row();

                        // VRAM
                        if let (Some(used), Some(total)) =
                            (stats.gpu_vram_used_mb, stats.gpu_vram_total_mb)
                        {
                            if total > 0 {
                                ui.label("VRAM:");
                                let vram_frac = used as f32 / total as f32;
                                let vram_color = if vram_frac > 0.9 {
                                    egui::Color32::from_rgb(220, 50, 50)
                                } else if vram_frac > 0.7 {
                                    egui::Color32::from_rgb(220, 180, 50)
                                } else {
                                    egui::Color32::from_rgb(50, 180, 50)
                                };
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::ProgressBar::new(vram_frac)
                                            .desired_width(100.0),
                                    );
                                    ui.colored_label(
                                        vram_color,
                                        format!(
                                            "{:.1} / {:.1} GB",
                                            used as f64 / 1024.0,
                                            total as f64 / 1024.0,
                                        ),
                                    );
                                });
                                ui.end_row();
                            }
                        }

                        // GPU utilization
                        if let Some(util) = stats.gpu_utilization_percent {
                            ui.label("Utilization:");
                            let util_color = if util > 80.0 {
                                egui::Color32::from_rgb(50, 180, 50)
                            } else if util > 30.0 {
                                egui::Color32::from_rgb(220, 180, 50)
                            } else {
                                egui::Color32::from_rgb(150, 150, 150)
                            };
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::ProgressBar::new(util / 100.0)
                                        .desired_width(100.0),
                                );
                                ui.colored_label(util_color, format!("{:.0}%", util));
                            });
                            ui.end_row();
                        }

                        // Temperature
                        if let Some(temp) = stats.gpu_temperature_c {
                            ui.label("Temperature:");
                            let temp_color = if temp > 85.0 {
                                egui::Color32::from_rgb(220, 50, 50)
                            } else if temp > 70.0 {
                                egui::Color32::from_rgb(220, 180, 50)
                            } else {
                                egui::Color32::from_rgb(50, 180, 50)
                            };
                            ui.colored_label(temp_color, format!("{:.0} C", temp));
                            ui.end_row();
                        }

                        // When on GPU but no vendor stats available (e.g. AMD without amd-smi)
                        if stats.using_gpu
                            && stats.gpu_vram_total_mb.is_none()
                            && stats.gpu_utilization_percent.is_none()
                        {
                            ui.label("Stats:");
                            ui.colored_label(
                                egui::Color32::from_gray(150),
                                "Unavailable (nvidia-smi / rocm-smi / amd-smi not found)",
                            );
                            ui.end_row();
                        }
                    });
            });
        });
    });
}

fn cpu_color(percent: f32) -> egui::Color32 {
    if percent > 80.0 {
        egui::Color32::from_rgb(220, 50, 50)
    } else if percent > 50.0 {
        egui::Color32::from_rgb(220, 180, 50)
    } else {
        egui::Color32::from_rgb(50, 180, 50)
    }
}

fn format_param_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

fn render_backtest_section(ui: &mut egui::Ui, state: &mut AppState) {
    let is_training = matches!(
        state.training_status,
        TrainingStatus::Training { .. } | TrainingStatus::Paused { .. }
    );
    let is_backtesting = matches!(state.backtest_status, BacktestStatus::Running { .. });

    match state.backtest_status.clone() {
        BacktestStatus::Idle => {
            let btn = egui::Button::new("Run Backtest");
            if ui
                .add_enabled(!is_training && !is_backtesting, btn)
                .on_hover_text("Walk-forward validation across expanding training windows")
                .clicked()
            {
                start_backtest(state);
            }
        }
        BacktestStatus::Running { fold, total_folds } => {
            ui.horizontal(|ui| {
                ui.spinner();
                if total_folds > 0 {
                    ui.label(format!("Fold {fold}/{total_folds} — training and evaluating…"));
                } else {
                    ui.label("Initialising backtest…");
                }
            });
            let frac = if total_folds > 0 {
                fold as f32 / total_folds as f32
            } else {
                0.0
            };
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(300));
        }
        BacktestStatus::Complete => {
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(50, 200, 100),
                    "Backtest complete.",
                );
                if ui.button("Re-run").clicked() {
                    state.backtest_status = BacktestStatus::Idle;
                    state.backtest_results = None;
                    state.backtest_progress = None;
                }
            });
            ui.add_space(4.0);

            if let Some(ref results) = state.backtest_results.clone() {
                render_backtest_results(ui, results);
            }
        }
        BacktestStatus::Error(ref msg) => {
            ui.horizontal(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 50, 50),
                    format!("Backtest error: {msg}"),
                );
                if ui.button("Dismiss").clicked() {
                    state.backtest_status = BacktestStatus::Idle;
                    state.backtest_progress = None;
                }
            });
        }
    }
}

fn render_backtest_results(ui: &mut egui::Ui, results: &BacktestResults) {
    ui.collapsing("Backtest Results", |ui| {
        // Summary stats
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Folds: {}", results.num_folds));
                ui.separator();
                ui.label(format!("OOS samples: {}", results.total_oos_samples));
            });
            ui.add_space(4.0);

            egui::Grid::new("backtest_mae_grid")
                .striped(true)
                .min_col_width(130.0)
                .show(ui, |ui| {
                    ui.strong("Output Group");
                    ui.strong("MAE (overall)");
                    ui.end_row();

                    ui.label("Volatility");
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 160, 80),
                        format!("{:.6}", results.vol_mae),
                    );
                    ui.end_row();

                    ui.label("Entropy");
                    ui.colored_label(
                        egui::Color32::from_rgb(100, 180, 255),
                        format!("{:.6}", results.entropy_mae),
                    );
                    ui.end_row();

                    ui.label("Kurtosis / Skew");
                    ui.colored_label(
                        egui::Color32::from_rgb(180, 130, 255),
                        format!("{:.6}", results.kurtosis_mae),
                    );
                    ui.end_row();
                });
        });

        ui.add_space(8.0);

        // Per-fold MAE chart (three lines)
        if results.fold_maes.len() > 1 {
            ui.label("Per-fold MAE over time:");
            ui.add_space(4.0);

            let vol_data: Vec<[f64; 2]> = results
                .fold_maes
                .iter()
                .enumerate()
                .map(|(i, &(v, _, _))| [i as f64, v])
                .collect();
            let entropy_data: Vec<[f64; 2]> = results
                .fold_maes
                .iter()
                .enumerate()
                .map(|(i, &(_, e, _))| [i as f64, e])
                .collect();
            let kurtosis_data: Vec<[f64; 2]> = results
                .fold_maes
                .iter()
                .enumerate()
                .map(|(i, &(_, _, k))| [i as f64, k])
                .collect();

            let hover_series = [
                HoverSeries { name: "Vol MAE",     data: &vol_data,     decimals: 6, suffix: "" },
                HoverSeries { name: "Entropy MAE", data: &entropy_data, decimals: 6, suffix: "" },
                HoverSeries { name: "Kurt MAE",    data: &kurtosis_data, decimals: 6, suffix: "" },
            ];

            let vol_plot = vol_data.clone();
            let entropy_plot = entropy_data.clone();
            let kurtosis_plot = kurtosis_data.clone();

            chart_utils::plot_with_y_drag(
                ui,
                "backtest_fold_mae",
                chart_utils::default_plot_interaction(
                    Plot::new("backtest_fold_mae")
                        .height(200.0),
                )
                .x_axis_label("Fold")
                .y_axis_label("MAE")
                .coordinates_formatter(
                    chart_utils::HOVER_CORNER,
                    chart_utils::hover_formatter(&hover_series),
                )
                .label_formatter(chart_utils::no_hover_label),
                |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(vol_plot))
                            .name("Vol MAE")
                            .color(egui::Color32::from_rgb(255, 160, 80)),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::from(entropy_plot))
                            .name("Entropy MAE")
                            .color(egui::Color32::from_rgb(100, 180, 255)),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::from(kurtosis_plot))
                            .name("Kurtosis/Skew MAE")
                            .color(egui::Color32::from_rgb(180, 130, 255)),
                    );
                },
            );
        }
    });
}

fn start_backtest(state: &mut AppState) {
    let progress = TrainingProgress::new();
    state.backtest_progress = Some(progress.clone());
    state.backtest_status = BacktestStatus::Running { fold: 0, total_folds: 0 };
    state.backtest_results = None;

    let market_data = state.market_data.clone();
    let use_gpu = state.use_gpu;
    let feature_flags = state.nn_feature_flags.clone();

    std::thread::spawn(move || {
        crate::nn::training::run_backtest(&market_data, &progress, use_gpu, &feature_flags);
    });
}

fn start_training(state: &mut AppState) {
    let progress = TrainingProgress::new();
    state.training_progress = Some(progress.clone());
    state.training_status = TrainingStatus::Training {
        epoch: 0,
        total_epochs: crate::config::NN_EPOCHS,
        loss: f64::NAN,
    };
    state.training_losses.clear();
    state.nn_predictions = crate::data::models::NnPredictions::default();

    let market_data = state.market_data.clone();
    let use_gpu = state.use_gpu;
    let feature_flags = state.nn_feature_flags.clone();

    std::thread::spawn(move || {
        crate::nn::training::train(&market_data, &progress, use_gpu, &feature_flags);
    });
}
