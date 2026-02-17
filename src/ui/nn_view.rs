use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::app::AppState;
use crate::data::models::TrainingStatus;
use crate::nn::training::TrainingProgress;

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
        ui.label("Input: 26 features (11 sector vols + 11 returns + cross-corr + spread + slope + VIX-proxy)");
        ui.label("Output: 5-day forward realized volatility prediction");
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

    // Device selector (only when idle or complete -- can't switch mid-training)
    let is_training = matches!(
        state.training_status,
        TrainingStatus::Training { .. } | TrainingStatus::Paused { .. }
    );
    if !is_training {
        ui.horizontal(|ui| {
            ui.label("Training Device:");
            ui.selectable_value(&mut state.use_gpu, true, "GPU (WGPU)");
            ui.selectable_value(&mut state.use_gpu, false, "CPU (NdArray)");

            // Show quick GPU detection status
            if state.compute_stats.gpu_detected {
                ui.colored_label(
                    egui::Color32::from_rgb(50, 180, 50),
                    format!(
                        "  Detected: {}",
                        state
                            .compute_stats
                            .gpu_name
                            .as_deref()
                            .unwrap_or("Unknown GPU")
                    ),
                );
            } else {
                // Run a quick detection on first render
                if !state.compute_stats.gpu_detected && state.compute_stats.gpu_name.is_none() {
                    if let Some(info) = crate::nn::gpu::detect_nvidia_gpu() {
                        state.compute_stats.gpu_detected = true;
                        state.compute_stats.gpu_name = Some(info.name);
                        state.compute_stats.gpu_vram_total_mb = Some(info.vram_total_mb);
                    } else {
                        // Mark as checked but not found
                        state.compute_stats.gpu_name = Some("None".to_string());
                    }
                }

                if state.compute_stats.gpu_name.as_deref() == Some("None") {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 100, 50),
                        "  No NVIDIA GPU detected (nvidia-smi not found)",
                    );
                }
            }
        });
        ui.add_space(4.0);
    }

    // Training controls
    ui.horizontal(|ui| {
        match &state.training_status {
            TrainingStatus::Idle => {
                if ui.button("Train Model").clicked() {
                    start_training(state);
                }
            }
            TrainingStatus::Training {
                epoch,
                total_epochs,
                loss,
            } => {
                ui.spinner();
                ui.label(format!(
                    "Training... Epoch {}/{} | Loss: {:.6}",
                    epoch, total_epochs, loss
                ));
                let frac = *epoch as f32 / *total_epochs as f32;
                ui.add(egui::ProgressBar::new(frac).show_percentage());

                if ui.button("Pause").clicked() {
                    if let Some(ref progress) = state.training_progress {
                        progress.request_pause();
                    }
                }

                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(200));
            }
            TrainingStatus::Paused {
                epoch,
                total_epochs,
                loss,
            } => {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 180, 50),
                    format!(
                        "Paused at Epoch {}/{} | Loss: {:.6}",
                        epoch, total_epochs, loss
                    ),
                );
                let frac = *epoch as f32 / *total_epochs as f32;
                ui.add(egui::ProgressBar::new(frac).show_percentage());

                if ui.button("Resume").clicked() {
                    if let Some(ref progress) = state.training_progress {
                        progress.request_resume();
                    }
                }
                if ui.button("Stop").clicked() {
                    state.training_status = TrainingStatus::Idle;
                    state.training_progress = None;
                }

                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(300));
            }
            TrainingStatus::Complete { final_loss } => {
                ui.colored_label(
                    egui::Color32::from_rgb(50, 180, 50),
                    format!("Training complete! Final loss: {:.6}", final_loss),
                );
                if ui.button("Retrain").clicked() {
                    state.training_status = TrainingStatus::Idle;
                    state.training_losses.clear();
                    state.nn_predictions.clear();
                    state.training_progress = None;
                }
            }
            TrainingStatus::Error(msg) => {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 50, 50),
                    format!("Error: {}", msg),
                );
                if ui.button("Retry").clicked() {
                    state.training_status = TrainingStatus::Idle;
                    state.training_progress = None;
                }
            }
        }
    });

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
        let loss_points: PlotPoints = state
            .training_losses
            .iter()
            .enumerate()
            .map(|(i, l)| [i as f64, *l])
            .collect();

        Plot::new("loss_plot")
            .height(200.0)
            .allow_drag(true)
            .allow_zoom(true)
            .x_axis_label("Epoch")
            .y_axis_label("MSE Loss")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(loss_points)
                        .name("Training Loss")
                        .color(egui::Color32::from_rgb(255, 100, 100)),
                );
            });
    }

    ui.add_space(8.0);

    // Predictions table
    if !state.nn_predictions.is_empty() {
        ui.heading("Predictions (5-Day Forward Vol)");
        ui.add_space(4.0);

        egui::Grid::new("predictions_table")
            .striped(true)
            .min_col_width(100.0)
            .show(ui, |ui| {
                ui.strong("Sector");
                ui.strong("Predicted Vol (%)");
                ui.end_row();

                for (sector, vol) in &state.nn_predictions {
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
    } else if matches!(state.training_status, TrainingStatus::Idle) {
        ui.add_space(8.0);
        ui.label("No predictions yet. Train the model to generate predictions.");
    }

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
                        ui.strong(&stats.backend_name);
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

fn start_training(state: &mut AppState) {
    let progress = TrainingProgress::new();
    state.training_progress = Some(progress.clone());
    state.training_status = TrainingStatus::Training {
        epoch: 0,
        total_epochs: crate::config::NN_EPOCHS,
        loss: f64::NAN,
    };
    state.training_losses.clear();
    state.nn_predictions.clear();

    let market_data = state.market_data.clone();
    let use_gpu = state.use_gpu;

    std::thread::spawn(move || {
        crate::nn::training::train(&market_data, &progress, use_gpu);
    });
}
