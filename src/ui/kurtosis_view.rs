use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::app::AppState;
use crate::ui::chart_utils::height_control;

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Kurtosis & Return Distribution Analysis");
    ui.add_space(8.0);

    if state.analysis.kurtosis.is_empty() {
        ui.label("Load market data first to compute distribution statistics.");
        return;
    }

    // Sector selector
    let symbols: Vec<String> = state.analysis.kurtosis.iter().map(|k| k.symbol.clone()).collect();
    let selected = state.selected_sector_idx.min(symbols.len().saturating_sub(1));

    ui.horizontal(|ui| {
        ui.label("Sector:");
        for (i, sym) in symbols.iter().enumerate() {
            if ui.selectable_label(selected == i, sym).clicked() {
                state.selected_sector_idx = i;
            }
        }
    });

    ui.add_space(8.0);

    let selected = state.selected_sector_idx.min(symbols.len().saturating_sub(1));
    let metrics = &state.analysis.kurtosis[selected];

    // Summary statistics cards
    ui.group(|ui| {
        ui.strong(format!("{} - Distribution Statistics", metrics.symbol));
        ui.add_space(4.0);

        ui.columns(6, |cols| {
            stat_card(&mut cols[0], "Mean", &format!("{:.4}%", metrics.mean * 100.0));
            stat_card(&mut cols[1], "Std Dev", &format!("{:.4}%", metrics.std_dev * 100.0));

            let kurt_color = if metrics.excess_kurtosis.abs() > 3.0 {
                egui::Color32::from_rgb(220, 50, 50)
            } else if metrics.excess_kurtosis.abs() > 1.0 {
                egui::Color32::from_rgb(220, 180, 50)
            } else {
                egui::Color32::from_rgb(50, 180, 50)
            };
            stat_card_colored(&mut cols[2], "Excess Kurtosis", &format!("{:.3}", metrics.excess_kurtosis), kurt_color);

            let skew_color = if metrics.skewness.abs() > 1.0 {
                egui::Color32::from_rgb(220, 50, 50)
            } else if metrics.skewness.abs() > 0.5 {
                egui::Color32::from_rgb(220, 180, 50)
            } else {
                egui::Color32::from_rgb(50, 180, 50)
            };
            stat_card_colored(&mut cols[3], "Skewness", &format!("{:.3}", metrics.skewness), skew_color);

            let jb_color = if metrics.jarque_bera > 10.0 {
                egui::Color32::from_rgb(220, 50, 50)
            } else {
                egui::Color32::from_rgb(50, 180, 50)
            };
            stat_card_colored(&mut cols[4], "Jarque-Bera", &format!("{:.1}", metrics.jarque_bera), jb_color);

            let tail_label = if metrics.excess_kurtosis > 1.0 {
                "Leptokurtic (fat tails)"
            } else if metrics.excess_kurtosis < -1.0 {
                "Platykurtic (thin tails)"
            } else {
                "Mesokurtic (near normal)"
            };
            stat_card(&mut cols[5], "Tail Type", tail_label);
        });
    });

    ui.add_space(8.0);

    // Distribution curve: empirical density vs fitted normal
    if !metrics.empirical_density.is_empty() {
        ui.heading("Return Distribution (KDE vs Normal Fit)");
        ui.add_space(4.0);

        let empirical_points: PlotPoints = metrics
            .empirical_density
            .iter()
            .map(|p| [p[0] * 100.0, p[1] / 100.0]) // convert x to % scale, adjust density
            .collect();

        let normal_points: PlotPoints = metrics
            .normal_density
            .iter()
            .map(|p| [p[0] * 100.0, p[1] / 100.0])
            .collect();

        height_control(ui, &mut state.chart_heights.kurtosis_distribution, "Distribution Plot Height");
        Plot::new("distribution_plot")
            .height(state.chart_heights.kurtosis_distribution)
            .allow_drag(true)
            .allow_zoom(true)
            .x_axis_label("Daily Log Return (%)")
            .y_axis_label("Density")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(empirical_points)
                        .name("Empirical (KDE)")
                        .color(egui::Color32::from_rgb(80, 160, 255))
                        .width(2.5),
                );
                plot_ui.line(
                    Line::new(normal_points)
                        .name("Normal Fit")
                        .color(egui::Color32::from_rgb(255, 100, 100))
                        .width(2.0)
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            });

        ui.add_space(4.0);
        ui.small("Blue = empirical density (Gaussian KDE). Red dashed = fitted normal distribution. Fat tails appear where the blue curve extends beyond the red.");
    }

    ui.add_space(12.0);

    // Rolling kurtosis over time
    if !metrics.rolling_kurtosis.is_empty() && !metrics.rolling_dates.is_empty() {
        ui.heading("Rolling Excess Kurtosis (63-day window)");
        ui.add_space(4.0);

        let base_date = metrics.rolling_dates.first().copied();
        let kurt_points: PlotPoints = metrics
            .rolling_kurtosis
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let day = if let Some(bd) = base_date {
                    if let Some(d) = metrics.rolling_dates.get(i) {
                        (*d - bd).num_days() as f64
                    } else {
                        i as f64
                    }
                } else {
                    i as f64
                };
                [day, *k]
            })
            .collect();

        // Reference line at 0 (normal distribution)
        let x_max = metrics
            .rolling_dates
            .last()
            .and_then(|d| base_date.map(|bd| (*d - bd).num_days() as f64))
            .unwrap_or(metrics.rolling_kurtosis.len() as f64);
        let zero_line: PlotPoints = vec![[0.0, 0.0], [x_max, 0.0]].into_iter().collect();

        height_control(ui, &mut state.chart_heights.kurtosis_rolling_kurtosis, "Rolling Kurtosis Chart Height");
        Plot::new("rolling_kurtosis_plot")
            .height(state.chart_heights.kurtosis_rolling_kurtosis)
            .allow_drag(true)
            .allow_zoom(true)
            .x_axis_label("Trading Days")
            .y_axis_label("Excess Kurtosis")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(kurt_points)
                        .name("Rolling Kurtosis")
                        .color(egui::Color32::from_rgb(180, 100, 255))
                        .width(1.8),
                );
                plot_ui.line(
                    Line::new(zero_line)
                        .name("Normal (0)")
                        .color(egui::Color32::from_rgb(150, 150, 150))
                        .style(egui_plot::LineStyle::dashed_loose()),
                );
            });
    }

    ui.add_space(12.0);

    // Rolling skewness over time
    if !metrics.rolling_skewness.is_empty() && !metrics.rolling_dates.is_empty() {
        ui.heading("Rolling Skewness (63-day window)");
        ui.add_space(4.0);

        let base_date = metrics.rolling_dates.first().copied();
        let skew_points: PlotPoints = metrics
            .rolling_skewness
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let day = if let Some(bd) = base_date {
                    if let Some(d) = metrics.rolling_dates.get(i) {
                        (*d - bd).num_days() as f64
                    } else {
                        i as f64
                    }
                } else {
                    i as f64
                };
                [day, *s]
            })
            .collect();

        let x_max = metrics
            .rolling_dates
            .last()
            .and_then(|d| base_date.map(|bd| (*d - bd).num_days() as f64))
            .unwrap_or(metrics.rolling_skewness.len() as f64);
        let zero_line: PlotPoints = vec![[0.0, 0.0], [x_max, 0.0]].into_iter().collect();

        height_control(ui, &mut state.chart_heights.kurtosis_rolling_skewness, "Rolling Skewness Chart Height");
        Plot::new("rolling_skewness_plot")
            .height(state.chart_heights.kurtosis_rolling_skewness)
            .allow_drag(true)
            .allow_zoom(true)
            .x_axis_label("Trading Days")
            .y_axis_label("Skewness")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(skew_points)
                        .name("Rolling Skewness")
                        .color(egui::Color32::from_rgb(255, 180, 50))
                        .width(1.8),
                );
                plot_ui.line(
                    Line::new(zero_line)
                        .name("Symmetric (0)")
                        .color(egui::Color32::from_rgb(150, 150, 150))
                        .style(egui_plot::LineStyle::dashed_loose()),
                );
            });
    }

    ui.add_space(12.0);

    // Cross-sector kurtosis comparison table
    if state.analysis.kurtosis.len() > 1 {
        ui.heading("Cross-Sector Kurtosis Comparison");
        ui.add_space(4.0);

        egui::Grid::new("kurtosis_table")
            .striped(true)
            .min_col_width(80.0)
            .show(ui, |ui| {
                ui.strong("Sector");
                ui.strong("Excess Kurtosis");
                ui.strong("Skewness");
                ui.strong("Jarque-Bera");
                ui.strong("Tail Type");
                ui.end_row();

                let mut sorted: Vec<_> = state.analysis.kurtosis.iter().collect();
                sorted.sort_by(|a, b| b.excess_kurtosis.partial_cmp(&a.excess_kurtosis).unwrap_or(std::cmp::Ordering::Equal));

                for km in &sorted {
                    ui.label(&km.symbol);

                    let kurt_color = kurt_color(km.excess_kurtosis);
                    ui.colored_label(kurt_color, format!("{:.3}", km.excess_kurtosis));

                    let skew_color = skew_color(km.skewness);
                    ui.colored_label(skew_color, format!("{:.3}", km.skewness));

                    let jb_color = if km.jarque_bera > 10.0 {
                        egui::Color32::from_rgb(220, 50, 50)
                    } else {
                        egui::Color32::from_rgb(50, 180, 50)
                    };
                    ui.colored_label(jb_color, format!("{:.1}", km.jarque_bera));

                    let tail = if km.excess_kurtosis > 1.0 {
                        "Fat tails"
                    } else if km.excess_kurtosis < -1.0 {
                        "Thin tails"
                    } else {
                        "Normal-like"
                    };
                    ui.label(tail);
                    ui.end_row();
                }
            });
    }

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(4.0);
    ui.small("Excess kurtosis > 0 indicates heavier tails than a normal distribution (leptokurtic). The Jarque-Bera test statistic measures departure from normality (critical value ~5.99 at 5% significance).");
}

fn stat_card(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.vertical(|ui| {
        ui.small(label);
        ui.strong(value);
    });
}

fn stat_card_colored(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    ui.vertical(|ui| {
        ui.small(label);
        ui.colored_label(color, value);
    });
}

fn kurt_color(k: f64) -> egui::Color32 {
    if k.abs() > 3.0 {
        egui::Color32::from_rgb(220, 50, 50)
    } else if k.abs() > 1.0 {
        egui::Color32::from_rgb(220, 180, 50)
    } else {
        egui::Color32::from_rgb(50, 180, 50)
    }
}

fn skew_color(s: f64) -> egui::Color32 {
    if s.abs() > 1.0 {
        egui::Color32::from_rgb(220, 50, 50)
    } else if s.abs() > 0.5 {
        egui::Color32::from_rgb(220, 180, 50)
    } else {
        egui::Color32::from_rgb(50, 180, 50)
    }
}
