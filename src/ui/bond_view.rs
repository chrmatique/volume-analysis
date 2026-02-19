use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};

use crate::analysis::bond_spreads;
use crate::app::AppState;
use crate::ui::chart_utils::height_control;

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Bond Spreads & Yield Curve");
    ui.add_space(8.0);

    if state.market_data.treasury_rates.is_empty() {
        ui.label("No treasury rate data available. Load market data first.");
        return;
    }

    // Yield curve for the latest date
    ui.collapsing("Current Yield Curve", |ui| {
        if let Some(latest_rate) = state.market_data.treasury_rates.first() {
            ui.label(format!("Date: {}", latest_rate.date));

            let curve = bond_spreads::yield_curve_for_date(latest_rate);
            if !curve.is_empty() {
                let bars: Vec<Bar> = curve
                    .iter()
                    .enumerate()
                    .map(|(i, (_label, rate))| {
                        Bar::new(i as f64, *rate)
                            .width(0.6)
                    })
                    .collect();

                height_control(ui, &mut state.chart_heights.bond_yield_curve, "Yield Curve Chart Height");
                Plot::new("yield_curve")
                    .height(state.chart_heights.bond_yield_curve)
                    .allow_drag(true)
                    .allow_zoom(true)
                    .y_axis_label("Yield (%)")
                    .show(ui, |plot_ui| {
                        plot_ui.bar_chart(
                            BarChart::new(bars)
                                .name("Yield Curve")
                                .color(egui::Color32::from_rgb(70, 130, 220)),
                        );
                    });

                // Show maturity labels
                ui.horizontal_wrapped(|ui| {
                    for (i, (label, rate)) in curve.iter().enumerate() {
                        ui.label(format!("[{}] {} = {:.2}%", i, label, rate));
                    }
                });
            }
        }
    });

    ui.add_space(8.0);

    // Term spread (10Y-2Y) over time
    if !state.analysis.bond_spreads.is_empty() {
        ui.heading("10Y-2Y Term Spread Over Time");
        ui.add_space(4.0);

        let spread_points: PlotPoints = state
            .analysis
            .bond_spreads
            .iter()
            .rev()
            .enumerate()
            .map(|(i, s)| [i as f64, s.spread_10y_2y])
            .collect();

        let zero_line: PlotPoints = PlotPoints::from_iter(
            (0..state.analysis.bond_spreads.len()).map(|i| [i as f64, 0.0]),
        );

        height_control(ui, &mut state.chart_heights.bond_term_spread, "Term Spread Chart Height");
        Plot::new("term_spread_plot")
            .height(state.chart_heights.bond_term_spread)
            .allow_drag(true)
            .allow_zoom(true)
            .x_axis_label("Trading Day (recent -> past)")
            .y_axis_label("Spread (percentage points)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(spread_points)
                        .name("10Y-2Y Spread")
                        .color(egui::Color32::from_rgb(255, 150, 50)),
                );
                plot_ui.line(
                    Line::new(zero_line)
                        .name("Zero")
                        .color(egui::Color32::from_rgb(150, 150, 150))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            });

        // Curve slope over time
        ui.add_space(8.0);
        ui.heading("Curve Slope (30Y - 3M)");
        ui.add_space(4.0);

        let slope_points: PlotPoints = state
            .analysis
            .bond_spreads
            .iter()
            .rev()
            .enumerate()
            .map(|(i, s)| [i as f64, s.curve_slope])
            .collect();

        height_control(ui, &mut state.chart_heights.bond_curve_slope, "Curve Slope Chart Height");
        Plot::new("curve_slope_plot")
            .height(state.chart_heights.bond_curve_slope)
            .allow_drag(true)
            .allow_zoom(true)
            .x_axis_label("Trading Day (recent -> past)")
            .y_axis_label("Slope (percentage points)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(slope_points)
                        .name("30Y-3M Slope")
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
            });

        // Summary
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        if let Some(latest) = state.analysis.bond_spreads.first() {
            ui.label(format!(
                "Latest ({}) | 10Y-2Y: {:.2} pp | Curve Slope: {:.2} pp | {}",
                latest.date,
                latest.spread_10y_2y,
                latest.curve_slope,
                if latest.spread_10y_2y < 0.0 {
                    "INVERTED"
                } else {
                    "Normal"
                }
            ));
        }

        // Inversion detection
        let inversions = bond_spreads::detect_inversions(&state.market_data.treasury_rates);
        if !inversions.is_empty() {
            ui.add_space(4.0);
            ui.colored_label(
                egui::Color32::from_rgb(220, 50, 50),
                format!(
                    "Warning: {} yield curve inversion(s) detected in data",
                    inversions.len()
                ),
            );
        }
    }
}
