use chrono::Datelike;
use eframe::egui;
use egui_plot::{BoxElem, BoxPlot, BoxSpread, Line, Plot, PlotPoints};

use crate::app::{AppState, CandleTimescale};
use crate::config;
use crate::data::models::OhlcvBar;
use crate::ui::chart_utils::{self, height_control, HoverSeries};

fn aggregate_group(bars: &[&OhlcvBar]) -> OhlcvBar {
    OhlcvBar {
        date: bars[0].date,
        open: bars[0].open,
        high: bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max),
        low: bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
        close: bars.last().unwrap().close,
        volume: bars.iter().map(|b| b.volume).sum(),
    }
}

fn aggregate_bars(bars: &[OhlcvBar], timescale: CandleTimescale) -> Vec<OhlcvBar> {
    match timescale {
        CandleTimescale::Day => bars.to_vec(),
        CandleTimescale::Week => {
            let mut result = Vec::new();
            let mut group: Vec<&OhlcvBar> = Vec::new();
            let mut current_key: Option<(i32, u32)> = None;
            for bar in bars {
                let key = (bar.date.year(), bar.date.iso_week().week());
                if Some(key) != current_key {
                    if !group.is_empty() {
                        result.push(aggregate_group(&group));
                    }
                    group.clear();
                    current_key = Some(key);
                }
                group.push(bar);
            }
            if !group.is_empty() {
                result.push(aggregate_group(&group));
            }
            result
        }
        CandleTimescale::Month => {
            let mut result = Vec::new();
            let mut group: Vec<&OhlcvBar> = Vec::new();
            let mut current_key: Option<(i32, u32)> = None;
            for bar in bars {
                let key = (bar.date.year(), bar.date.month());
                if Some(key) != current_key {
                    if !group.is_empty() {
                        result.push(aggregate_group(&group));
                    }
                    group.clear();
                    current_key = Some(key);
                }
                group.push(bar);
            }
            if !group.is_empty() {
                result.push(aggregate_group(&group));
            }
            result
        }
    }
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Sector Volatility Analysis");
    ui.add_space(8.0);

    if state.market_data.sectors.is_empty() {
        ui.label("No data loaded. Click 'Refresh Data' to fetch market data.");
        return;
    }

    // Sector selector + timescale
    ui.horizontal(|ui| {
        ui.label("Select Sector:");
        egui::ComboBox::from_id_salt("sector_selector")
            .selected_text(
                state
                    .market_data
                    .sectors
                    .get(state.selected_sector_idx)
                    .map(|s| {
                        let name = config::SECTOR_ETFS
                            .iter()
                            .find(|(sym, _)| *sym == s.symbol)
                            .map(|(_, n)| *n)
                            .unwrap_or("Unknown");
                        format!("{} ({})", name, s.symbol)
                    })
                    .unwrap_or_else(|| "Select...".to_string()),
            )
            .show_ui(ui, |ui| {
                for (i, sector) in state.market_data.sectors.iter().enumerate() {
                    let name = config::SECTOR_ETFS
                        .iter()
                        .find(|(sym, _)| *sym == sector.symbol)
                        .map(|(_, n)| *n)
                        .unwrap_or("Unknown");
                    ui.selectable_value(
                        &mut state.selected_sector_idx,
                        i,
                        format!("{} ({})", name, sector.symbol),
                    );
                }
            });

        ui.add_space(16.0);
        ui.label("Candle Interval:");
        egui::ComboBox::from_id_salt("candle_timescale")
            .selected_text(state.candle_timescale.label())
            .show_ui(ui, |ui| {
                for &ts in CandleTimescale::all() {
                    ui.selectable_value(&mut state.candle_timescale, ts, ts.label());
                }
            });
    });

    ui.add_space(8.0);

    let sector = match state.market_data.sectors.get(state.selected_sector_idx) {
        Some(s) => s,
        None => return,
    };

    let vol_metrics = state
        .analysis
        .volatility
        .iter()
        .find(|v| v.symbol == sector.symbol);

    // Price chart
    ui.collapsing("Price Chart", |ui| {
        height_control(ui, &mut state.chart_heights.sector_price, "Price Chart Height");

        let agg_bars = aggregate_bars(&sector.bars, state.candle_timescale);

        let price_data: Vec<[f64; 2]> = agg_bars
            .iter()
            .enumerate()
            .map(|(i, b)| [i as f64, b.close])
            .collect();
        let hover = [HoverSeries { name: &sector.symbol, data: &price_data, decimals: 2, suffix: "" }];

        let box_width = match state.candle_timescale {
            CandleTimescale::Day => 0.7,
            CandleTimescale::Week => 0.75,
            CandleTimescale::Month => 0.8,
        };

        let x_label = match state.candle_timescale {
            CandleTimescale::Day => "Trading Day",
            CandleTimescale::Week => "Week",
            CandleTimescale::Month => "Month",
        };

        let candles: Vec<BoxElem> = agg_bars
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let (body_lo, body_hi) = if b.close >= b.open {
                    (b.open, b.close)
                } else {
                    (b.close, b.open)
                };
                let fill = if b.close >= b.open {
                    egui::Color32::from_rgb(38, 166, 91)
                } else {
                    egui::Color32::from_rgb(214, 48, 49)
                };
                BoxElem::new(
                    i as f64,
                    BoxSpread::new(b.low, body_lo, b.close, body_hi, b.high),
                )
                .box_width(box_width)
                .whisker_width(box_width * 0.4)
                .fill(fill)
                .stroke(egui::Stroke::new(1.0, fill))
            })
            .collect();

        chart_utils::plot_with_xy_zoom(
            ui,
            "price_plot",
            Plot::new("price_plot")
                .height(state.chart_heights.sector_price)
                .allow_drag(egui::Vec2b::new(true, true))
                .allow_scroll(false)
                .allow_zoom(egui::Vec2b::new(true, true))
                .x_axis_label(x_label)
                .y_axis_label("Price ($)")
                .coordinates_formatter(chart_utils::HOVER_CORNER, chart_utils::hover_formatter(&hover))
                .label_formatter(chart_utils::no_hover_label),
            |plot_ui| {
                plot_ui.box_plot(
                    BoxPlot::new(candles)
                        .name(&sector.symbol)
                        .vertical(),
                );
            },
        );
    });

    ui.add_space(8.0);

    // Volatility chart
    if let Some(vm) = vol_metrics {
        ui.label(format!(
            "Showing {}-day and {}-day rolling volatility",
            config::SHORT_VOL_WINDOW,
            config::LONG_VOL_WINDOW
        ));

        let short_data: Vec<[f64; 2]> = vm
            .short_window_vol
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64, *v * 100.0])
            .collect();
        let short_points: PlotPoints = short_data.iter().copied().collect();

        let long_data: Vec<[f64; 2]> = vm
            .long_window_vol
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64, *v * 100.0])
            .collect();
        let long_points: PlotPoints = long_data.iter().copied().collect();

        let park_data: Vec<[f64; 2]> = vm
            .parkinson_vol
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64, *v * 100.0])
            .collect();
        let park_points: PlotPoints = park_data.iter().copied().collect();

        let short_name = format!("{}D Vol", config::SHORT_VOL_WINDOW);
        let long_name = format!("{}D Vol", config::LONG_VOL_WINDOW);
        let vol_hover = [
            HoverSeries { name: &short_name, data: &short_data, decimals: 1, suffix: "%" },
            HoverSeries { name: &long_name, data: &long_data, decimals: 1, suffix: "%" },
            HoverSeries { name: "Parkinson Vol", data: &park_data, decimals: 1, suffix: "%" },
        ];

        height_control(ui, &mut state.chart_heights.sector_vol, "Volatility Chart Height");
        chart_utils::plot_with_y_drag(
            ui,
            "vol_plot",
            chart_utils::default_plot_interaction(
                Plot::new("vol_plot")
                    .height(state.chart_heights.sector_vol),
            )
                .x_axis_label("Trading Day (aligned)")
                .y_axis_label("Annualized Vol (%)")
                .legend(egui_plot::Legend::default())
                .coordinates_formatter(chart_utils::HOVER_CORNER, chart_utils::hover_formatter(&vol_hover))
                .label_formatter(chart_utils::no_hover_label),
            |plot_ui| {
                plot_ui.line(
                    Line::new(short_points)
                        .name(format!("{}D Vol", config::SHORT_VOL_WINDOW))
                        .color(egui::Color32::from_rgb(255, 100, 100)),
                );
                plot_ui.line(
                    Line::new(long_points)
                        .name(format!("{}D Vol", config::LONG_VOL_WINDOW))
                        .color(egui::Color32::from_rgb(100, 100, 255)),
                );
                plot_ui.line(
                    Line::new(park_points)
                        .name("Parkinson Vol")
                        .color(egui::Color32::from_rgb(100, 220, 100)),
                );
            },
        );

        // Vol ratio chart
        ui.add_space(8.0);
        ui.label("Volatility Ratio (Short / Long) - above 1.0 indicates rising vol regime");

        let ratio_data: Vec<[f64; 2]> = vm
            .vol_ratio
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64, *v])
            .collect();
        let ratio_points: PlotPoints = ratio_data.iter().copied().collect();

        let baseline: PlotPoints = PlotPoints::from_iter(
            (0..vm.vol_ratio.len()).map(|i| [i as f64, 1.0]),
        );

        let ratio_hover = [HoverSeries { name: "Vol Ratio", data: &ratio_data, decimals: 2, suffix: "" }];

        height_control(ui, &mut state.chart_heights.sector_ratio, "Vol Ratio Chart Height");
        chart_utils::plot_with_y_drag(
            ui,
            "ratio_plot",
            chart_utils::default_plot_interaction(
                Plot::new("ratio_plot")
                    .height(state.chart_heights.sector_ratio),
            )
                .x_axis_label("Trading Day (aligned)")
                .y_axis_label("Vol Ratio")
                .coordinates_formatter(chart_utils::HOVER_CORNER, chart_utils::hover_formatter(&ratio_hover))
                .label_formatter(chart_utils::no_hover_label),
            |plot_ui| {
                plot_ui.line(
                    Line::new(ratio_points)
                        .name("Vol Ratio")
                        .color(egui::Color32::from_rgb(255, 180, 50)),
                );
                plot_ui.line(
                    Line::new(baseline)
                        .name("Baseline (1.0)")
                        .color(egui::Color32::from_rgb(150, 150, 150))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            },
        );

        // Summary stats
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);

        if let (Some(sv), Some(lv), Some(vr)) = (
            vm.short_window_vol.last(),
            vm.long_window_vol.last(),
            vm.vol_ratio.last(),
        ) {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Latest: {}D Vol = {:.1}% | {}D Vol = {:.1}% | Ratio = {:.2}",
                    config::SHORT_VOL_WINDOW,
                    sv * 100.0,
                    config::LONG_VOL_WINDOW,
                    lv * 100.0,
                    vr
                ));
            });
        }
    } else {
        ui.label("No volatility data computed for this sector yet.");
    }
}
