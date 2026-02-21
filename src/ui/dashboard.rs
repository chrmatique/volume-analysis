use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::app::AppState;
use crate::ui::chart_utils::height_control;
use crate::config;

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Market Structure Dashboard");
    ui.add_space(8.0);

    if state.market_data.sectors.is_empty() {
        ui.label("No data loaded. Click 'Refresh Data' to fetch market data.");
        return;
    }

    // Key metrics row
    ui.horizontal(|ui| {
        let n_sectors = state.market_data.sectors.len();
        metric_card(ui, "Sectors Loaded", &format!("{}", n_sectors));

        if let Some(ref bench) = state.market_data.benchmark {
            if let Some(last) = bench.bars.last() {
                metric_card(ui, "SPY Last Close", &format!("${:.2}", last.close));
            }
        }

        metric_card(
            ui,
            "Avg Cross-Correlation",
            &format!("{:.3}", state.analysis.avg_cross_correlation),
        );

        if let Some(spread) = state.analysis.bond_spreads.first() {
            metric_card(
                ui,
                "10Y-2Y Spread",
                &format!("{:.2} bps", spread.spread_10y_2y * 100.0),
            );
        }

        let n_rates = state.market_data.treasury_rates.len();
        metric_card(ui, "Treasury Data Points", &format!("{}", n_rates));
    });

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    // 3D Market Randomness Distribution
    if state.market_data.sectors.len() >= 2 {
        render_3d_section(ui, state);

        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);
    }

    // Sector heatmap
    ui.heading("Sector Volatility Heatmap");
    ui.add_space(8.0);

    egui::Grid::new("sector_heatmap")
        .striped(true)
        .min_col_width(100.0)
        .show(ui, |ui| {
            ui.strong("Sector");
            ui.strong("Symbol");
            ui.strong("Last Close");
            ui.strong("21D Vol");
            ui.strong("63D Vol");
            ui.strong("Vol Ratio");
            ui.strong("Bars");
            ui.end_row();

            for (i, sector) in state.market_data.sectors.iter().enumerate() {
                let name = config::SECTOR_ETFS
                    .iter()
                    .find(|(s, _)| *s == sector.symbol)
                    .map(|(_, n)| *n)
                    .unwrap_or("Unknown");

                ui.label(name);
                ui.label(&sector.symbol);

                if let Some(last) = sector.bars.last() {
                    ui.label(format!("${:.2}", last.close));
                } else {
                    ui.label("-");
                }

                if let Some(vm) = state
                    .analysis
                    .volatility
                    .iter()
                    .find(|v| v.symbol == sector.symbol)
                {
                    let sv = vm.short_window_vol.last().copied().unwrap_or(0.0);
                    let lv = vm.long_window_vol.last().copied().unwrap_or(0.0);
                    let vr = vm.vol_ratio.last().copied().unwrap_or(0.0);

                    let vol_color = vol_to_color(sv);
                    ui.colored_label(vol_color, format!("{:.1}%", sv * 100.0));
                    ui.colored_label(vol_to_color(lv), format!("{:.1}%", lv * 100.0));

                    let ratio_color = if vr > 1.2 {
                        egui::Color32::from_rgb(220, 50, 50)
                    } else if vr < 0.8 {
                        egui::Color32::from_rgb(50, 180, 50)
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.colored_label(ratio_color, format!("{:.2}", vr));
                } else {
                    ui.label("-");
                    ui.label("-");
                    ui.label("-");
                }

                ui.label(format!("{}", sector.bars.len()));
                ui.end_row();

                if i == state.selected_sector_idx {
                    // selection indicator is handled by grid striping
                }
            }
        });

    // Put/Call Ratio & SKEW
    render_put_call_skew_section(ui, state);

    // FMP sector performance
    if !state.market_data.sector_performance.is_empty() {
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);
        ui.heading("FMP Sector Performance (Real-Time)");
        ui.add_space(8.0);

        egui::Grid::new("fmp_sector_perf")
            .striped(true)
            .min_col_width(120.0)
            .show(ui, |ui| {
                ui.strong("Sector");
                ui.strong("Change %");
                ui.end_row();

                for sp in &state.market_data.sector_performance {
                    ui.label(&sp.sector);
                    let color = if sp.changes_percentage >= 0.0 {
                        egui::Color32::from_rgb(50, 180, 50)
                    } else {
                        egui::Color32::from_rgb(220, 50, 50)
                    };
                    ui.colored_label(color, format!("{:+.2}%", sp.changes_percentage));
                    ui.end_row();
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Put/Call Ratio & SKEW section
// ---------------------------------------------------------------------------

fn render_put_call_skew_section(ui: &mut egui::Ui, state: &mut AppState) {
    let has_pc = !state.market_data.put_call_ratio.is_empty();
    let has_skew = !state.market_data.skew_history.is_empty();

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);
    ui.heading("Put/Call Ratio & SKEW");
    ui.add_space(4.0);

    if !has_pc && !has_skew {
        ui.label("No Put/Call or SKEW data. Data is fetched with Refresh Data.");
        return;
    }

    if has_pc {
        let pc_points: PlotPoints = state
            .market_data
            .put_call_ratio
            .iter()
            .rev()
            .enumerate()
            .map(|(i, r)| [i as f64, r.pc_ratio])
            .collect();

        height_control(ui, &mut state.chart_heights.put_call_skew, "P/C Ratio & SKEW Chart Height");
        Plot::new("put_call_ratio_plot")
            .height(state.chart_heights.put_call_skew)
            .allow_drag(true)
            .allow_scroll(false)
            .allow_zoom(false)
            .x_axis_label("Trading Day (recent -> past)")
            .y_axis_label("P/C Ratio")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(pc_points)
                        .name("Total P/C Ratio")
                        .color(egui::Color32::from_rgb(255, 150, 50)),
                );
            });

        ui.add_space(8.0);
    }

    if has_skew {
        let skew_points: PlotPoints = state
            .market_data
            .skew_history
            .iter()
            .rev()
            .enumerate()
            .map(|(i, r)| [i as f64, r.skew])
            .collect();

        Plot::new("skew_plot")
            .height(state.chart_heights.put_call_skew)
            .allow_drag(true)
            .allow_scroll(false)
            .allow_zoom(false)
            .x_axis_label("Trading Day (recent -> past)")
            .y_axis_label("SKEW")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(skew_points)
                        .name("CBOE SKEW")
                        .color(egui::Color32::from_rgb(70, 180, 220)),
                );
            });
    }
}

// ---------------------------------------------------------------------------
// 3D Market Randomness section
// ---------------------------------------------------------------------------

fn render_3d_section(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Market Randomness - 3D Joint Return Distribution");
    ui.add_space(4.0);

    let n_sectors = state.market_data.sectors.len();
    if n_sectors < 2 {
        return;
    }

    let symbols: Vec<String> = state.market_data.sectors.iter().map(|s| s.symbol.clone()).collect();

    // Sector pair selector + rotation controls
    let mut redraw = false;
    ui.horizontal(|ui| {
        ui.label("Sector X:");
        let prev_x = state.plot_3d.sector_x_idx;
        egui::ComboBox::from_id_salt("sector_x_combo")
            .selected_text(&symbols[state.plot_3d.sector_x_idx.min(n_sectors - 1)])
            .show_ui(ui, |ui| {
                for (i, sym) in symbols.iter().enumerate() {
                    ui.selectable_value(&mut state.plot_3d.sector_x_idx, i, sym);
                }
            });
        if state.plot_3d.sector_x_idx != prev_x {
            redraw = true;
        }

        ui.label("Sector Y:");
        let prev_y = state.plot_3d.sector_y_idx;
        egui::ComboBox::from_id_salt("sector_y_combo")
            .selected_text(&symbols[state.plot_3d.sector_y_idx.min(n_sectors - 1)])
            .show_ui(ui, |ui| {
                for (i, sym) in symbols.iter().enumerate() {
                    ui.selectable_value(&mut state.plot_3d.sector_y_idx, i, sym);
                }
            });
        if state.plot_3d.sector_y_idx != prev_y {
            redraw = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Pitch:");
        let prev_pitch = state.plot_3d.pitch;
        ui.add(egui::Slider::new(&mut state.plot_3d.pitch, 0.05..=1.2).step_by(0.01));
        if (state.plot_3d.pitch - prev_pitch).abs() > 0.001 {
            redraw = true;
        }

        ui.label("Yaw:");
        let prev_yaw = state.plot_3d.yaw;
        ui.add(egui::Slider::new(&mut state.plot_3d.yaw, -1.5..=1.5).step_by(0.01));
        if (state.plot_3d.yaw - prev_yaw).abs() > 0.001 {
            redraw = true;
        }
    });

    ui.add_space(4.0);

    // Check if we need to re-render the 3D plot
    if state.plot_3d.needs_redraw || redraw {
        state.plot_3d.needs_redraw = false;

        let sx = state.plot_3d.sector_x_idx.min(n_sectors - 1);
        let sy = state.plot_3d.sector_y_idx.min(n_sectors - 1);

        let returns_x = state.market_data.sectors[sx].log_returns();
        let returns_y = state.market_data.sectors[sy].log_returns();

        let sym_x = symbols[sx].clone();
        let sym_y = symbols[sy].clone();

        let tex = render_3d_surface(
            &returns_x,
            &returns_y,
            &sym_x,
            &sym_y,
            state.plot_3d.pitch as f64,
            state.plot_3d.yaw as f64,
            640,
            480,
        );

        if let Some(rgba_data) = tex {
            let image = egui::ColorImage::from_rgba_unmultiplied([640, 480], &rgba_data);
            state.plot_3d.texture = Some(ui.ctx().load_texture(
                "3d_distribution",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    // Display the rendered texture
    if let Some(ref texture) = state.plot_3d.texture {
        let size = egui::vec2(640.0, 480.0);
        ui.image(egui::load::SizedTexture::new(texture.id(), size));
    } else {
        ui.label("Rendering 3D distribution...");
    }

    ui.add_space(8.0);

    // Randomness metrics table
    if !state.analysis.randomness.is_empty() {
        ui.heading("Sector Randomness Metrics");
        ui.add_space(4.0);

        egui::Grid::new("randomness_table")
            .striped(true)
            .min_col_width(90.0)
            .show(ui, |ui| {
                ui.strong("Sector");
                ui.strong("Entropy");
                ui.strong("Hurst Exp.");
                ui.strong("AC(1)");
                ui.strong("AC(5)");
                ui.strong("Behavior");
                ui.end_row();

                for rm in &state.analysis.randomness {
                    ui.label(&rm.symbol);

                    ui.label(format!("{:.3}", rm.entropy));

                    let hurst_color = if (rm.hurst_exponent - 0.5).abs() < 0.05 {
                        egui::Color32::from_rgb(50, 180, 50)
                    } else if rm.hurst_exponent > 0.5 {
                        egui::Color32::from_rgb(80, 160, 255)
                    } else {
                        egui::Color32::from_rgb(220, 150, 50)
                    };
                    ui.colored_label(hurst_color, format!("{:.3}", rm.hurst_exponent));

                    let ac1_color = if rm.autocorrelation_lag1.abs() > 0.1 {
                        egui::Color32::from_rgb(220, 50, 50)
                    } else {
                        egui::Color32::from_rgb(150, 150, 150)
                    };
                    ui.colored_label(ac1_color, format!("{:.4}", rm.autocorrelation_lag1));
                    ui.colored_label(ac1_color, format!("{:.4}", rm.autocorrelation_lag5));

                    let behavior = if (rm.hurst_exponent - 0.5).abs() < 0.05 {
                        "Random Walk"
                    } else if rm.hurst_exponent > 0.5 {
                        "Trending"
                    } else {
                        "Mean-Reverting"
                    };
                    ui.label(behavior);
                    ui.end_row();
                }
            });

        ui.add_space(4.0);
        ui.small("Entropy: higher = more random. Hurst: 0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting. AC = autocorrelation (near 0 = random).");
    }
}

/// Render a 3D surface plot of the joint return distribution using plotters.
/// Returns an RGBA pixel buffer suitable for egui texture loading.
fn render_3d_surface(
    returns_x: &[f64],
    returns_y: &[f64],
    label_x: &str,
    label_y: &str,
    pitch: f64,
    yaw: f64,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    use plotters::prelude::*;

    let min_len = returns_x.len().min(returns_y.len());
    if min_len < 10 {
        return None;
    }

    let rx = &returns_x[returns_x.len() - min_len..];
    let ry = &returns_y[returns_y.len() - min_len..];

    // Compute 2D KDE
    let grid_size = 35;
    let (x_grid, y_grid, density) =
        crate::analysis::randomness::kde_2d_grid(rx, ry, grid_size);

    if x_grid.is_empty() || y_grid.is_empty() {
        return None;
    }

    // Find max density for normalization
    let max_density = density
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(0.0_f64, f64::max);

    if max_density <= 0.0 {
        return None;
    }

    let x_min = x_grid[0];
    let x_max = *x_grid.last().unwrap();
    let y_min = y_grid[0];
    let y_max = *y_grid.last().unwrap();

    // Render to RGB buffer
    let mut rgb_buf = vec![0u8; (width * height * 3) as usize];

    {
        let root =
            BitMapBackend::with_buffer(&mut rgb_buf, (width, height)).into_drawing_area();
        root.fill(&RGBColor(24, 24, 32)).ok()?;

        let caption = format!("Joint Distribution: {} vs {}", label_x, label_y);

        let mut chart = ChartBuilder::on(&root)
            .caption(&caption, ("sans-serif", 18).into_font().color(&WHITE))
            .margin(8)
            .build_cartesian_3d(
                x_min..x_max,
                0.0..(max_density * 1.1),
                y_min..y_max,
            )
            .ok()?;

        chart.with_projection(|mut pb| {
            pb.pitch = pitch;
            pb.yaw = yaw;
            pb.scale = 0.85;
            pb.into_matrix()
        });

        chart
            .configure_axes()
            .label_style(("sans-serif", 11).into_font().color(&WHITE))
            .axis_panel_style(RGBColor(40, 40, 50).filled())
            .bold_grid_style(RGBColor(60, 60, 70))
            .light_grid_style(RGBColor(45, 45, 55))
            .x_labels(5)
            .y_labels(5)
            .z_labels(5)
            .draw()
            .ok()?;

        // Draw the surface with per-quad coloring using individual polygons
        for ix in 0..grid_size - 1 {
            for iy in 0..grid_size - 1 {
                let d00 = density[ix][iy];
                let d10 = density[ix + 1][iy];
                let d11 = density[ix + 1][iy + 1];
                let d01 = density[ix][iy + 1];
                let avg_d = (d00 + d10 + d11 + d01) / 4.0;
                let t = (avg_d / max_density).clamp(0.0, 1.0);

                let color = heat_color(t);

                let quad = vec![
                    (x_grid[ix], d00, y_grid[iy]),
                    (x_grid[ix + 1], d10, y_grid[iy]),
                    (x_grid[ix + 1], d11, y_grid[iy + 1]),
                    (x_grid[ix], d01, y_grid[iy + 1]),
                ];

                let _ = chart.draw_series(std::iter::once(
                    Polygon::new(quad, color.mix(0.85).filled()),
                ));
            }
        }

        // Add wireframe lines for depth perception
        for ix in (0..grid_size).step_by(3) {
            let line: Vec<(f64, f64, f64)> = (0..grid_size)
                .map(|iy| (x_grid[ix], density[ix][iy], y_grid[iy]))
                .collect();
            let _ = chart.draw_series(LineSeries::new(
                line,
                RGBColor(200, 200, 220).stroke_width(1),
            ));
        }
        for iy in (0..grid_size).step_by(3) {
            let line: Vec<(f64, f64, f64)> = (0..grid_size)
                .map(|ix| (x_grid[ix], density[ix][iy], y_grid[iy]))
                .collect();
            let _ = chart.draw_series(LineSeries::new(
                line,
                RGBColor(200, 200, 220).stroke_width(1),
            ));
        }

        root.present().ok()?;
    }

    // Convert RGB -> RGBA
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for pixel in rgb_buf.chunks_exact(3) {
        rgba.push(pixel[0]);
        rgba.push(pixel[1]);
        rgba.push(pixel[2]);
        rgba.push(255);
    }

    Some(rgba)
}

/// Heat colormap: blue -> cyan -> green -> yellow -> red
fn heat_color(t: f64) -> plotters::style::RGBColor {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    plotters::style::RGBColor(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

fn metric_card(ui: &mut egui::Ui, label: &str, value: &str) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.small(label);
                ui.strong(value);
            });
        });
}

fn vol_to_color(vol: f64) -> egui::Color32 {
    let pct = vol * 100.0;
    if pct > 30.0 {
        egui::Color32::from_rgb(220, 50, 50)
    } else if pct > 20.0 {
        egui::Color32::from_rgb(220, 150, 50)
    } else if pct > 10.0 {
        egui::Color32::from_rgb(200, 200, 50)
    } else {
        egui::Color32::from_rgb(50, 180, 50)
    }
}
