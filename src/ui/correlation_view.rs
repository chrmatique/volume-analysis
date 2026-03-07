use eframe::egui;

use crate::app::AppState;

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Cross-Sector Correlation Matrix");
    ui.add_space(8.0);

    // --- Lookback window selector ---
    ui.horizontal(|ui| {
        ui.label("Lookback:");
        let prev = state.corr_lookback;
        for days in [30_usize, 60, 90] {
            ui.selectable_value(&mut state.corr_lookback, days, format!("{}d", days));
        }
        if state.corr_lookback != prev {
            state.recompute_correlation();
        }
    });
    ui.add_space(8.0);

    let corr = match &state.analysis.correlation {
        Some(c) if !c.symbols.is_empty() => c.clone(),
        _ => {
            ui.label("No correlation data available. Load market data first.");
            return;
        }
    };

    ui.label(format!(
        "Average cross-sector correlation: {:.3}",
        state.analysis.avg_cross_correlation
    ));
    ui.add_space(8.0);

    // --- PCA section ---
    if let Some(pca) = state.analysis.pca.clone() {
        render_pca_section(ui, &pca);
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);
    }

    // --- Correlation matrix grid ---
    let n = corr.symbols.len();
    let cell_size = 48.0;

    egui::ScrollArea::both().id_salt("corr_matrix_scroll").show(ui, |ui| {
        egui::Grid::new("corr_matrix")
            .min_col_width(cell_size)
            .max_col_width(cell_size)
            .spacing(egui::vec2(2.0, 2.0))
            .show(ui, |ui| {
                ui.label(""); // empty corner cell
                for sym in &corr.symbols {
                    ui.vertical_centered(|ui| {
                        ui.small(sym);
                    });
                }
                ui.end_row();

                for i in 0..n {
                    ui.small(&corr.symbols[i]);
                    for j in 0..n {
                        let val = corr.matrix[i][j];
                        let color = correlation_color(val);
                        let text_color = if val.abs() > 0.5 {
                            egui::Color32::WHITE
                        } else {
                            egui::Color32::BLACK
                        };

                        let (rect, _resp) = ui.allocate_exact_size(
                            egui::vec2(cell_size, 24.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().rect_filled(rect, 2.0, color);
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            format!("{:.2}", val),
                            egui::FontId::proportional(11.0),
                            text_color,
                        );
                    }
                    ui.end_row();
                }
            });
    });

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);

    ui.horizontal(|ui| {
        ui.label("Legend: ");
        color_swatch(ui, egui::Color32::from_rgb(220, 50, 50), "-1.0");
        color_swatch(ui, egui::Color32::from_rgb(240, 240, 240), " 0.0");
        color_swatch(ui, egui::Color32::from_rgb(50, 50, 220), "+1.0");
    });
}

fn render_pca_section(ui: &mut egui::Ui, pca: &crate::data::models::PcaResult) {
    let n_pcs = pca.eigenvalues.len();
    if n_pcs == 0 {
        return;
    }

    // Summary stats
    let signal_count = pca.eigenvalues.iter().filter(|&&e| e > pca.mp_upper).count();
    let total_variance: f64 = pca.eigenvalues.iter().sum();
    let signal_variance: f64 = pca
        .eigenvalues
        .iter()
        .filter(|&&e| e > pca.mp_upper)
        .sum();
    let absorption_ratio = if total_variance > 0.0 {
        signal_variance / total_variance * 100.0
    } else {
        0.0
    };

    ui.collapsing("PCA / Eigenvalue Analysis", |ui| {
        ui.horizontal(|ui| {
            ui.label(format!(
                "Signal PCs: {} / {}   |   Absorption ratio: {:.1}%   |   MP noise boundary: {:.3}",
                signal_count, n_pcs, absorption_ratio, pca.mp_upper
            ));
        });
        ui.add_space(6.0);

        // Show top PCs (cap at 6 for readability)
        let top_n = n_pcs.min(6);
        let n_sectors = pca.symbols.len();
        let label_col_width = 52.0_f32;
        let cell_w = 56.0_f32;
        let cell_h = 22.0_f32;

        egui::ScrollArea::horizontal().id_salt("pca_heatmap_scroll").show(ui, |ui| {
            egui::Grid::new("pca_loadings")
                .min_col_width(cell_w)
                .max_col_width(cell_w)
                .spacing(egui::vec2(2.0, 2.0))
                .show(ui, |ui| {
                    // Header row: PC labels with eigenvalue + MP indicator
                    let (corner_rect, _) =
                        ui.allocate_exact_size(egui::vec2(label_col_width, cell_h), egui::Sense::hover());
                    ui.painter().text(
                        corner_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Sector",
                        egui::FontId::proportional(10.0),
                        ui.visuals().text_color(),
                    );

                    for pc in 0..top_n {
                        let lambda = pca.eigenvalues[pc];
                        let is_signal = lambda > pca.mp_upper;
                        let label = if is_signal {
                            format!("PC{}\nλ={:.2}*", pc + 1, lambda)
                        } else {
                            format!("PC{}\nλ={:.2}", pc + 1, lambda)
                        };

                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(cell_w, cell_h * 2.0),
                            egui::Sense::hover(),
                        );

                        if is_signal {
                            // Signal PC: draw a highlight border
                            ui.painter().rect_stroke(
                                rect,
                                2.0,
                                egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 50)),
                            );
                        }

                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            label,
                            egui::FontId::proportional(9.5),
                            if is_signal {
                                egui::Color32::from_rgb(255, 200, 50)
                            } else {
                                ui.visuals().text_color()
                            },
                        );
                    }
                    ui.end_row();

                    // One row per sector
                    for s in 0..n_sectors {
                        // Sector label
                        let (lbl_rect, _) =
                            ui.allocate_exact_size(egui::vec2(label_col_width, cell_h), egui::Sense::hover());
                        ui.painter().text(
                            lbl_rect.left_center() + egui::vec2(2.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            &pca.symbols[s],
                            egui::FontId::proportional(10.0),
                            ui.visuals().text_color(),
                        );

                        // Loading cells
                        for pc in 0..top_n {
                            let loading = pca.eigenvectors[pc][s];
                            let color = correlation_color(loading);
                            let text_color = if loading.abs() > 0.5 {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::BLACK
                            };

                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(cell_w, cell_h),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, 2.0, color);
                            ui.painter().text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                format!("{:.2}", loading),
                                egui::FontId::proportional(10.0),
                                text_color,
                            );
                        }
                        ui.end_row();
                    }
                });
        });

        ui.add_space(4.0);
        ui.small("* = eigenvalue exceeds Marchenko-Pastur noise bound (statistically significant PC)");

        // Variance-explained bar
        ui.add_space(8.0);
        ui.label("Variance explained per PC:");
        if total_variance > 0.0 {
            ui.horizontal_wrapped(|ui| {
                for pc in 0..top_n {
                    let frac = pca.eigenvalues[pc] / total_variance * 100.0;
                    let is_signal = pca.eigenvalues[pc] > pca.mp_upper;
                    ui.label(egui::RichText::new(format!("PC{}: {:.1}%", pc + 1, frac))
                        .color(if is_signal {
                            egui::Color32::from_rgb(255, 200, 50)
                        } else {
                            ui.visuals().text_color()
                        }));
                }
            });
        }
    });
}

fn correlation_color(val: f64) -> egui::Color32 {
    let clamped = val.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        let t = clamped as f32;
        egui::Color32::from_rgb(
            (240.0 * (1.0 - t)) as u8,
            (240.0 * (1.0 - t)) as u8,
            (240.0 * (1.0 - t) + 220.0 * t) as u8,
        )
    } else {
        let t = (-clamped) as f32;
        egui::Color32::from_rgb(
            (240.0 * (1.0 - t) + 220.0 * t) as u8,
            (240.0 * (1.0 - t)) as u8,
            (240.0 * (1.0 - t)) as u8,
        )
    }
}

fn color_swatch(ui: &mut egui::Ui, color: egui::Color32, label: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 16.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.label(label);
}
