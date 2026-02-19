/// Shared chart utilities for all UI views that render plots.

use eframe::egui;

/// Inline height-adjustment drag control placed immediately above a chart.
/// Allows all drawn charts to be vertically resized via a shared implementation.
pub fn height_control(ui: &mut egui::Ui, height: &mut f32, label: &str) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgba_unmultiplied(80, 120, 200, 18))
        .inner_margin(egui::Margin::symmetric(8.0, 3.0))
        .rounding(egui::Rounding::same(4.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::from_rgb(100, 160, 255), "⇕");
                ui.colored_label(egui::Color32::from_gray(170), label);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(
                        egui::DragValue::new(height)
                            .speed(2.0)
                            .range(80.0..=800.0)
                            .suffix(" px"),
                    );
                    ui.colored_label(egui::Color32::from_gray(130), "drag to resize ·");
                });
            });
        });
    ui.add_space(2.0);
}
