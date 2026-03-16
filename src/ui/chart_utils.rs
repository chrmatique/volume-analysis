/// Shared chart utilities for all UI views that render plots.

use eframe::egui;
use eframe::egui::Vec2b;
use egui_plot::{CoordinatesFormatter, Corner, Plot, PlotBounds, PlotPoint, PlotUi};

// ── Hover label utilities ───────────────────────────────────────────────────

/// A named data series for hover display. Borrows the underlying data so no
/// heap allocation is needed beyond what the view already holds.
pub struct HoverSeries<'a> {
    pub name: &'a str,
    pub data: &'a [[f64; 2]],
    pub decimals: usize,
    pub suffix: &'a str,
}

/// Build a `CoordinatesFormatter` that shows the nearest Y value for each
/// series at the cursor's X position.  Use with
/// `Plot::coordinates_formatter(Corner::RightBottom, hover_formatter(&series))`.
pub fn hover_formatter<'a>(series: &'a [HoverSeries<'a>]) -> CoordinatesFormatter<'a> {
    CoordinatesFormatter::new(move |cursor: &PlotPoint, _bounds: &PlotBounds| {
        let x = cursor.x;
        let mut text = format!("x: {:.0}", x);
        for s in series {
            if let Some(idx) = nearest_x_index(s.data, x) {
                use std::fmt::Write;
                let _ = write!(
                    text,
                    "\n{}: {:.prec$}{}",
                    s.name,
                    s.data[idx][1],
                    s.suffix,
                    prec = s.decimals
                );
            }
        }
        text
    })
}

/// Variant of [`hover_formatter`] for charts with discrete, labelled X
/// positions (e.g. yield-curve maturity names).  `x_labels[i]` is shown
/// instead of the numeric X value when the cursor is nearest to index `i`.
pub fn hover_formatter_labeled_x<'a>(
    series: &'a [HoverSeries<'a>],
    x_labels: &'a [String],
) -> CoordinatesFormatter<'a> {
    CoordinatesFormatter::new(move |cursor: &PlotPoint, _bounds: &PlotBounds| {
        let x = cursor.x;
        let x_idx = x.round().max(0.0) as usize;
        let x_display = x_labels
            .get(x_idx)
            .map(|s| s.as_str())
            .unwrap_or("?");
        let mut text = x_display.to_string();
        for s in series {
            if let Some(idx) = nearest_x_index(s.data, x) {
                use std::fmt::Write;
                let _ = write!(
                    text,
                    "\n{}: {:.prec$}{}",
                    s.name,
                    s.data[idx][1],
                    s.suffix,
                    prec = s.decimals
                );
            }
        }
        text
    })
}

/// Pass to `Plot::label_formatter` to suppress the default per-line hover
/// tooltip (we show data in the corner instead).
pub fn no_hover_label(_name: &str, _point: &PlotPoint) -> String {
    String::new()
}

/// The fixed corner where hover labels are displayed.
pub const HOVER_CORNER: Corner = Corner::RightBottom;

/// Binary-search for the index of the data point whose X is closest to
/// `target_x`.  Assumes `data` is sorted ascending by `[0]` (X).
fn nearest_x_index(data: &[[f64; 2]], target_x: f64) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let idx = data.partition_point(|p| p[0] < target_x);
    if idx == 0 {
        return Some(0);
    }
    if idx >= data.len() {
        return Some(data.len() - 1);
    }
    let left_dist = (data[idx - 1][0] - target_x).abs();
    let right_dist = (data[idx][0] - target_x).abs();
    if left_dist <= right_dist {
        Some(idx - 1)
    } else {
        Some(idx)
    }
}

// ── Y-axis drag-to-zoom ─────────────────────────────────────────────────────

/// Per-chart state cached across frames for the Y-axis drag sensor.
#[derive(Default, Clone)]
struct YAxisDragState {
    /// Inner plot frame rect from the previous frame, used to position the
    /// drag sensor on the current frame before `Plot::show()` is called.
    plot_frame: Option<egui::Rect>,
}

/// How many pixels of drag translate to ×1 zoom change (larger = faster).
const Y_DRAG_SENSITIVITY: f32 = 0.005;

/// Width of the invisible hit area placed to the left of the plot frame.
const Y_AXIS_HIT_WIDTH: f32 = 50.0;

/// Drop-in replacement for `Plot::show()` that adds click-and-drag Y-axis
/// scaling. Drag **up** on the Y-axis label area to zoom in; drag **down**
/// to zoom out.
///
/// `id_source` should be the same string passed to `Plot::new(…)` so that the
/// per-chart state is uniquely keyed. The `plot` argument should already have
/// all modifiers applied (e.g. via [`default_plot_interaction`]).
pub fn plot_with_y_drag<S: std::hash::Hash>(
    ui: &mut egui::Ui,
    id_source: S,
    plot: Plot<'_>,
    build_fn: impl FnOnce(&mut PlotUi),
) {
    let state_id = egui::Id::new(("y_drag_state", id_source));

    // Read state cached from the previous frame.
    let state: YAxisDragState = ui
        .data(|d| d.get_temp::<YAxisDragState>(state_id))
        .unwrap_or_default();

    // Place an invisible drag-sense widget over the Y-axis label area (using
    // the previous frame's rect). By formally claiming the drag through egui's
    // interaction system, the parent ScrollArea won't also scroll when the user
    // is adjusting the Y-axis.
    let y_delta_px: f32 = if let Some(frame) = state.plot_frame {
        let sense_rect = egui::Rect::from_min_size(
            egui::pos2(frame.left() - Y_AXIS_HIT_WIDTH, frame.top()),
            egui::vec2(Y_AXIS_HIT_WIDTH, frame.height()),
        );
        let resp = ui.interact(sense_rect, state_id.with("y_sense"), egui::Sense::drag());
        if resp.dragged() {
            resp.drag_delta().y
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Run the plot, applying Y-zoom inside the closure when a drag is active.
    let response = plot.show(ui, |plot_ui| {
        if y_delta_px.abs() > 0.001 {
            let bounds = plot_ui.plot_bounds();
            let y_min = bounds.min()[1];
            let y_max = bounds.max()[1];
            let center = (y_min + y_max) * 0.5;
            let half_range = (y_max - y_min) * 0.5;

            // Drag up (negative delta) → zoom_factor > 1 → shrink range.
            // Drag down (positive delta) → zoom_factor < 1 → expand range.
            let zoom_factor = ((1.0 - y_delta_px * Y_DRAG_SENSITIVITY) as f64).max(0.1);
            let new_half = half_range / zoom_factor;

            let new_bounds = PlotBounds::from_min_max(
                [bounds.min()[0], center - new_half],
                [bounds.max()[0], center + new_half],
            );
            plot_ui.set_plot_bounds(new_bounds);
        }
        build_fn(plot_ui);
    });

    // Cache the inner frame rect for the next frame.
    let new_state = YAxisDragState {
        plot_frame: Some(*response.transform.frame()),
    };
    ui.data_mut(|d| d.insert_temp(state_id, new_state));
}

/// Drop-in replacement for `Plot::show()` that adds both click-and-drag
/// Y-axis scaling **and** mousewheel-driven X-axis zoom anchored at the
/// pointer position.
///
/// Scroll **up** (positive `raw_scroll_delta.y`) zooms in (fewer candles);
/// scroll **down** zooms out (more candles). Y-axis drag behaves identically
/// to [`plot_with_y_drag`].
pub fn plot_with_xy_zoom<S: std::hash::Hash + Copy>(
    ui: &mut egui::Ui,
    id_source: S,
    plot: Plot<'_>,
    build_fn: impl FnOnce(&mut PlotUi),
) {
    let state_id = egui::Id::new(("y_drag_state", id_source));
    let state: YAxisDragState = ui
        .data(|d| d.get_temp::<YAxisDragState>(state_id))
        .unwrap_or_default();

    let x_scroll_delta: f32 = if let Some(frame) = state.plot_frame {
        if ui.rect_contains_pointer(frame) {
            let delta = ui.input(|i| i.raw_scroll_delta.y);
            if delta.abs() > 0.001 {
                // Consume the scroll event so the parent ScrollArea doesn't
                // also scroll the app window while the user is X-zooming.
                ui.input_mut(|i| {
                    i.raw_scroll_delta = egui::Vec2::ZERO;
                    i.smooth_scroll_delta = egui::Vec2::ZERO;
                });
            }
            delta
        } else {
            0.0
        }
    } else {
        0.0
    };

    plot_with_y_drag(ui, id_source, plot, |plot_ui| {
        if x_scroll_delta.abs() > 0.001 {
            let bounds = plot_ui.plot_bounds();
            let pointer_x = plot_ui
                .pointer_coordinate()
                .map(|p| p.x)
                .unwrap_or((bounds.min()[0] + bounds.max()[0]) * 0.5);

            let zoom_factor =
                (1.0 + x_scroll_delta as f64 * 0.002).clamp(0.8, 1.25);
            let new_min_x = pointer_x - (pointer_x - bounds.min()[0]) / zoom_factor;
            let new_max_x = pointer_x + (bounds.max()[0] - pointer_x) / zoom_factor;

            plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                [new_min_x, bounds.min()[1]],
                [new_max_x, bounds.max()[1]],
            ));
        }
        build_fn(plot_ui);
    });
}

// ── Plot interaction presets ─────────────────────────────────────────────────

/// Apply the standard Y-axis-only interaction settings to a `Plot`.
/// - Drag: Y-axis only (vertical panning)
/// - Scroll/zoom: Y-axis only (vertical stretch/compress)
/// - X-axis is locked on both axes; double-click resets to auto-bounds.
pub fn default_plot_interaction(plot: Plot<'_>) -> Plot<'_> {
    plot.allow_drag(Vec2b::new(false, true))
        .allow_scroll(false)
        .allow_zoom(Vec2b::new(false, true))
}

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
