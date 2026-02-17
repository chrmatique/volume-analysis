mod app;
mod config;
mod data;
mod analysis;
mod nn;
mod ui;

use app::MktNoiseApp;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Volume Analysis",
        options,
        Box::new(|_cc| Ok(Box::new(MktNoiseApp::default()))),
    )
}
