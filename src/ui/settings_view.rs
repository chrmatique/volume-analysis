use std::sync::{Arc, Mutex};

use eframe::egui;

use crate::app::AppState;
use crate::data::models::{ScreenshotCompression, ScreenshotFileType};

pub fn render(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Settings");
    ui.add_space(8.0);

    let mut prev_visible = false;

    // API Keys section (topmost)
    render_api_keys_section(ui, state, &mut prev_visible);

    // Screenshot settings section
    render_screenshot_section(ui, state, &mut prev_visible);

    // NN Training Settings section
    render_nn_training_section(ui, state, &mut prev_visible);
}

fn render_api_keys_section(
    ui: &mut egui::Ui,
    state: &mut AppState,
    prev_visible: &mut bool,
) {
    if *prev_visible {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);
    }

    ui.heading("API Keys");
    ui.add_space(4.0);

    ui.group(|ui| {
        // Per-widget reveal toggle stored in egui's memory so it persists across frames
        let reveal_id = ui.id().with("fmp_key_reveal");
        let mut reveal: bool = ui.memory(|m| m.data.get_temp(reveal_id).unwrap_or(false));

        egui::Grid::new("api_keys_grid")
            .num_columns(2)
            .spacing(egui::vec2(12.0, 6.0))
            .show(ui, |ui| {
                ui.label("FMP API Key:");
                ui.horizontal(|ui| {
                    let text_edit = egui::TextEdit::singleline(&mut state.fmp_api_key)
                        .desired_width(260.0)
                        .password(!reveal);
                    ui.add(text_edit);

                    let eye_label = if reveal { "🙈 Hide" } else { "👁 Show" };
                    if ui.button(eye_label).clicked() {
                        reveal = !reveal;
                    }
                });
                ui.end_row();
            });

        ui.memory_mut(|m| m.data.insert_temp(reveal_id, reveal));

        ui.add_space(8.0);

        if ui.button("Save").clicked() {
            // Update the process environment so the next fetch picks it up
            unsafe { std::env::set_var("FMP_API_KEY", &state.fmp_api_key); }

            // Persist to .env on disk
            match crate::config::save_env_file("FMP_API_KEY", &state.fmp_api_key) {
                Ok(_) => {
                    // Clear cached FMP files so the next refresh fetches fresh data
                    let _ = std::fs::remove_file("fmp_treasury_rates.json");
                    let _ = std::fs::remove_file("fmp_sector_performance.json");
                    state.status_message =
                        "FMP API key saved. Click 'Refresh Data' to fetch with the new key."
                            .to_string();
                }
                Err(e) => {
                    state.status_message = format!("Failed to save FMP API key: {e}");
                }
            }
        }

        ui.add_space(4.0);
        ui.label("Key is used for Treasury rates and Sector performance data. Click 'Refresh Data' after saving to fetch with the new key.");
    });

    *prev_visible = true;
}

fn render_screenshot_section(
    ui: &mut egui::Ui,
    state: &mut AppState,
    prev_visible: &mut bool,
) {
    if *prev_visible {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);
    }

    ui.heading("Screenshot");
    ui.add_space(4.0);

    ui.group(|ui| {
        egui::Grid::new("screenshot_settings_grid")
            .num_columns(2)
            .spacing(egui::vec2(12.0, 6.0))
            .show(ui, |ui| {
                // Save path — native folder browser
                ui.label("Save Path:");
                ui.horizontal(|ui| {
                    // Show the current path as greyed-out, non-editable text
                    ui.add_enabled(
                        false,
                        egui::TextEdit::singleline(&mut state.screenshot_settings.save_path.clone())
                            .desired_width(220.0),
                    );

                    let picking = state.folder_picker_result.is_some();
                    let btn = ui.add_enabled(!picking, egui::Button::new("Browse…"));
                    if btn.clicked() {
                        let slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
                        state.folder_picker_result = Some(slot.clone());
                        let initial = state.screenshot_settings.save_path.clone();
                        std::thread::spawn(move || {
                            let chosen = open_folder_dialog(&initial);
                            if let Ok(mut guard) = slot.lock() {
                                *guard = chosen;
                            }
                        });
                    }
                    if picking {
                        ui.spinner();
                    }
                });
                ui.end_row();

                // File type
                ui.label("File Type:");
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut state.screenshot_settings.file_type,
                        ScreenshotFileType::Png,
                        "PNG",
                    );
                    ui.selectable_value(
                        &mut state.screenshot_settings.file_type,
                        ScreenshotFileType::Jpeg,
                        "JPEG",
                    );
                    ui.selectable_value(
                        &mut state.screenshot_settings.file_type,
                        ScreenshotFileType::Tiff,
                        "TIFF",
                    )
                    .on_hover_text("TIFF compression level is informational only");
                });
                ui.end_row();

                // Compression
                ui.label("Compression:");
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut state.screenshot_settings.compression,
                        ScreenshotCompression::None,
                        "None",
                    );
                    ui.selectable_value(
                        &mut state.screenshot_settings.compression,
                        ScreenshotCompression::Low,
                        "Low",
                    );
                    ui.selectable_value(
                        &mut state.screenshot_settings.compression,
                        ScreenshotCompression::High,
                        "High",
                    );
                });
                ui.end_row();
            });

        ui.add_space(8.0);

        if ui.button("Save Settings").clicked() {
            match crate::data::cache::save_json(
                "screenshot_settings.json",
                &state.screenshot_settings,
            ) {
                Ok(_) => state.status_message = "Screenshot settings saved.".to_string(),
                Err(_) => state.status_message = "Failed to save screenshot settings.".to_string(),
            }
        }

        ui.label("Use the 📷 camera button in the tab bar to capture a screenshot.");
    });

    *prev_visible = true;
}

/// Open a native OS folder-selection dialog and return the chosen path.
///
/// On Windows, uses PowerShell's `FolderBrowserDialog`. On other platforms,
/// falls back to a plain `zenity` GTK call. Returns `None` if the user cancels.
fn open_folder_dialog(initial_path: &str) -> Option<String> {
    #[cfg(windows)]
    {
        // PowerShell one-liner: create a WinForms FolderBrowserDialog, show it,
        // and print the selected path to stdout.
        let script = format!(
            r#"
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = 'Select screenshot save folder'
$d.SelectedPath = '{}'
$d.ShowNewFolderButton = $true
if ($d.ShowDialog() -eq 'OK') {{ Write-Output $d.SelectedPath }}
"#,
            initial_path.replace('\'', "''")
        );
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
            .ok()?;
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
        None
    }

    #[cfg(not(windows))]
    {
        let output = std::process::Command::new("zenity")
            .args([
                "--file-selection",
                "--directory",
                "--title=Select screenshot save folder",
                &format!("--filename={}/", initial_path),
            ])
            .output()
            .ok()?;
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
        None
    }
}

fn render_nn_training_section(
    ui: &mut egui::Ui,
    state: &mut AppState,
    prev_visible: &mut bool,
) {
    // Add divider before this section if previous section was visible
    if *prev_visible {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);
    }

    ui.heading("Neural Network Training");
    ui.add_space(4.0);

    ui.group(|ui| {
        ui.label("Select input features to use during model training:");
        ui.add_space(4.0);

        // Sector Volatility checkbox
        let mut vol_enabled = state.nn_feature_flags.sector_volatility;
        ui.checkbox(&mut vol_enabled, "Sector Volatility (11 features)");
        if vol_enabled != state.nn_feature_flags.sector_volatility {
            state.nn_feature_flags.sector_volatility = vol_enabled;
        }

        // Market Randomness checkbox
        let mut rand_enabled = state.nn_feature_flags.market_randomness;
        ui.checkbox(&mut rand_enabled, "Market Randomness (22 features)");
        if rand_enabled != state.nn_feature_flags.market_randomness {
            state.nn_feature_flags.market_randomness = rand_enabled;
        }

        // Kurtosis checkbox
        let mut kurt_enabled = state.nn_feature_flags.kurtosis;
        ui.checkbox(&mut kurt_enabled, "Kurtosis (22 features)");
        if kurt_enabled != state.nn_feature_flags.kurtosis {
            state.nn_feature_flags.kurtosis = kurt_enabled;
        }

        ui.add_space(8.0);

        if ui.button("Save Settings").clicked() {
            if let Ok(_) = crate::data::cache::save_json("nn_feature_flags.json", &state.nn_feature_flags) {
                state.status_message = "Settings saved successfully.".to_string();
            } else {
                state.status_message = "Failed to save settings.".to_string();
            }
        }

        ui.label("Settings are applied when you start a new training session.");
    });

    *prev_visible = true;
}
