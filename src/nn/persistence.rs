use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkGzFileRecorder},
};
use serde::{Deserialize, Serialize};

use crate::config;
use crate::data::cache;
use crate::nn::model::{VolPredictionModel, VolPredictionModelConfig, NUM_FEATURES, OUTPUT_SIZE};

/// Metadata saved alongside the trained model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub trained_at: String,
    pub final_loss: f64,
    pub epochs: usize,
}

const MODEL_FILENAME: &str = "vol_model";
const METADATA_FILENAME: &str = "vol_model_metadata";

/// Save the trained model to disk in gzip-compressed MessagePack format.
pub fn save_model<B: burn::tensor::backend::Backend>(
    model: &crate::nn::model::VolPredictionModel<B>,
    final_loss: f64,
) -> Result<(), String> {
    let cache_dir = cache::cache_dir().map_err(|e| e.to_string())?;
    let model_path = cache_dir.join(MODEL_FILENAME);

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();

    model
        .clone()
        .save_file(&model_path, &recorder)
        .map_err(|e| format!("Failed to save model: {}", e))?;

    let metadata = ModelMetadata {
        trained_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        final_loss,
        epochs: config::NN_EPOCHS,
    };

    cache::save_json(METADATA_FILENAME, &metadata).map_err(|e| format!("Failed to save metadata: {}", e))?;

    tracing::info!(
        "Model saved to {} (compressed, final loss: {:.6})",
        model_path.with_extension("mpk.gz").display(),
        final_loss
    );
    Ok(())
}

/// Load the trained model from disk. Returns (model, metadata) or None if not found/invalid.
pub fn load_model() -> Option<(VolPredictionModel<NdArray>, ModelMetadata)> {
    let cache_dir = cache::cache_dir().ok()?;
    let model_path = cache_dir.join(MODEL_FILENAME);

    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::default();
    let device = <NdArray as burn::tensor::backend::Backend>::Device::default();

    let model_config = VolPredictionModelConfig {
        input_size: NUM_FEATURES,
        hidden_size: config::NN_HIDDEN_SIZE,
        output_size: OUTPUT_SIZE,
    };

    let model = model_config
        .init::<NdArray>(&device)
        .load_file(&model_path, &recorder, &device)
        .map_err(|e| {
            tracing::debug!("Model load failed: {}", e);
            e
        })
        .ok()?;

    let metadata: ModelMetadata = cache::load_json(METADATA_FILENAME).ok()?;

    tracing::info!(
        "Model loaded from disk (trained {}, loss: {:.6})",
        metadata.trained_at,
        metadata.final_loss
    );

    Some((model, metadata))
}
