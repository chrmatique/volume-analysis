use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};
use sysinfo::System;

use crate::config;
use crate::data::models::{ComputeStats, MarketData, NnPredictions, TrainingStatus};
use crate::nn::dataset::{build_dataset, VolBatcher};
use crate::nn::model::{VolPredictionModelConfig, NUM_FEATURES, OUTPUT_SIZE};

/// GPU training backend: Wgpu with autodiff
pub type GpuBackend = Autodiff<Wgpu>;

/// CPU training backend: NdArray with autodiff
pub type CpuBackend = Autodiff<NdArray>;

/// Shared state for communicating training progress to the UI
#[derive(Clone)]
pub struct TrainingProgress {
    pub status: Arc<Mutex<TrainingStatus>>,
    pub losses: Arc<Mutex<Vec<f64>>>,
    pub predictions: Arc<Mutex<NnPredictions>>,
    pub pause_flag: Arc<AtomicBool>,
    pub compute_stats: Arc<Mutex<ComputeStats>>,
}

impl TrainingProgress {
    pub fn new() -> Self {
        Self {
            status: Arc::new(Mutex::new(TrainingStatus::Idle)),
            losses: Arc::new(Mutex::new(Vec::new())),
            predictions: Arc::new(Mutex::new(NnPredictions::default())),
            pause_flag: Arc::new(AtomicBool::new(false)),
            compute_stats: Arc::new(Mutex::new(ComputeStats::default())),
        }
    }

    pub fn request_pause(&self) {
        self.pause_flag.store(true, Ordering::SeqCst);
    }

    pub fn request_resume(&self) {
        self.pause_flag.store(false, Ordering::SeqCst);
    }

    pub fn is_paused(&self) -> bool {
        self.pause_flag.load(Ordering::SeqCst)
    }
}

/// Run the full training pipeline, selecting GPU or CPU backend.
pub fn train(market_data: &MarketData, progress: &TrainingProgress, use_gpu: bool) {
    // Prefer vendor-specific stats (NVIDIA via nvidia-smi, AMD via rocm-smi/amd-smi)
    let gpu_stats = crate::nn::gpu::poll_gpu_stats();
    let adapter_name = crate::nn::gpu::detect_wgpu_adapters()
        .into_iter()
        .next()
        .map(|a| a.name);

    // Populate initial GPU detection info from stats or adapter name
    if let Ok(mut stats) = progress.compute_stats.lock() {
        if use_gpu {
            stats.gpu_detected = true;
            stats.gpu_name = gpu_stats
                .as_ref()
                .map(|i| i.name.clone())
                .or(adapter_name.clone());
            if let Some(ref info) = gpu_stats {
                stats.gpu_vram_total_mb = Some(info.vram_total_mb);
                stats.gpu_vram_used_mb = Some(info.vram_used_mb);
                stats.gpu_utilization_percent = Some(info.utilization_percent);
                stats.gpu_temperature_c = Some(info.temperature_c);
            }
        }
    }

    if use_gpu {
        let backend_label = format!(
            "WGPU GPU: {}",
            gpu_stats
                .as_ref()
                .map(|i| i.name.as_str())
                .or(adapter_name.as_deref())
                .unwrap_or("Default")
        );
        if let Ok(mut stats) = progress.compute_stats.lock() {
            stats.backend_name = backend_label;
            stats.using_gpu = true;
        }

        tracing::info!("Starting GPU training with Wgpu backend");
        let device = <Wgpu as burn::tensor::backend::Backend>::Device::default();
        train_impl::<GpuBackend>(device, market_data, progress);
    } else {
        if let Ok(mut stats) = progress.compute_stats.lock() {
            stats.backend_name = "NdArray (CPU) + Autodiff".to_string();
            stats.using_gpu = false;
        }

        tracing::info!("Starting CPU training with NdArray backend");
        let device = <NdArray as burn::tensor::backend::Backend>::Device::default();
        train_impl::<CpuBackend>(device, market_data, progress);
    }
}

/// Generic training implementation that works with any autodiff backend.
fn train_impl<B: AutodiffBackend>(
    device: B::Device,
    market_data: &MarketData,
    progress: &TrainingProgress,
) {
    // System info for compute stats
    let mut sys = System::new_all();
    sys.refresh_all();
    let total_memory_mb = sys.total_memory() / (1024 * 1024);

    // Update status
    set_status(progress, TrainingStatus::Training {
        epoch: 0,
        total_epochs: config::NN_EPOCHS,
        loss: f64::NAN,
    });

    // Build dataset
    let dataset = build_dataset(market_data, config::NN_LOOKBACK_DAYS, config::NN_FORWARD_DAYS);

    if dataset.samples.is_empty() {
        set_status(progress, TrainingStatus::Error(
            "Not enough data to build training dataset. Load more market data.".into(),
        ));
        return;
    }

    let total = dataset.samples.len();
    let train_size = (total as f64 * 0.8) as usize;

    if train_size < config::NN_BATCH_SIZE || total - train_size < 1 {
        set_status(progress, TrainingStatus::Error(
            format!("Dataset too small ({} samples). Need more data.", total),
        ));
        return;
    }

    // Split chronologically
    let train_samples = dataset.samples[..train_size].to_vec();
    let _val_samples = dataset.samples[train_size..].to_vec();

    let train_dataset = crate::nn::dataset::VolDataset { samples: train_samples };

    let batcher = VolBatcher::<B>::new(device.clone());

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config::NN_BATCH_SIZE)
        .shuffle(42)
        .build(train_dataset);

    // Initialize model
    let model_config = VolPredictionModelConfig {
        input_size: NUM_FEATURES,
        hidden_size: config::NN_HIDDEN_SIZE,
        output_size: OUTPUT_SIZE,
    };
    let mut model = model_config.init::<B>(&device);

    let param_count = model.num_params();

    // Update initial compute stats
    update_compute_stats(progress, &mut sys, total_memory_mb, 0, 0.0, param_count);

    // Optimizer
    let mut optim = AdamConfig::new().init();

    // Training loop
    let mut best_loss = f64::INFINITY;
    for epoch in 0..config::NN_EPOCHS {
        // Pause check: spin-wait while paused
        while progress.is_paused() {
            if let Ok(status) = progress.status.lock() {
                if matches!(*status, TrainingStatus::Training { .. }) {
                    drop(status);
                    let current_loss = progress.losses.lock()
                        .ok()
                        .and_then(|l| l.last().copied())
                        .unwrap_or(f64::NAN);
                    set_status(progress, TrainingStatus::Paused {
                        epoch,
                        total_epochs: config::NN_EPOCHS,
                        loss: current_loss,
                    });
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // When resuming from pause, set status back to Training
        set_status(progress, TrainingStatus::Training {
            epoch,
            total_epochs: config::NN_EPOCHS,
            loss: progress.losses.lock()
                .ok()
                .and_then(|l| l.last().copied())
                .unwrap_or(f64::NAN),
        });

        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut samples_this_epoch = 0_usize;

        for batch in dataloader.iter() {
            // Check pause mid-epoch too; set Paused so UI updates
            while progress.is_paused() {
                set_status(progress, TrainingStatus::Paused {
                    epoch,
                    total_epochs: config::NN_EPOCHS,
                    loss: progress.losses.lock()
                        .ok()
                        .and_then(|l| l.last().copied())
                        .unwrap_or(f64::NAN),
                });
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            let batch_size = batch.inputs.dims()[0];
            let output = model.forward(batch.inputs);
            let loss = mse_loss(output, batch.targets);

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            let loss_scalar = loss_val.first().copied().unwrap_or(f32::NAN) as f64;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config::NN_LEARNING_RATE, model, grads);

            epoch_loss += loss_scalar;
            batch_count += 1;
            samples_this_epoch += batch_size;
        }

        let epoch_duration = epoch_start.elapsed();
        let epoch_ms = epoch_duration.as_millis() as u64;

        let avg_loss = if batch_count > 0 {
            epoch_loss / batch_count as f64
        } else {
            f64::NAN
        };

        let samples_per_sec = if epoch_ms > 0 {
            samples_this_epoch as f64 / (epoch_ms as f64 / 1000.0)
        } else {
            0.0
        };

        if avg_loss < best_loss {
            best_loss = avg_loss;
        }

        // Update progress
        if let Ok(mut losses) = progress.losses.lock() {
            losses.push(avg_loss);
        }
        set_status(progress, TrainingStatus::Training {
            epoch: epoch + 1,
            total_epochs: config::NN_EPOCHS,
            loss: avg_loss,
        });

        // Update compute stats (including live GPU stats via nvidia-smi)
        update_compute_stats(progress, &mut sys, total_memory_mb, epoch_ms, samples_per_sec, param_count);
        update_gpu_live_stats(progress);
    }

    // Generate predictions using the trained model in inference mode
    let valid_model = model.valid();
    let inference_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
    generate_predictions::<B::InnerBackend>(&valid_model, market_data, &inference_device, progress);

    // Save model to disk BEFORE setting Complete status so the UI's load_model()
    // call is guaranteed to find the file on the very first repaint after Complete.
    if let Err(e) = crate::nn::persistence::save_model(&valid_model, best_loss) {
        tracing::warn!("Failed to save trained model: {}", e);
    }

    set_status(progress, TrainingStatus::Complete { final_loss: best_loss });
}

/// Update CPU/memory compute stats
fn update_compute_stats(
    progress: &TrainingProgress,
    sys: &mut System,
    total_memory_mb: u64,
    epoch_ms: u64,
    samples_per_sec: f64,
    param_count: usize,
) {
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    let cpu_usage = sys.global_cpu_usage();
    let used_memory_mb = sys.used_memory() / (1024 * 1024);

    if let Ok(mut stats) = progress.compute_stats.lock() {
        stats.cpu_usage_percent = cpu_usage;
        stats.memory_used_mb = used_memory_mb;
        stats.memory_total_mb = total_memory_mb;
        stats.epoch_duration_ms = epoch_ms;
        stats.samples_per_sec = samples_per_sec;
        stats.total_params = param_count;
    }
}

/// Poll nvidia-smi for live GPU utilization, VRAM, and temperature
fn update_gpu_live_stats(progress: &TrainingProgress) {
    if let Some(info) = crate::nn::gpu::poll_gpu_stats() {
        if let Ok(mut stats) = progress.compute_stats.lock() {
            stats.gpu_vram_used_mb = Some(info.vram_used_mb);
            stats.gpu_utilization_percent = Some(info.utilization_percent);
            stats.gpu_temperature_c = Some(info.temperature_c);
        }
    }
}

/// Mean squared error loss
fn mse_loss<B: AutodiffBackend>(
    predictions: burn::tensor::Tensor<B, 2>,
    targets: burn::tensor::Tensor<B, 2>,
) -> burn::tensor::Tensor<B, 1> {
    let diff = predictions - targets;
    let sq = diff.clone() * diff;
    sq.mean().unsqueeze()
}

/// Run inference with a trained model and return predictions for each sector.
/// Public for use when loading a saved model from disk.
pub fn run_inference(
    model: &crate::nn::model::VolPredictionModel<burn::backend::NdArray>,
    market_data: &MarketData,
) -> NnPredictions {
    let device = <burn::backend::NdArray as burn::tensor::backend::Backend>::Device::default();
    run_inference_impl(model, market_data, &device)
}

fn run_inference_impl<B: burn::tensor::backend::Backend>(
    model: &crate::nn::model::VolPredictionModel<B>,
    market_data: &MarketData,
    device: &B::Device,
) -> NnPredictions {
    let dataset = build_dataset(market_data, config::NN_LOOKBACK_DAYS, config::NN_FORWARD_DAYS);

    if let Some(last_sample) = dataset.samples.last() {
        let seq_len = last_sample.features.len();
        let num_features = last_sample.features.first().map(|f| f.len()).unwrap_or(0);

        let mut input_data: Vec<f32> = Vec::with_capacity(seq_len * num_features);
        for step in &last_sample.features {
            for &f in step {
                input_data.push(f as f32);
            }
        }

        let input = burn::tensor::Tensor::<B, 1>::from_floats(input_data.as_slice(), device)
            .reshape([1_usize, seq_len, num_features]);

        let pred = model.forward(input);
        let pred_val = pred.into_data().to_vec::<f32>().unwrap_or_default();

        let predicted_vol = pred_val.get(0).copied().unwrap_or(0.0) as f64;
        let vol: Vec<(String, f64)> = market_data
            .sectors
            .iter()
            .map(|s| (s.symbol.clone(), predicted_vol))
            .collect();

        let randomness: Vec<(String, f64)> = market_data
            .sectors
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let entropy = pred_val.get(1 + i).copied().unwrap_or(0.0) as f64;
                (s.symbol.clone(), entropy)
            })
            .collect();

        let mut kurtosis = Vec::with_capacity(market_data.sectors.len());
        for (i, s) in market_data.sectors.iter().enumerate() {
            let k = pred_val.get(12 + i * 2).copied().unwrap_or(0.0) as f64;
            let sk = pred_val.get(12 + i * 2 + 1).copied().unwrap_or(0.0) as f64;
            kurtosis.push((s.symbol.clone(), k, sk));
        }

        return NnPredictions {
            vol,
            randomness,
            kurtosis,
        };
    }

    NnPredictions::default()
}

/// Generate predictions for each sector using the trained model
fn generate_predictions<B: burn::tensor::backend::Backend>(
    model: &crate::nn::model::VolPredictionModel<B>,
    market_data: &MarketData,
    device: &B::Device,
    progress: &TrainingProgress,
) {
    let predictions = run_inference_impl(model, market_data, device);
    if let Ok(mut preds) = progress.predictions.lock() {
        *preds = predictions;
    }
}

fn set_status(progress: &TrainingProgress, status: TrainingStatus) {
    if let Ok(mut s) = progress.status.lock() {
        *s = status;
    }
}
