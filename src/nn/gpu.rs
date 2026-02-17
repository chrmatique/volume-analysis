use std::process::Command;

/// GPU information collected via nvidia-smi
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub utilization_percent: f32,
    pub temperature_c: f32,
}

/// Detect an NVIDIA GPU by querying nvidia-smi.
/// Returns `Some(GpuInfo)` if an NVIDIA GPU is found, `None` otherwise.
pub fn detect_nvidia_gpu() -> Option<GpuInfo> {
    query_nvidia_smi()
}

/// Poll live GPU stats (VRAM usage, utilization, temperature).
/// Intended to be called periodically during training.
pub fn poll_gpu_stats() -> Option<GpuInfo> {
    query_nvidia_smi()
}

fn query_nvidia_smi() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Take the first GPU line if multiple GPUs are present
    let line = stdout.lines().next()?.trim().to_string();
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() < 5 {
        return None;
    }

    Some(GpuInfo {
        name: parts[0].to_string(),
        vram_total_mb: parts[1].parse().unwrap_or(0),
        vram_used_mb: parts[2].parse().unwrap_or(0),
        utilization_percent: parts[3].parse().unwrap_or(0.0),
        temperature_c: parts[4].parse().unwrap_or(0.0),
    })
}
