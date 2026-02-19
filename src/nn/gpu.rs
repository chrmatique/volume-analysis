use std::process::Command;

use crate::data::models::GpuAdapterInfo;

/// GPU information collected via nvidia-smi or rocm-smi/amd-smi
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub utilization_percent: f32,
    pub temperature_c: f32,
}

/// Detect all WGPU-capable adapters (NVIDIA, AMD, Intel) via wgpu.
/// Returns an empty vec if wgpu fails to enumerate (e.g. no GPU drivers).
pub fn detect_wgpu_adapters() -> Vec<GpuAdapterInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let mut adapters = Vec::new();
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        let name = info.name.clone();
        let backend = format!("{:?}", info.backend);
        let is_nvidia = name.to_lowercase().contains("nvidia");
        let is_amd = name.to_lowercase().contains("amd") || name.to_lowercase().contains("radeon");

        adapters.push(GpuAdapterInfo {
            name: name.to_string(),
            backend,
            is_nvidia,
            is_amd,
        });
    }
    adapters
}

/// Detect an NVIDIA GPU by querying nvidia-smi.
/// Returns `Some(GpuInfo)` if an NVIDIA GPU is found, `None` otherwise.
pub fn detect_nvidia_gpu() -> Option<GpuInfo> {
    query_nvidia_smi()
}

/// Detect an AMD GPU by querying rocm-smi (Linux) or amd-smi (Windows).
/// Returns `Some(GpuInfo)` if AMD stats are available, `None` otherwise.
pub fn detect_amd_gpu() -> Option<GpuInfo> {
    #[cfg(target_os = "linux")]
    return query_rocm_smi();

    #[cfg(windows)]
    return query_amd_smi();

    #[cfg(not(any(target_os = "linux", windows)))]
    return None;
}

/// Poll live GPU stats (VRAM usage, utilization, temperature).
/// Prefers NVIDIA (nvidia-smi), then AMD (rocm-smi/amd-smi).
pub fn poll_gpu_stats() -> Option<GpuInfo> {
    detect_nvidia_gpu().or_else(detect_amd_gpu)
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

#[cfg(target_os = "linux")]
fn query_rocm_smi() -> Option<GpuInfo> {
    let output = Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--showuse", "--showtemp"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let name = "AMD GPU".to_string();
    let mut vram_total_mb = 0u64;
    let mut vram_used_mb = 0u64;
    let mut utilization_percent = 0.0f32;
    let mut temperature_c = 0.0f32;

    for line in stdout.lines() {
        let line = line.trim();
        if line.contains("GPU use") {
            if let Some(pct) = line.split('%').next().and_then(|s| s.split_whitespace().last()) {
                utilization_percent = pct.parse().unwrap_or(0.0);
            }
        } else if line.contains("Temperature") {
            if let Some(temp) = line.split("Temperature (Sensor").next().and_then(|s| {
                s.split_whitespace()
                    .find(|w| w.ends_with('C'))
                    .and_then(|w| w.trim_end_matches('C').parse::<f32>().ok())
            }) {
                temperature_c = temp;
            } else if let Some(t) = line.split(' ').find_map(|w| w.parse::<f32>().ok()) {
                temperature_c = t;
            }
        } else if line.contains("VRAM Total Memory") || line.contains("vram") {
            let mb = line
                .split_whitespace()
                .find_map(|w| w.parse::<u64>().ok())
                .unwrap_or(0);
            if vram_total_mb == 0 {
                vram_total_mb = mb;
            } else {
                vram_used_mb = mb;
            }
        }
    }

    if vram_total_mb == 0 && vram_used_mb == 0 {
        return None;
    }

    Some(GpuInfo {
        name,
        vram_total_mb: if vram_total_mb > 0 {
            vram_total_mb
        } else {
            vram_used_mb * 2
        },
        vram_used_mb,
        utilization_percent,
        temperature_c,
    })
}

#[cfg(target_os = "linux")]
fn query_amd_smi() -> Option<GpuInfo> {
    None::<GpuInfo>
}

#[cfg(windows)]
fn query_rocm_smi() -> Option<GpuInfo> {
    None::<GpuInfo>
}

#[cfg(windows)]
fn query_amd_smi() -> Option<GpuInfo> {
    let output = Command::new("amd-smi")
        .args(["metric"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let name = "AMD GPU".to_string();
    let mut vram_total_mb = 0u64;
    let mut vram_used_mb = 0u64;
    let mut utilization_percent = 0.0f32;
    let mut temperature_c = 0.0f32;

    for line in stdout.lines() {
        let line = line.trim().to_lowercase();
        if line.contains("memory") {
            if let Some(mb) = line
                .split_whitespace()
                .find_map(|w| w.replace(",", "").parse::<u64>().ok())
            {
                if vram_total_mb == 0 {
                    vram_total_mb = mb;
                } else {
                    vram_used_mb = mb;
                }
            }
        } else if line.contains("utilization") || line.contains("gpu use") {
            if let Some(pct) = line
                .split_whitespace()
                .find_map(|w| w.trim_end_matches('%').parse::<f32>().ok())
            {
                utilization_percent = pct;
            }
        } else if line.contains("temperature") || line.contains("temp") {
            if let Some(t) = line
                .split_whitespace()
                .find_map(|w| w.trim_end_matches('c').parse::<f32>().ok())
            {
                temperature_c = t;
            }
        }
    }

    if vram_total_mb == 0 && utilization_percent == 0.0 && temperature_c == 0.0 {
        return None;
    }

    Some(GpuInfo {
        name,
        vram_total_mb: if vram_total_mb > 0 {
            vram_total_mb
        } else {
            8192
        },
        vram_used_mb,
        utilization_percent,
        temperature_c,
    })
}

#[cfg(not(any(target_os = "linux", windows)))]
fn query_rocm_smi() -> Option<GpuInfo> {
    None::<GpuInfo>
}

#[cfg(not(any(target_os = "linux", windows)))]
fn query_amd_smi() -> Option<GpuInfo> {
    None::<GpuInfo>
}

/// Trait for GPU detection, enabling mock injection in tests.
#[allow(dead_code)]
pub trait GpuDetector {
    fn adapters(&self) -> Vec<GpuAdapterInfo>;
    fn nvidia_stats(&self) -> Option<GpuInfo>;
    fn amd_stats(&self) -> Option<GpuInfo>;
}

/// Real detector using wgpu and vendor-specific CLI tools.
#[allow(dead_code)]
pub struct RealGpuDetector;

impl GpuDetector for RealGpuDetector {
    fn adapters(&self) -> Vec<GpuAdapterInfo> {
        detect_wgpu_adapters()
    }

    fn nvidia_stats(&self) -> Option<GpuInfo> {
        detect_nvidia_gpu()
    }

    fn amd_stats(&self) -> Option<GpuInfo> {
        detect_amd_gpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock detector that returns a fake AMD GPU for testing without hardware.
    struct MockGpuDetector;

    impl GpuDetector for MockGpuDetector {
        fn adapters(&self) -> Vec<GpuAdapterInfo> {
            vec![GpuAdapterInfo {
                name: "AMD Radeon RX 6800 (Mock)".to_string(),
                backend: "Vulkan".to_string(),
                is_nvidia: false,
                is_amd: true,
            }]
        }

        fn nvidia_stats(&self) -> Option<GpuInfo> {
            None
        }

        fn amd_stats(&self) -> Option<GpuInfo> {
            Some(GpuInfo {
                name: "AMD Radeon RX 6800 (Mock)".to_string(),
                vram_total_mb: 16384,
                vram_used_mb: 2048,
                utilization_percent: 45.0,
                temperature_c: 62.0,
            })
        }
    }

    #[test]
    fn mock_amd_adapter_detected() {
        let mock = MockGpuDetector;
        let adapters = mock.adapters();
        assert!(!adapters.is_empty());
        assert!(adapters[0].is_amd);
        assert!(!adapters[0].is_nvidia);
        assert!(adapters[0].name.contains("AMD"));
    }

    #[test]
    fn mock_amd_training_device_available() {
        let mock = MockGpuDetector;
        let adapters = mock.adapters();
        assert!(adapters.iter().any(|a| a.is_amd));
        let amd = adapters.iter().find(|a| a.is_amd).unwrap();
        assert_eq!(amd.backend, "Vulkan");
    }
}
