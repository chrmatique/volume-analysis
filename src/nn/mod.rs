pub mod dataset;
pub mod gpu;
pub mod model;
pub mod persistence;
pub mod training;

/// Type alias for the persisted model (NdArray backend, always available)
pub type LoadedModel = model::VolPredictionModel<burn::backend::NdArray>;
