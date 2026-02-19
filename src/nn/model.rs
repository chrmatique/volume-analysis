use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Lstm, LstmConfig},
    tensor::{backend::Backend, Tensor},
};

/// LSTM-based volatility prediction model
///
/// Architecture: LSTM -> take last hidden state -> Linear -> prediction
#[derive(Module, Debug)]
pub struct VolPredictionModel<B: Backend> {
    lstm: Lstm<B>,
    output_layer: Linear<B>,
}

/// Configuration for the model
#[derive(burn::config::Config)]
pub struct VolPredictionModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl VolPredictionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VolPredictionModel<B> {
        let lstm = LstmConfig::new(self.input_size, self.hidden_size, true)
            .init(device);

        let output_layer = LinearConfig::new(self.hidden_size, self.output_size)
            .init(device);

        VolPredictionModel {
            lstm,
            output_layer,
        }
    }
}

impl<B: Backend> VolPredictionModel<B> {
    /// Forward pass
    ///
    /// Input shape: [batch_size, seq_length, input_size]
    /// Output shape: [batch_size, output_size]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Run LSTM: output shape [batch_size, seq_length, hidden_size]
        let (lstm_output, _state) = self.lstm.forward(input, None);

        // Get the last time step: [batch_size, hidden_size]
        let dims = lstm_output.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let hidden_size = dims[2];
        let last_step = lstm_output.slice([0..batch_size, (seq_len - 1)..seq_len, 0..hidden_size]);
        let last_step = last_step.squeeze::<2>(1);

        // Linear projection: [batch_size, output_size]
        self.output_layer.forward(last_step)
    }
}

/// Number of input features per time step
/// 26 base + 22 randomness (entropy, hurst per sector) + 22 kurtosis (kurtosis, skew per sector)
pub const NUM_FEATURES: usize = 70;

/// Output size: 1 vol + 11 entropy + 22 (kurtosis, skew per sector)
pub const OUTPUT_SIZE: usize = 34;
