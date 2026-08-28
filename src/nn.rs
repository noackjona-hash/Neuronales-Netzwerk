//! Custom Deep Feedforward Neural Network implemented from scratch.
//! Supports arbitrary layer sizes, activation functions, Xavier/He initialization,
//! forward pass caching for real-time visualization, genetic mutation with Gaussian noise,
//! multi-mode crossover, and JSON serialization.

#![allow(dead_code)]

use crate::math::{GaussianRng, Matrix};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Activation functions for neural network layers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: max(alpha * x, x)
    LeakyReLU { alpha: f32 },
    /// Hyperbolic Tangent: (e^x - e^-x) / (e^x + e^-x) -> [-1, 1]
    Tanh,
    /// Logistic Sigmoid: 1 / (1 + e^-x) -> [0, 1]
    Sigmoid,
    /// Linear identity: f(x) = x
    Linear,
}

impl Activation {
    /// Apply scalar activation function manually from scratch.
    #[inline(always)]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    *alpha * x
                }
            }
            Activation::Tanh => {
                // Stable manual tanh computation
                if x > 20.0 {
                    1.0
                } else if x < -20.0 {
                    -1.0
                } else {
                    let exp_pos = x.exp();
                    let exp_neg = (-x).exp();
                    (exp_pos - exp_neg) / (exp_pos + exp_neg)
                }
            }
            Activation::Sigmoid => {
                if x > 20.0 {
                    1.0
                } else if x < -20.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-x).exp())
                }
            }
            Activation::Linear => x,
        }
    }

    /// In-place batch activation across a slice.
    pub fn apply_slice(&self, slice: &mut [f32]) {
        for val in slice.iter_mut() {
            *val = self.apply(*val);
        }
    }
}

/// A single dense fully-connected layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Matrix,
    pub biases: Vec<f32>,
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with Xavier/Glorot or He weight initialization.
    pub fn new<R: Rng + ?Sized>(
        in_features: usize,
        out_features: usize,
        activation: Activation,
        rng: &mut R,
    ) -> Self {
        // Initialization standard deviation based on activation function
        let std_dev = match activation {
            Activation::ReLU | Activation::LeakyReLU { .. } => {
                // He / Kaiming normal initialization: sqrt(2 / fan_in)
                (2.0 / in_features as f32).sqrt()
            }
            Activation::Tanh | Activation::Sigmoid | Activation::Linear => {
                // Xavier / Glorot normal initialization: sqrt(2 / (fan_in + fan_out))
                (2.0 / (in_features + out_features) as f32).sqrt()
            }
        };

        let weights = Matrix::from_fn(out_features, in_features, |_r, _c| {
            GaussianRng::sample(0.0, std_dev, rng)
        });

        // Initialize biases with small constant or zero
        let biases = vec![0.0f32; out_features];

        Self {
            in_features,
            out_features,
            weights,
            biases,
            activation,
        }
    }

    /// Forward pass: y = activation(W * x + b).
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_features];
        self.weights.forward_linear(input, &self.biases, &mut output);
        self.activation.apply_slice(&mut output);
        output
    }

    /// Forward pass into a mutable slice.
    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) {
        self.weights.forward_linear(input, &self.biases, output);
        self.activation.apply_slice(output);
    }

    /// Mutate layer weights and biases with Gaussian noise.
    pub fn mutate<R: Rng + ?Sized>(&mut self, rate: f32, strength: f32, rng: &mut R) {
        // Mutate weights
        for w in self.weights.data.iter_mut() {
            if rng.gen::<f32>() < rate {
                // 90% Gaussian perturbation, 10% completely new random weight
                if rng.gen::<f32>() < 0.90 {
                    *w += GaussianRng::sample(0.0, strength, rng);
                } else {
                    *w = GaussianRng::sample(0.0, 1.0, rng);
                }
                // Clamp to prevent explosive divergence
                *w = w.clamp(-5.0, 5.0);
            }
        }

        // Mutate biases
        for b in self.biases.iter_mut() {
            if rng.gen::<f32>() < rate {
                if rng.gen::<f32>() < 0.90 {
                    *b += GaussianRng::sample(0.0, strength, rng);
                } else {
                    *b = GaussianRng::sample(0.0, 1.0, rng);
                }
                *b = b.clamp(-5.0, 5.0);
            }
        }
    }

    /// Crossover two layers of identical topology.
    pub fn crossover<R: Rng + ?Sized>(
        parent_a: &Layer,
        parent_b: &Layer,
        rng: &mut R,
    ) -> Layer {
        assert_eq!(parent_a.in_features, parent_b.in_features);
        assert_eq!(parent_a.out_features, parent_b.out_features);

        let in_features = parent_a.in_features;
        let out_features = parent_a.out_features;
        let activation = parent_a.activation;

        // Uniform + blended crossover for weights
        let total_weights = in_features * out_features;
        let mut new_weights_data = Vec::with_capacity(total_weights);

        for i in 0..total_weights {
            let wa = parent_a.weights.data[i];
            let wb = parent_b.weights.data[i];

            let weight_choice: f32 = rng.gen();
            let w = if weight_choice < 0.45 {
                wa
            } else if weight_choice < 0.90 {
                wb
            } else {
                // Blend arithmetic crossover
                let alpha: f32 = rng.gen_range(-0.1..1.1);
                wa * alpha + wb * (1.0 - alpha)
            };
            new_weights_data.push(w);
        }

        // Crossover biases
        let mut new_biases = Vec::with_capacity(out_features);
        for i in 0..out_features {
            let ba = parent_a.biases[i];
            let bb = parent_b.biases[i];
            let b = if rng.gen::<bool>() { ba } else { bb };
            new_biases.push(b);
        }

        Layer {
            in_features,
            out_features,
            weights: Matrix {
                rows: out_features,
                cols: in_features,
                data: new_weights_data,
            },
            biases: new_biases,
            activation,
        }
    }
}

/// Multi-layer Deep Feedforward Neural Network.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    /// Construct a neural network with given topology and activations per layer.
    /// `layer_sizes`: e.g. [9, 16, 12, 2]
    /// `activations`: activations for each transition (len == layer_sizes.len() - 1)
    pub fn new<R: Rng + ?Sized>(
        layer_sizes: &[usize],
        activations: &[Activation],
        rng: &mut R,
    ) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "NeuralNetwork must have at least 2 layers (input and output)"
        );
        assert_eq!(
            layer_sizes.len() - 1,
            activations.len(),
            "Number of activations must equal number of layer transitions"
        );

        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activations[i],
                rng,
            ));
        }

        Self {
            layer_sizes: layer_sizes.to_vec(),
            layers,
        }
    }

    /// Create default car driving brain architecture:
    /// Inputs: 9 (7 ray sensors, forward velocity, steering/angular state)
    /// Hidden 1: 16 (ReLU)
    /// Hidden 2: 12 (Tanh)
    /// Outputs: 2 (Steering in [-1, 1] via Tanh, Throttle in [0, 1] via Sigmoid)
    pub fn default_car_brain<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::new(
            &[9, 16, 12, 2],
            &[
                Activation::ReLU,
                Activation::Tanh,
                Activation::Tanh, // Output layer will have custom post-mapping if needed
            ],
            rng,
        )
    }

    /// Forward pass through all layers.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.layer_sizes[0],
            "Input size ({}) must match network input layer ({})",
            input.len(),
            self.layer_sizes[0]
        );

        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Forward pass returning intermediate layer activations for visualizer HUD.
    /// Returns: (final_output, all_layer_activations including input)
    pub fn forward_with_cache(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());

        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
            activations.push(current.clone());
        }

        (current, activations)
    }

    /// Mutate all network layers in-place.
    pub fn mutate<R: Rng + ?Sized>(&mut self, rate: f32, strength: f32, rng: &mut R) {
        for layer in &mut self.layers {
            layer.mutate(rate, strength, rng);
        }
    }

    /// Genetic crossover of two parent neural networks.
    pub fn crossover<R: Rng + ?Sized>(
        parent_a: &NeuralNetwork,
        parent_b: &NeuralNetwork,
        rng: &mut R,
    ) -> NeuralNetwork {
        assert_eq!(parent_a.layer_sizes, parent_b.layer_sizes);
        let mut child_layers = Vec::with_capacity(parent_a.layers.len());

        for (la, lb) in parent_a.layers.iter().zip(parent_b.layers.iter()) {
            child_layers.push(Layer::crossover(la, lb, rng));
        }

        NeuralNetwork {
            layer_sizes: parent_a.layer_sizes.clone(),
            layers: child_layers,
        }
    }

    /// Serialize topology and weights to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize topology and weights from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Total trainable parameters (weights + biases).
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        for layer in &self.layers {
            count += layer.weights.data.len() + layer.biases.len();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_activations() {
        assert_eq!(Activation::ReLU.apply(-5.0), 0.0);
        assert_eq!(Activation::ReLU.apply(5.0), 5.0);
        assert_eq!(Activation::LeakyReLU { alpha: 0.1 }.apply(-10.0), -1.0);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-6);
        assert!((Activation::Tanh.apply(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_neural_network_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let net = NeuralNetwork::new(
            &[4, 8, 2],
            &[Activation::ReLU, Activation::Tanh],
            &mut rng,
        );

        let input = vec![0.5, -0.2, 1.0, 0.0];
        let output = net.forward(&input);

        assert_eq!(output.len(), 2);
        assert!(output[0] >= -1.0 && output[0] <= 1.0);
        assert!(output[1] >= -1.0 && output[1] <= 1.0);
    }

    #[test]
    fn test_json_serialization() {
        let mut rng = StdRng::seed_from_u64(123);
        let net = NeuralNetwork::default_car_brain(&mut rng);
        let json = net.to_json().expect("Serialization should succeed");

        let deserialized: NeuralNetwork =
            NeuralNetwork::from_json(&json).expect("Deserialization should succeed");

        assert_eq!(net.layer_sizes, deserialized.layer_sizes);
        assert_eq!(net.parameter_count(), deserialized.parameter_count());

        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let out1 = net.forward(&input);
        let out2 = deserialized.forward(&input);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
