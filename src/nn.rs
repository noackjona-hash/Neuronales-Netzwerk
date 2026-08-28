//! Custom Deep Feedforward Neural Network implemented from scratch.
//! Supports arbitrary hidden layer counts, dynamic layer sizes, activation functions,
//! Xavier/He initialization, forward pass caching for real-time visualization,
//! genetic mutation with Gaussian noise, multi-mode crossover, and JSON serialization.

#![allow(dead_code)]

use crate::math::{GaussianRng, Matrix};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Activation functions for neural network layers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    LeakyReLU { alpha: f32 },
    Tanh,
    Sigmoid,
    Linear,
}

impl Activation {
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
    pub fn new<R: Rng + ?Sized>(
        in_features: usize,
        out_features: usize,
        activation: Activation,
        rng: &mut R,
    ) -> Self {
        let std_dev = match activation {
            Activation::ReLU | Activation::LeakyReLU { .. } => {
                (2.0 / in_features as f32).sqrt()
            }
            Activation::Tanh | Activation::Sigmoid | Activation::Linear => {
                (2.0 / (in_features + out_features) as f32).sqrt()
            }
        };

        let weights = Matrix::from_fn(out_features, in_features, |_r, _c| {
            GaussianRng::sample(0.0, std_dev, rng)
        });

        let biases = vec![0.0f32; out_features];

        Self {
            in_features,
            out_features,
            weights,
            biases,
            activation,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_features];
        self.weights.forward_linear(input, &self.biases, &mut output);
        self.activation.apply_slice(&mut output);
        output
    }

    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) {
        self.weights.forward_linear(input, &self.biases, output);
        self.activation.apply_slice(output);
    }

    pub fn mutate<R: Rng + ?Sized>(&mut self, rate: f32, strength: f32, rng: &mut R) {
        for w in self.weights.data.iter_mut() {
            if rng.gen::<f32>() < rate {
                if rng.gen::<f32>() < 0.88 {
                    *w += GaussianRng::sample(0.0, strength, rng);
                } else {
                    *w = GaussianRng::sample(0.0, 1.2, rng);
                }
                *w = w.clamp(-6.0, 6.0);
            }
        }

        for b in self.biases.iter_mut() {
            if rng.gen::<f32>() < rate {
                if rng.gen::<f32>() < 0.88 {
                    *b += GaussianRng::sample(0.0, strength, rng);
                } else {
                    *b = GaussianRng::sample(0.0, 1.2, rng);
                }
                *b = b.clamp(-6.0, 6.0);
            }
        }
    }

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
                let alpha: f32 = rng.gen_range(-0.15..1.15);
                wa * alpha + wb * (1.0 - alpha)
            };
            new_weights_data.push(w);
        }

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

/// Multi-layer Deep Feedforward Neural Network with arbitrary depth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
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

    /// Deep car driving brain architecture with 4 hidden layers:
    /// Topology: [14 -> 32 -> 24 -> 16 -> 12 -> 2]
    pub fn default_car_brain<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::new(
            &[14, 32, 24, 16, 12, 2],
            &[
                Activation::LeakyReLU { alpha: 0.05 }, // Hidden 1: 32 (Spatial perception & speed scaling)
                Activation::ReLU,                      // Hidden 2: 24 (Corner geometry & upcoming curvature)
                Activation::Tanh,                      // Hidden 3: 16 (Grip boundaries & drift control)
                Activation::Tanh,                      // Hidden 4: 12 (Fine motor steering & trail braking)
                Activation::Tanh,                      // Output: 2 (Steering & Throttle/Brake)
            ],
            rng,
        )
    }

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

    pub fn mutate<R: Rng + ?Sized>(&mut self, rate: f32, strength: f32, rng: &mut R) {
        for layer in &mut self.layers {
            layer.mutate(rate, strength, rng);
        }
    }

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

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

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
    fn test_deep_neural_network_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let net = NeuralNetwork::default_car_brain(&mut rng);

        assert_eq!(net.layer_sizes, vec![14, 32, 24, 16, 12, 2]);
        assert_eq!(net.layers.len(), 5);

        let input = vec![0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.5, 0.2, 0.75, 0.0, 0.0, 0.1, -0.2];
        let (output, activations) = net.forward_with_cache(&input);

        assert_eq!(output.len(), 2);
        assert_eq!(activations.len(), 6);
        assert!(output[0] >= -1.0 && output[0] <= 1.0);
        assert!(output[1] >= -1.0 && output[1] <= 1.0);
    }
}
