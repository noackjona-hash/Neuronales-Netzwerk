//! Genetic Algorithm and Population Evolution engine built from scratch.
//! Features Adaptive Hypermutation, Stagnation Detection, Novelty Injection (Random Immigrants),
//! Elitism, and Multi-Mode Parent Selection to escape local minima and prevent premature convergence.

#![allow(dead_code)]

use crate::car::{Car, DeathReason};
use crate::math::Vec2;
use crate::nn::NeuralNetwork;
use crate::track::Track;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Selection strategy for parent choosing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionMethod {
    Tournament,
    RouletteWheel,
    RankBased,
}

/// Evolution Hyperparameters with dynamic adaptive mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub elitism_count: usize,       // Top N agents preserved untouched
    pub base_mutation_rate: f32,    // Base probability of mutating each weight
    pub base_mutation_strength: f32,// Base Gaussian standard deviation
    pub tournament_size: usize,     // Tournament candidates
    pub selection_method: SelectionMethod,
    pub max_generation_time: f32,   // Maximum seconds per generation
    pub novelty_ratio: f32,         // Ratio of population reserved for fresh random explorers (e.g. 0.15)
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 70,
            elitism_count: 5,
            base_mutation_rate: 0.09,
            base_mutation_strength: 0.24,
            tournament_size: 4,
            selection_method: SelectionMethod::Tournament,
            max_generation_time: 45.0,
            novelty_ratio: 0.15,
        }
    }
}

/// An individual agent combining a physical vehicle and a deep neural network brain.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub car: Car,
    pub brain: NeuralNetwork,
    pub fitness: f32,
    pub is_elite: bool,
    pub is_novelty_immigrant: bool,
    pub color_rgba: [u8; 4],
}

impl Agent {
    pub fn new(id: usize, position: Vec2, heading_angle: f32, brain: NeuralNetwork) -> Self {
        let hue = (id as f32 * 137.5) % 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.85, 0.95);

        Self {
            id,
            car: Car::new(position, heading_angle),
            brain,
            fitness: 0.0,
            is_elite: false,
            is_novelty_immigrant: false,
            color_rgba: [r, g, b, 230],
        }
    }

    /// Step agent forward: read 12 telemetry inputs -> compute deep brain -> apply car controls -> integrate physics.
    pub fn step(&mut self, dt: f32, track: &Track) {
        if !self.car.is_alive {
            return;
        }

        let inputs = self.car.get_network_inputs();
        let outputs = self.brain.forward(&inputs);

        // Output 0: Steering in [-1.0, 1.0]
        let steer = outputs[0].clamp(-1.0, 1.0);

        // Output 1: Throttle & Brake
        let gas_brake = outputs[1];
        let (throttle, brake) = if gas_brake >= 0.0 {
            (0.15 + 0.85 * gas_brake.clamp(0.0, 1.0), 0.0)
        } else {
            (0.0, (-gas_brake).clamp(0.0, 1.0))
        };

        self.car.apply_controls(steer, throttle, brake);
        self.car.update(dt, track);
        self.fitness = self.car.fitness;
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Generation summary metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub max_checkpoints: usize,
    pub max_laps: usize,
    pub top_speed: f32,
    pub death_wall_count: usize,
    pub death_timeout_count: usize,
    pub death_wrongway_count: usize,
    pub stagnant_generations: usize,
    pub mutation_temperature: f32,
}

/// The evolutionary population manager with anti-stagnation mechanisms.
pub struct Population {
    pub config: EvolutionConfig,
    pub agents: Vec<Agent>,
    pub generation: usize,
    pub generation_time: f32,
    pub rng: StdRng,

    // Statistics & Records
    pub best_ever_fitness: f32,
    pub best_ever_brain: Option<NeuralNetwork>,
    pub best_current_agent_idx: usize,
    pub stats_history: Vec<GenerationStats>,

    // Stagnation & Adaptive Mutation Engine
    pub stagnant_generations: usize,
    pub mutation_temperature: f32, // Multiplier for mutation strength (1.0 = normal, 3.0+ = heat burst)
}

impl Population {
    pub fn new(config: EvolutionConfig, track: &Track, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agents = Vec::with_capacity(config.population_size);

        for i in 0..config.population_size {
            let brain = NeuralNetwork::default_car_brain(&mut rng);
            agents.push(Agent::new(i, track.start_position, track.start_angle, brain));
        }

        Self {
            config,
            agents,
            generation: 1,
            generation_time: 0.0,
            rng,
            best_ever_fitness: 0.0,
            best_ever_brain: None,
            best_current_agent_idx: 0,
            stats_history: Vec::new(),
            stagnant_generations: 0,
            mutation_temperature: 1.0,
        }
    }

    pub fn alive_count(&self) -> usize {
        self.agents.iter().filter(|a| a.car.is_alive).count()
    }

    pub fn current_best_fitness(&self) -> f32 {
        self.agents
            .iter()
            .map(|a| a.fitness)
            .fold(0.0f32, |acc, f| acc.max(f))
    }

    pub fn leader_idx(&self) -> usize {
        let mut best_idx = 0;
        let mut best_fit = -1.0f32;

        for (i, agent) in self.agents.iter().enumerate() {
            let effective_fit = agent.fitness + if agent.car.is_alive { 100000.0 } else { 0.0 };
            if effective_fit > best_fit {
                best_fit = effective_fit;
                best_idx = i;
            }
        }
        best_idx
    }

    pub fn step(&mut self, dt: f32, track: &Track) {
        self.generation_time += dt;

        for agent in &mut self.agents {
            if agent.car.is_alive {
                agent.step(dt, track);
            }
        }

        self.best_current_agent_idx = self.leader_idx();
    }

    pub fn is_generation_over(&self) -> bool {
        self.alive_count() == 0 || self.generation_time >= self.config.max_generation_time
    }

    /// Advance to the next generation with Adaptive Hypermutation & Novelty Injection.
    pub fn advance_generation(&mut self, track: &Track) {
        // 1. Sort agents descending by fitness
        self.agents
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let gen_best_fitness = self.agents[0].fitness;
        let sum_fitness: f32 = self.agents.iter().map(|a| a.fitness).sum();
        let avg_fitness = sum_fitness / self.agents.len().max(1) as f32;

        let max_checkpoints = self.agents.iter().map(|a| a.car.checkpoints_hit).max().unwrap_or(0);
        let max_laps = self.agents.iter().map(|a| a.car.laps_completed).max().unwrap_or(0);
        let top_speed = self
            .agents
            .iter()
            .map(|a| a.car.top_speed_recorded)
            .fold(0.0f32, |acc, s| acc.max(s));

        // 2. Check for Improvement or Stagnation
        let is_new_record = gen_best_fitness > self.best_ever_fitness + 50.0;
        if is_new_record {
            self.best_ever_fitness = gen_best_fitness;
            self.best_ever_brain = Some(self.agents[0].brain.clone());
            self.stagnant_generations = 0;
            self.mutation_temperature = 1.0; // Cool down to fine-tune
        } else {
            self.stagnant_generations += 1;
            // Adaptive Temperature Ramp: exponentially scale mutation if stuck
            let heat_factor = 1.0 + (self.stagnant_generations as f32 * 0.12).min(4.0);
            self.mutation_temperature = heat_factor;
        }

        let mut death_wall = 0;
        let mut death_timeout = 0;
        let mut death_wrongway = 0;

        for a in &self.agents {
            match a.car.death_reason {
                DeathReason::WallCollision => death_wall += 1,
                DeathReason::IdleTimeout => death_timeout += 1,
                DeathReason::WrongWay => death_wrongway += 1,
                DeathReason::Alive => {}
            }
        }

        // Record history
        self.stats_history.push(GenerationStats {
            generation: self.generation,
            best_fitness: gen_best_fitness,
            avg_fitness,
            max_checkpoints,
            max_laps,
            top_speed,
            death_wall_count: death_wall,
            death_timeout_count: death_timeout,
            death_wrongway_count: death_wrongway,
            stagnant_generations: self.stagnant_generations,
            mutation_temperature: self.mutation_temperature,
        });

        // 3. Compute Adaptive Mutation Hyperparameters
        let effective_mutation_rate = (self.config.base_mutation_rate * (1.0 + self.stagnant_generations as f32 * 0.04))
            .clamp(0.08, 0.40);
        let effective_mutation_strength = (self.config.base_mutation_strength * self.mutation_temperature)
            .clamp(0.20, 1.20);

        // 4. Construct Next Generation
        let mut new_agents = Vec::with_capacity(self.config.population_size);

        // A. Elitism: Keep Top N Champions intact
        let num_elites = self.config.elitism_count.min(self.agents.len());
        for i in 0..num_elites {
            let mut elite_agent = Agent::new(
                i,
                track.start_position,
                track.start_angle,
                self.agents[i].brain.clone(),
            );
            elite_agent.is_elite = true;
            elite_agent.color_rgba = [255, 215, 0, 255];
            new_agents.push(elite_agent);
        }

        // B. Champion Exploratory Variants (5% of population)
        if let Some(champ) = &self.best_ever_brain {
            let champ_variants = ((self.config.population_size as f32) * 0.08).round() as usize;
            for _ in 0..champ_variants {
                if new_agents.len() >= self.config.population_size { break; }
                let mut variant = champ.clone();
                // Mutate with higher strength
                variant.mutate(effective_mutation_rate * 1.5, effective_mutation_strength * 1.3, &mut self.rng);
                let id = new_agents.len();
                let mut agent = Agent::new(id, track.start_position, track.start_angle, variant);
                agent.color_rgba = [0, 255, 200, 240]; // Cyan aura
                new_agents.push(agent);
            }
        }

        // C. Novelty Random Immigrants (15% of population) to inject fresh genetic lineages
        let num_novelty = ((self.config.population_size as f32) * self.config.novelty_ratio).round() as usize;
        for _ in 0..num_novelty {
            if new_agents.len() >= self.config.population_size { break; }
            let fresh_brain = NeuralNetwork::default_car_brain(&mut self.rng);
            let id = new_agents.len();
            let mut agent = Agent::new(id, track.start_position, track.start_angle, fresh_brain);
            agent.is_novelty_immigrant = true;
            agent.color_rgba = [255, 105, 180, 240]; // Pink/purple aura
            new_agents.push(agent);
        }

        // D. Main Population: Tournament Selection + Crossover + Adaptive Mutation
        while new_agents.len() < self.config.population_size {
            let idx_a = select_parent_idx(
                &self.agents,
                self.config.selection_method,
                self.config.tournament_size,
                &mut self.rng,
            );
            let idx_b = select_parent_idx(
                &self.agents,
                self.config.selection_method,
                self.config.tournament_size,
                &mut self.rng,
            );

            let mut child_brain =
                NeuralNetwork::crossover(&self.agents[idx_a].brain, &self.agents[idx_b].brain, &mut self.rng);

            child_brain.mutate(
                effective_mutation_rate,
                effective_mutation_strength,
                &mut self.rng,
            );

            let id = new_agents.len();
            new_agents.push(Agent::new(
                id,
                track.start_position,
                track.start_angle,
                child_brain,
            ));
        }

        // 5. Install new generation
        self.agents = new_agents;
        self.generation += 1;
        self.generation_time = 0.0;
        self.best_current_agent_idx = 0;
    }

    pub fn reset_to_track(&mut self, track: &Track) {
        self.generation = 1;
        self.generation_time = 0.0;
        self.stagnant_generations = 0;
        self.mutation_temperature = 1.0;
        self.stats_history.clear();

        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.car.reset(track.start_position, track.start_angle);
            agent.fitness = 0.0;
            agent.is_elite = false;
            agent.is_novelty_immigrant = false;
            let hue = (i as f32 * 137.5) % 360.0;
            let (r, g, b) = hsv_to_rgb(hue, 0.85, 0.95);
            agent.color_rgba = [r, g, b, 230];
        }
    }

    pub fn inject_champion_brain(&mut self, brain: NeuralNetwork, track: &Track) {
        self.best_ever_brain = Some(brain.clone());
        self.best_ever_fitness = 0.0;
        self.stagnant_generations = 0;
        self.mutation_temperature = 1.0;

        self.agents[0] = Agent::new(0, track.start_position, track.start_angle, brain.clone());
        self.agents[0].is_elite = true;
        self.agents[0].color_rgba = [255, 215, 0, 255];

        for i in 1..self.agents.len() {
            let mut mutated = brain.clone();
            mutated.mutate(self.config.base_mutation_rate, self.config.base_mutation_strength, &mut self.rng);
            self.agents[i] = Agent::new(i, track.start_position, track.start_angle, mutated);
        }

        self.generation_time = 0.0;
    }
}

fn select_parent_idx(
    agents: &[Agent],
    method: SelectionMethod,
    tournament_size: usize,
    rng: &mut StdRng,
) -> usize {
    match method {
        SelectionMethod::Tournament => {
            let k = tournament_size.max(2).min(agents.len());
            let mut best_idx = rng.gen_range(0..agents.len());
            let mut best_fitness = agents[best_idx].fitness;

            for _ in 1..k {
                let candidate_idx = rng.gen_range(0..agents.len());
                if agents[candidate_idx].fitness > best_fitness {
                    best_fitness = agents[candidate_idx].fitness;
                    best_idx = candidate_idx;
                }
            }
            best_idx
        }
        SelectionMethod::RouletteWheel => {
            let total_fitness: f32 = agents.iter().map(|a| a.fitness.max(1.0)).sum();
            let target = rng.gen_range(0.0..total_fitness);
            let mut running = 0.0f32;

            for (i, agent) in agents.iter().enumerate() {
                running += agent.fitness.max(1.0);
                if running >= target {
                    return i;
                }
            }
            0
        }
        SelectionMethod::RankBased => {
            let n = agents.len();
            let total_rank_sum = (n * (n + 1)) / 2;
            let target = rng.gen_range(0..total_rank_sum);

            let mut running = 0;
            for (i, _) in agents.iter().enumerate() {
                let rank_weight = n - i;
                running += rank_weight;
                if running >= target {
                    return i;
                }
            }
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_advance() {
        let track = Track::preset_grand_prix();
        let config = EvolutionConfig {
            population_size: 20,
            elitism_count: 2,
            ..Default::default()
        };
        let mut pop = Population::new(config, &track, 42);
        assert_eq!(pop.agents.len(), 20);
        assert_eq!(pop.generation, 1);

        for _ in 0..10 {
            pop.step(0.016, &track);
        }

        pop.advance_generation(&track);
        assert_eq!(pop.generation, 2);
        assert_eq!(pop.agents.len(), 20);
        assert_eq!(pop.stats_history.len(), 1);
    }
}
