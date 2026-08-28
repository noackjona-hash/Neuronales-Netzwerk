//! 2D Top-down arcade/semi-realistic car physics, raycast sensor array,
//! collision detection, checkpoint progression, and fitness evaluation.

#![allow(dead_code)]

use crate::math::{LineSegment, Ray2, RaycastHit, Vec2};
use crate::track::Track;
use std::f32::consts::PI;

/// Reason for car death / simulation termination for this agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeathReason {
    Alive,
    WallCollision,
    IdleTimeout,
    WrongWay,
}

/// Car configuration constants.
#[derive(Debug, Clone, Copy)]
pub struct CarConfig {
    pub length: f32,
    pub width: f32,
    pub max_forward_speed: f32,
    pub max_reverse_speed: f32,
    pub acceleration_rate: f32,
    pub braking_rate: f32,
    pub steer_speed: f32,
    pub max_steer_angle: f32,
    pub drag_coeff: f32,
    pub lateral_grip: f32,
    pub ray_sensor_range: f32,
    pub ray_sensor_angles: [f32; 7],
    pub checkpoint_timeout: f32, // Seconds before timeout if no new checkpoint is reached
}

impl Default for CarConfig {
    fn default() -> Self {
        Self {
            length: 32.0,
            width: 16.0,
            max_forward_speed: 620.0,
            max_reverse_speed: 120.0,
            acceleration_rate: 450.0,
            braking_rate: 650.0,
            steer_speed: 3.8,
            max_steer_angle: 0.55, // ~31.5 degrees
            drag_coeff: 0.0008,
            lateral_grip: 0.88, // 0.88 lateral velocity damping per step (allows controlled drift)
            ray_sensor_range: 280.0,
            ray_sensor_angles: [
                -75.0 * PI / 180.0,
                -45.0 * PI / 180.0,
                -20.0 * PI / 180.0,
                0.0,
                20.0 * PI / 180.0,
                45.0 * PI / 180.0,
                75.0 * PI / 180.0,
            ],
            checkpoint_timeout: 3.5,
        }
    }
}

/// 2D Car instance.
#[derive(Debug, Clone)]
pub struct Car {
    pub config: CarConfig,
    pub position: Vec2,
    pub velocity: Vec2,
    pub heading_angle: f32, // Heading in radians
    pub angular_velocity: f32,
    pub steer_input: f32,    // [-1.0, 1.0]
    pub throttle_input: f32, // [0.0, 1.0]
    pub brake_input: f32,    // [0.0, 1.0]

    // Sensors
    pub sensor_readings: [f32; 7], // Normalized [0.0, 1.0]
    pub sensor_hits: [Option<RaycastHit>; 7],

    // Progress & Fitness
    pub is_alive: bool,
    pub death_reason: DeathReason,
    pub current_checkpoint_idx: usize,
    pub checkpoints_hit: usize,
    pub laps_completed: usize,
    pub distance_traveled: f32,
    pub time_alive: f32,
    pub time_since_last_checkpoint: f32,
    pub fitness: f32,
    pub top_speed_recorded: f32,
}

impl Car {
    /// Create a new car placed at track start position and heading.
    pub fn new(position: Vec2, heading_angle: f32) -> Self {
        Self::with_config(position, heading_angle, CarConfig::default())
    }

    /// Create car with custom configuration.
    pub fn with_config(position: Vec2, heading_angle: f32, config: CarConfig) -> Self {
        Self {
            config,
            position,
            velocity: Vec2::ZERO,
            heading_angle,
            angular_velocity: 0.0,
            steer_input: 0.0,
            throttle_input: 0.0,
            brake_input: 0.0,
            sensor_readings: [1.0; 7],
            sensor_hits: [None; 7],
            is_alive: true,
            death_reason: DeathReason::Alive,
            current_checkpoint_idx: 0,
            checkpoints_hit: 0,
            laps_completed: 0,
            distance_traveled: 0.0,
            time_alive: 0.0,
            time_since_last_checkpoint: 0.0,
            fitness: 0.0,
            top_speed_recorded: 0.0,
        }
    }

    /// Reset car state for a new generation or track restart.
    pub fn reset(&mut self, position: Vec2, heading_angle: f32) {
        self.position = position;
        self.velocity = Vec2::ZERO;
        self.heading_angle = heading_angle;
        self.angular_velocity = 0.0;
        self.steer_input = 0.0;
        self.throttle_input = 0.0;
        self.brake_input = 0.0;
        self.sensor_readings = [1.0; 7];
        self.sensor_hits = [None; 7];
        self.is_alive = true;
        self.death_reason = DeathReason::Alive;
        self.current_checkpoint_idx = 0;
        self.checkpoints_hit = 0;
        self.laps_completed = 0;
        self.distance_traveled = 0.0;
        self.time_alive = 0.0;
        self.time_since_last_checkpoint = 0.0;
        self.fitness = 0.0;
        self.top_speed_recorded = 0.0;
    }

    /// Forward unit vector of the car.
    #[inline(always)]
    pub fn forward_vector(&self) -> Vec2 {
        Vec2::from_angle(self.heading_angle)
    }

    /// Right unit vector (lateral) of the car.
    #[inline(always)]
    pub fn right_vector(&self) -> Vec2 {
        self.forward_vector().perpendicular()
    }

    /// Forward speed (positive forward, negative reverse).
    #[inline(always)]
    pub fn forward_speed(&self) -> f32 {
        self.velocity.dot(self.forward_vector())
    }

    /// Lateral slip speed.
    #[inline(always)]
    pub fn lateral_speed(&self) -> f32 {
        self.velocity.dot(self.right_vector())
    }

    /// Oriented bounding box corners in world coordinates [Front-Left, Front-Right, Rear-Right, Rear-Left].
    pub fn bounding_box_corners(&self) -> [Vec2; 4] {
        let fwd = self.forward_vector() * (self.config.length * 0.5);
        let right = self.right_vector() * (self.config.width * 0.5);

        [
            self.position + fwd - right, // Front-Left
            self.position + fwd + right, // Front-Right
            self.position - fwd + right, // Rear-Right
            self.position - fwd - right, // Rear-Left
        ]
    }

    /// Get 4 bounding box edge segments for collision detection.
    pub fn bounding_box_edges(&self) -> [LineSegment; 4] {
        let pts = self.bounding_box_corners();
        [
            LineSegment::new(pts[0], pts[1]), // Front bumper
            LineSegment::new(pts[1], pts[2]), // Right side
            LineSegment::new(pts[2], pts[3]), // Rear bumper
            LineSegment::new(pts[3], pts[0]), // Left side
        ]
    }

    /// Update raycast sensors against track walls.
    pub fn update_sensors(&mut self, track: &Track) {
        let fwd_angle = self.heading_angle;
        let sensor_origin = self.position + self.forward_vector() * (self.config.length * 0.4);

        for (i, &rel_angle) in self.config.ray_sensor_angles.iter().enumerate() {
            let ray_angle = fwd_angle + rel_angle;
            let ray_dir = Vec2::from_angle(ray_angle);
            let ray = Ray2::new(sensor_origin, ray_dir, self.config.ray_sensor_range);

            let hit = ray.cast_segments(&track.wall_segments);
            match hit {
                Some(h) => {
                    self.sensor_readings[i] = h.fraction;
                    self.sensor_hits[i] = Some(h);
                }
                None => {
                    self.sensor_readings[i] = 1.0;
                    self.sensor_hits[i] = None;
                }
            }
        }
    }

    /// Build neural network input vector (9 inputs):
    /// [Ray0..Ray6 (7 distances), Normalized Speed, Steering Angle / Angular velocity].
    pub fn get_network_inputs(&self) -> [f32; 9] {
        let norm_speed = (self.forward_speed() / self.config.max_forward_speed).clamp(-0.2, 1.2);
        let norm_steer = self.steer_input.clamp(-1.0, 1.0);

        [
            self.sensor_readings[0],
            self.sensor_readings[1],
            self.sensor_readings[2],
            self.sensor_readings[3],
            self.sensor_readings[4],
            self.sensor_readings[5],
            self.sensor_readings[6],
            norm_speed,
            norm_steer,
        ]
    }

    /// Apply control inputs: steer ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1].
    pub fn apply_controls(&mut self, steer: f32, throttle: f32, brake: f32) {
        self.steer_input = steer.clamp(-1.0, 1.0);
        self.throttle_input = throttle.clamp(0.0, 1.0);
        self.brake_input = brake.clamp(0.0, 1.0);
    }

    /// Physics integration step.
    pub fn physics_step(&mut self, dt: f32) {
        if !self.is_alive {
            return;
        }

        let fwd = self.forward_vector();
        let right = self.right_vector();

        let mut fwd_spd = self.forward_speed();
        let mut lat_spd = self.lateral_speed();

        // 1. Engine propulsion & braking forces
        let accel_force = self.throttle_input * self.config.acceleration_rate;
        let brake_force = self.brake_input * self.config.braking_rate;

        // Apply acceleration
        fwd_spd += accel_force * dt;

        // Apply braking
        if fwd_spd > 0.0 {
            fwd_spd = (fwd_spd - brake_force * dt).max(0.0);
        } else if self.throttle_input < 0.05 && self.brake_input > 0.1 {
            // Reverse when stopped
            fwd_spd = (fwd_spd - self.config.acceleration_rate * 0.4 * dt)
                .max(-self.config.max_reverse_speed);
        }

        // Top speed clamp
        fwd_spd = fwd_spd.clamp(-self.config.max_reverse_speed, self.config.max_forward_speed);

        // 2. Air resistance and rolling friction
        let drag_force = fwd_spd * fwd_spd.abs() * self.config.drag_coeff;
        let rolling_friction = fwd_spd * 0.25;
        fwd_spd -= (drag_force + rolling_friction) * dt;

        // 3. Lateral grip / drift physics: damp lateral sliding velocity
        let lateral_friction_factor = (1.0 - self.config.lateral_grip * (dt * 60.0).min(1.0)).max(0.0);
        lat_spd *= lateral_friction_factor;

        // Reconstruct 2D velocity vector
        self.velocity = fwd * fwd_spd + right * lat_spd;

        // 4. Steering and yaw rotation
        // Steering response scales with speed (cannot turn while completely stationary)
        let speed_factor = (fwd_spd.abs() / 120.0).min(1.0);
        let turn_direction = if fwd_spd >= -1e-3 { 1.0 } else { -1.0 };
        let target_turn_rate = self.steer_input
            * self.config.steer_speed
            * speed_factor
            * turn_direction;

        self.angular_velocity = target_turn_rate;
        self.heading_angle += self.angular_velocity * dt;

        // Keep angle in [-PI, PI] range
        if self.heading_angle > PI {
            self.heading_angle -= 2.0 * PI;
        } else if self.heading_angle < -PI {
            self.heading_angle += 2.0 * PI;
        }

        // 5. Integrate position
        let step_dist = self.velocity.length() * dt;
        self.position += self.velocity * dt;
        self.distance_traveled += step_dist;
        self.time_alive += dt;
        self.time_since_last_checkpoint += dt;

        if fwd_spd > self.top_speed_recorded {
            self.top_speed_recorded = fwd_spd;
        }
    }

    /// Check collisions with track walls.
    pub fn check_wall_collisions(&mut self, track: &Track) {
        if !self.is_alive {
            return;
        }

        let car_edges = self.bounding_box_edges();
        for edge in &car_edges {
            for wall in &track.wall_segments {
                if edge.intersect_segment(wall).is_some() {
                    self.is_alive = false;
                    self.death_reason = DeathReason::WallCollision;
                    return;
                }
            }
        }
    }

    /// Check checkpoint crossings and calculate updated fitness.
    pub fn check_checkpoints(&mut self, track: &Track) {
        if !self.is_alive {
            return;
        }

        let num_checkpoints = track.checkpoints.len();
        if num_checkpoints == 0 {
            return;
        }

        // Check if idle timeout exceeded
        if self.time_since_last_checkpoint > self.config.checkpoint_timeout {
            self.is_alive = false;
            self.death_reason = DeathReason::IdleTimeout;
            return;
        }

        // Look at next checkpoint target
        let next_idx = (self.current_checkpoint_idx + 1) % num_checkpoints;
        let gate = &track.checkpoints[next_idx];

        // Test if car crossed the gate line segment
        let car_center_prev = self.position - self.velocity * 0.033;
        let car_motion_seg = LineSegment::new(car_center_prev, self.position);

        let crossed = car_motion_seg.intersect_segment(&gate.gate_line).is_some()
            || self.position.distance_sq(gate.center) < (track.track_width * 0.5).powi(2);

        if crossed {
            // Verify forward direction (prevent backwards cheat)
            let fwd_alignment = self.forward_vector().dot(gate.forward_normal);
            if fwd_alignment > -0.1 {
                self.current_checkpoint_idx = next_idx;
                self.checkpoints_hit += 1;
                self.time_since_last_checkpoint = 0.0;

                if next_idx == 0 && self.checkpoints_hit > num_checkpoints / 2 {
                    self.laps_completed += 1;
                }
            } else {
                // Moving backwards through checkpoint
                self.is_alive = false;
                self.death_reason = DeathReason::WrongWay;
                return;
            }
        }

        // Calculate continuous segment progress towards next checkpoint
        let curr_gate = &track.checkpoints[self.current_checkpoint_idx];
        let next_gate = &track.checkpoints[next_idx];
        let seg_vec = next_gate.center - curr_gate.center;
        let seg_len = seg_vec.length();

        let progress_frac = if seg_len > 1e-4 {
            let car_vec = self.position - curr_gate.center;
            (car_vec.dot(seg_vec) / (seg_len * seg_len)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Comprehensive fitness function:
        // - Rewards checkpoint gates reached
        // - Rewards laps completed
        // - Rewards continuous fractional distance along segment
        // - Rewards maintaining higher average speed
        let avg_speed = if self.time_alive > 0.1 {
            self.distance_traveled / self.time_alive
        } else {
            0.0
        };

        let base_fitness = (self.checkpoints_hit as f32) * 1000.0
            + (self.laps_completed as f32) * 20000.0
            + progress_frac * 500.0
            + avg_speed * 1.5;

        self.fitness = base_fitness.max(0.0);
    }

    /// Full update cycle: physics -> collision -> checkpoints -> sensors.
    pub fn update(&mut self, dt: f32, track: &Track) {
        if !self.is_alive {
            return;
        }

        self.physics_step(dt);
        self.check_wall_collisions(track);
        self.check_checkpoints(track);
        if self.is_alive {
            self.update_sensors(track);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_car_physics() {
        let mut car = Car::new(Vec2::new(100.0, 100.0), 0.0);
        car.apply_controls(0.0, 1.0, 0.0);

        for _ in 0..10 {
            car.physics_step(0.016);
        }

        assert!(car.velocity.x > 0.0);
        assert!(car.position.x > 100.0);
        assert_eq!(car.is_alive, true);
    }

    #[test]
    fn test_sensors_and_inputs() {
        let track = Track::preset_super_speedway();
        let mut car = Car::new(track.start_position, track.start_angle);
        car.update_sensors(&track);

        let inputs = car.get_network_inputs();
        assert_eq!(inputs.len(), 9);
        for &r in &inputs[0..7] {
            assert!(r >= 0.0 && r <= 1.0);
        }
    }
}
