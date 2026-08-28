//! Realistic 2D Vehicle Dynamics (Bicycle / Two-Track Model with Tire Slip Angles,
//! Lateral Grip Saturation, Weight Transfer, and Skid Mark Generation),
//! Raycast Sensor Array, OBB Collision Detection, and Fitness Progression.

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

/// Vehicle Physical Parameters (based on real-world sports car dynamics).
#[derive(Debug, Clone, Copy)]
pub struct CarConfig {
    // Physical dimensions & mass
    pub length: f32,          // Visual length (pixels/units)
    pub width: f32,           // Visual width (pixels/units)
    pub mass: f32,            // Mass (kg)
    pub inertia: f32,         // Yaw moment of inertia Iz (kg*m^2)
    pub dist_cg_front: f32,   // Distance from CG to front axle (m)
    pub dist_cg_rear: f32,    // Distance from CG to rear axle (m)
    pub cg_height: f32,       // CG height for dynamic weight transfer (m)
    
    // Tires & Cornering
    pub cornering_stiffness_front: f32, // Front tire cornering stiffness (N/rad)
    pub cornering_stiffness_rear: f32,  // Rear tire cornering stiffness (N/rad)
    pub friction_coeff: f32,            // Peak tire friction coefficient (mu)
    
    // Drivetrain & Aerodynamics
    pub max_engine_force: f32,          // Maximum forward propulsion force (N)
    pub max_brake_force: f32,           // Maximum braking force (N)
    pub max_reverse_force: f32,         // Maximum reverse force (N)
    pub max_steer_angle: f32,           // Maximum wheel steering angle (radians)
    pub steer_speed: f32,               // Steering actuator speed (rad/s)
    pub drag_coeff: f32,                // Aerodynamic drag coefficient
    pub rolling_resistance: f32,        // Rolling resistance coefficient
    
    // Sensors & Rules
    pub ray_sensor_range: f32,
    pub ray_sensor_angles: [f32; 7],
    pub checkpoint_timeout: f32,        // Seconds before timeout
}

impl Default for CarConfig {
    fn default() -> Self {
        Self {
            length: 32.0,
            width: 16.0,
            mass: 1100.0,
            inertia: 1750.0,
            dist_cg_front: 1.25,
            dist_cg_rear: 1.35,
            cg_height: 0.45,
            cornering_stiffness_front: 45000.0,
            cornering_stiffness_rear: 52000.0,
            friction_coeff: 1.15,
            max_engine_force: 7200.0,
            max_brake_force: 11000.0,
            max_reverse_force: 3500.0,
            max_steer_angle: 0.58, // ~33.2 degrees
            steer_speed: 6.5,
            drag_coeff: 0.38,
            rolling_resistance: 12.0,
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

/// A persistent skid mark on the track.
#[derive(Debug, Clone, Copy)]
pub struct SkidMark {
    pub start: Vec2,
    pub end: Vec2,
    pub alpha: f32,
}

/// 2D Realistic Physics Car.
#[derive(Debug, Clone)]
pub struct Car {
    pub config: CarConfig,
    pub position: Vec2,
    
    // Body state in local & world coordinates
    pub heading_angle: f32,     // Yaw orientation (rad)
    pub velocity_local: Vec2,   // (u = longitudinal forward, v = lateral sideways)
    pub yaw_rate: f32,          // Angular velocity omega (rad/s)
    pub steer_angle: f32,       // Current front wheel steering angle (rad)
    
    // Control inputs
    pub steer_input: f32,       // [-1.0, 1.0]
    pub throttle_input: f32,    // [0.0, 1.0]
    pub brake_input: f32,       // [0.0, 1.0]

    // Realistic physics telemetry
    pub slip_angle_front: f32,
    pub slip_angle_rear: f32,
    pub lateral_force_front: f32,
    pub lateral_force_rear: f32,
    pub is_skidding: bool,

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
    pub fn new(position: Vec2, heading_angle: f32) -> Self {
        Self::with_config(position, heading_angle, CarConfig::default())
    }

    pub fn with_config(position: Vec2, heading_angle: f32, config: CarConfig) -> Self {
        Self {
            config,
            position,
            heading_angle,
            velocity_local: Vec2::ZERO,
            yaw_rate: 0.0,
            steer_angle: 0.0,
            steer_input: 0.0,
            throttle_input: 0.0,
            brake_input: 0.0,
            slip_angle_front: 0.0,
            slip_angle_rear: 0.0,
            lateral_force_front: 0.0,
            lateral_force_rear: 0.0,
            is_skidding: false,
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

    pub fn reset(&mut self, position: Vec2, heading_angle: f32) {
        self.position = position;
        self.heading_angle = heading_angle;
        self.velocity_local = Vec2::ZERO;
        self.yaw_rate = 0.0;
        self.steer_angle = 0.0;
        self.steer_input = 0.0;
        self.throttle_input = 0.0;
        self.brake_input = 0.0;
        self.slip_angle_front = 0.0;
        self.slip_angle_rear = 0.0;
        self.lateral_force_front = 0.0;
        self.lateral_force_rear = 0.0;
        self.is_skidding = false;
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

    /// Forward unit vector in world coordinates.
    #[inline(always)]
    pub fn forward_vector(&self) -> Vec2 {
        Vec2::from_angle(self.heading_angle)
    }

    /// Right (lateral) unit vector in world coordinates.
    #[inline(always)]
    pub fn right_vector(&self) -> Vec2 {
        self.forward_vector().perpendicular()
    }

    /// World velocity vector.
    #[inline(always)]
    pub fn world_velocity(&self) -> Vec2 {
        self.forward_vector() * self.velocity_local.x + self.right_vector() * self.velocity_local.y
    }

    /// Forward speed in body frame.
    #[inline(always)]
    pub fn forward_speed(&self) -> f32 {
        self.velocity_local.x
    }

    /// Lateral speed in body frame.
    #[inline(always)]
    pub fn lateral_speed(&self) -> f32 {
        self.velocity_local.y
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

    /// Build neural network input vector (11 rich inputs for deep neural control):
    /// [Ray0..Ray6 (7 distances), Normalized Speed, Lateral Slip, Yaw Rate, Steer Angle].
    pub fn get_network_inputs(&self) -> [f32; 11] {
        let norm_speed = (self.velocity_local.x / 40.0).clamp(-0.5, 1.5);
        let norm_lat_speed = (self.velocity_local.y / 15.0).clamp(-1.0, 1.0);
        let norm_yaw_rate = (self.yaw_rate / 3.0).clamp(-1.0, 1.0);
        let norm_steer = (self.steer_angle / self.config.max_steer_angle).clamp(-1.0, 1.0);

        [
            self.sensor_readings[0],
            self.sensor_readings[1],
            self.sensor_readings[2],
            self.sensor_readings[3],
            self.sensor_readings[4],
            self.sensor_readings[5],
            self.sensor_readings[6],
            norm_speed,
            norm_lat_speed,
            norm_yaw_rate,
            norm_steer,
        ]
    }

    /// Apply control inputs: steer ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1].
    pub fn apply_controls(&mut self, steer: f32, throttle: f32, brake: f32) {
        self.steer_input = steer.clamp(-1.0, 1.0);
        self.throttle_input = throttle.clamp(0.0, 1.0);
        self.brake_input = brake.clamp(0.0, 1.0);
    }

    /// Realistic 2D Non-Linear Bicycle Dynamics Physics Step.
    pub fn physics_step(&mut self, dt: f32) {
        if !self.is_alive {
            return;
        }

        let g = 9.81f32;
        let mass = self.config.mass;
        let a = self.config.dist_cg_front;
        let b = self.config.dist_cg_rear;
        let l = a + b;
        let h_cg = self.config.cg_height;
        let iz = self.config.inertia;

        // 1. Actuate Steering with realistic actuator speed
        let target_steer = self.steer_input * self.config.max_steer_angle;
        let steer_diff = target_steer - self.steer_angle;
        let max_steer_step = self.config.steer_speed * dt;
        self.steer_angle += steer_diff.clamp(-max_steer_step, max_steer_step);

        let mut u = self.velocity_local.x; // Longitudinal speed
        let mut v = self.velocity_local.y; // Lateral speed
        let mut omega = self.yaw_rate;     // Yaw rate

        // 2. Compute normal axle loads with longitudinal weight transfer
        let weight_front_static = (b / l) * mass * g;
        let weight_rear_static = (a / l) * mass * g;

        // Approximate longitudinal acceleration for weight transfer
        let approx_ax = (self.throttle_input * self.config.max_engine_force
            - self.brake_input * self.config.max_brake_force)
            / mass;
        let delta_weight = (h_cg / l) * mass * approx_ax;

        let fz_front = (weight_front_static - delta_weight).clamp(mass * g * 0.1, mass * g * 0.9);
        let fz_rear = (weight_rear_static + delta_weight).clamp(mass * g * 0.1, mass * g * 0.9);

        // 3. Low-Speed Kinematic Blend vs High-Speed Dynamic Model
        let speed_mag = (u * u + v * v).sqrt();
        let kinematic_blend = (1.0 - (speed_mag / 3.0)).clamp(0.0, 1.0);

        if kinematic_blend > 0.95 {
            // Pure low-speed kinematic regime
            let drive_force = self.throttle_input * self.config.max_engine_force * 0.3
                - self.brake_input * self.config.max_brake_force * 0.4;
            u += (drive_force / mass) * dt;
            u = u.clamp(-6.0, 45.0);
            v *= 0.85; // Suppress sideways drift when almost stopped

            // Kinematic yaw rate
            let target_omega = (u / l) * self.steer_angle.tan();
            omega = target_omega;
            self.slip_angle_front = 0.0;
            self.slip_angle_rear = 0.0;
            self.is_skidding = false;
        } else {
            // Full dynamic regime with Pacejka-like non-linear tire curves
            let safe_u = u.abs().max(1.5) * u.signum();

            // Slip angles (rad)
            let alpha_f = ((v + a * omega) / safe_u).atan() - self.steer_angle;
            let alpha_r = ((v - b * omega) / safe_u).atan();

            self.slip_angle_front = alpha_f;
            self.slip_angle_rear = alpha_r;

            // Non-linear lateral tire forces (tanh saturation model)
            let max_lat_f = self.config.friction_coeff * fz_front;
            let max_lat_r = self.config.friction_coeff * fz_rear;

            let fy_f = -max_lat_f * (self.config.cornering_stiffness_front * alpha_f / max_lat_f).tanh();
            let fy_r = -max_lat_r * (self.config.cornering_stiffness_rear * alpha_r / max_lat_r).tanh();

            self.lateral_force_front = fy_f;
            self.lateral_force_rear = fy_r;

            // Detect active skidding/drifting
            self.is_skidding = alpha_f.abs() > 0.14 || alpha_r.abs() > 0.12;

            // Longitudinal forces
            let engine_force = self.throttle_input * self.config.max_engine_force;
            let brake_force = if u > 0.1 {
                self.brake_input * self.config.max_brake_force
            } else if self.throttle_input < 0.05 && self.brake_input > 0.1 {
                -self.config.max_reverse_force
            } else {
                0.0
            };

            let aero_drag = self.config.drag_coeff * u * u.abs();
            let rolling_res = self.config.rolling_resistance * mass * g * 0.001 * u.signum();

            let fx_rear = engine_force - brake_force * 0.5;
            let fx_front = -brake_force * 0.5;

            // Equations of motion in vehicle frame
            let total_fx = fx_rear + fx_front * self.steer_angle.cos()
                - fy_f * self.steer_angle.sin()
                - aero_drag
                - rolling_res;
            let total_fy = fy_r + fy_f * self.steer_angle.cos() + fx_front * self.steer_angle.sin();
            let yaw_torque = a * (fy_f * self.steer_angle.cos() + fx_front * self.steer_angle.sin()) - b * fy_r;

            // Acceleration derivatives (including non-inertial Coriolis/centrifugal acceleration v * omega)
            let du_dt = total_fx / mass + v * omega;
            let dv_dt = total_fy / mass - u * omega;
            let domega_dt = yaw_torque / iz;

            u += du_dt * dt;
            v += dv_dt * dt;
            omega += domega_dt * dt;

            // Apply damping at near-zero velocity
            u = u.clamp(-8.0, 52.0);
            v *= 0.98;
            omega *= 0.96;
        }

        self.velocity_local = Vec2::new(u, v);
        self.yaw_rate = omega;

        // 4. Integrate Orientation & World Position
        self.heading_angle += self.yaw_rate * dt;

        // Normalize angle into [-PI, PI]
        if self.heading_angle > PI {
            self.heading_angle -= 2.0 * PI;
        } else if self.heading_angle < -PI {
            self.heading_angle += 2.0 * PI;
        }

        // Scale physics velocity (m/s) to screen pixels (approx 12 pixels per meter)
        let pixels_per_meter = 12.5f32;
        let world_vel_pixels = self.world_velocity() * pixels_per_meter;

        let step_dist = world_vel_pixels.length() * dt;
        self.position += world_vel_pixels * dt;
        self.distance_traveled += step_dist;
        self.time_alive += dt;
        self.time_since_last_checkpoint += dt;

        let cur_speed_kmh = u.abs() * 3.6;
        if cur_speed_kmh > self.top_speed_recorded {
            self.top_speed_recorded = cur_speed_kmh;
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
        let car_center_prev = self.position - self.world_velocity() * 0.04;
        let car_motion_seg = LineSegment::new(car_center_prev, self.position);

        let crossed = car_motion_seg.intersect_segment(&gate.gate_line).is_some()
            || self.position.distance_sq(gate.center) < (track.track_width * 0.5).powi(2);

        if crossed {
            // Verify forward direction (prevent backwards cheat)
            let fwd_alignment = self.forward_vector().dot(gate.forward_normal);
            if fwd_alignment > -0.15 {
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

        // Multi-factor reward fitness function
        let avg_speed = if self.time_alive > 0.1 {
            self.distance_traveled / self.time_alive
        } else {
            0.0
        };

        let base_fitness = (self.checkpoints_hit as f32) * 1000.0
            + (self.laps_completed as f32) * 25000.0
            + progress_frac * 600.0
            + avg_speed * 2.0;

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

        for _ in 0..20 {
            car.physics_step(0.016);
        }

        assert!(car.velocity_local.x > 0.0);
        assert!(car.position.x > 100.0);
        assert_eq!(car.is_alive, true);
    }

    #[test]
    fn test_sensors_and_inputs() {
        let track = Track::preset_super_speedway();
        let mut car = Car::new(track.start_position, track.start_angle);
        car.update_sensors(&track);

        let inputs = car.get_network_inputs();
        assert_eq!(inputs.len(), 11);
        for &r in &inputs[0..7] {
            assert!(r >= 0.0 && r <= 1.0);
        }
    }
}
