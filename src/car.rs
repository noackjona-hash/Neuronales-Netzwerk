//! Realistic 2D Vehicle Dynamics (Bicycle / Two-Track Model with Tire Slip Angles,
//! Lateral Grip Saturation, Weight Transfer, and Skid Mark Generation),
//! Dynamic Lookahead Raycast Sensor Array, OBB Collision Detection, and Pro-Racer Fitness.

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

/// Vehicle Physical Parameters.
#[derive(Debug, Clone, Copy)]
pub struct CarConfig {
    pub length: f32,
    pub width: f32,
    pub mass: f32,
    pub inertia: f32,
    pub dist_cg_front: f32,
    pub dist_cg_rear: f32,
    pub cg_height: f32,
    
    // Tires & Cornering
    pub cornering_stiffness_front: f32,
    pub cornering_stiffness_rear: f32,
    pub friction_coeff: f32,
    
    // Drivetrain & Aerodynamics
    pub max_engine_force: f32,
    pub max_brake_force: f32,
    pub max_reverse_force: f32,
    pub max_steer_angle: f32,
    pub steer_speed: f32,
    pub drag_coeff: f32,
    pub rolling_resistance: f32,
    
    // Sensors & Rules
    pub ray_sensor_base_range: f32,
    pub ray_sensor_angles: [f32; 9],
    pub checkpoint_timeout: f32,
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
            cornering_stiffness_front: 48000.0,
            cornering_stiffness_rear: 55000.0,
            friction_coeff: 1.20,
            max_engine_force: 7800.0,
            max_brake_force: 12000.0,
            max_reverse_force: 3500.0,
            max_steer_angle: 0.60,
            steer_speed: 7.2,
            drag_coeff: 0.36,
            rolling_resistance: 12.0,
            ray_sensor_base_range: 280.0,
            ray_sensor_angles: [
                -85.0 * PI / 180.0,
                -60.0 * PI / 180.0,
                -35.0 * PI / 180.0,
                -15.0 * PI / 180.0,
                0.0,
                15.0 * PI / 180.0,
                35.0 * PI / 180.0,
                60.0 * PI / 180.0,
                85.0 * PI / 180.0,
            ],
            checkpoint_timeout: 4.5,
        }
    }
}

/// 2D Realistic Physics Car with 14 High-Dimensional Telemetry Inputs.
#[derive(Debug, Clone)]
pub struct Car {
    pub config: CarConfig,
    pub position: Vec2,
    
    // Body state in local & world coordinates
    pub heading_angle: f32,
    pub velocity_local: Vec2,   // (u = longitudinal forward, v = lateral sideways)
    pub yaw_rate: f32,
    pub steer_angle: f32,
    
    // Control inputs
    pub steer_input: f32,
    pub throttle_input: f32,
    pub brake_input: f32,

    // Telemetry & Lookahead
    pub slip_angle_front: f32,
    pub slip_angle_rear: f32,
    pub lateral_force_front: f32,
    pub lateral_force_rear: f32,
    pub is_skidding: bool,
    pub target_apex_angle_1: f32, // Next checkpoint angle difference
    pub target_apex_angle_2: f32, // 2nd checkpoint curvature lookahead

    // 9 Raycast Sensors
    pub sensor_readings: [f32; 9],
    pub sensor_hits: [Option<RaycastHit>; 9],

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
    pub clean_racing_line_score: f32,
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
            target_apex_angle_1: 0.0,
            target_apex_angle_2: 0.0,
            sensor_readings: [1.0; 9],
            sensor_hits: [None; 9],
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
            clean_racing_line_score: 0.0,
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
        self.target_apex_angle_1 = 0.0;
        self.target_apex_angle_2 = 0.0;
        self.sensor_readings = [1.0; 9];
        self.sensor_hits = [None; 9];
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
        self.clean_racing_line_score = 0.0;
    }

    #[inline(always)]
    pub fn forward_vector(&self) -> Vec2 {
        Vec2::from_angle(self.heading_angle)
    }

    #[inline(always)]
    pub fn right_vector(&self) -> Vec2 {
        self.forward_vector().perpendicular()
    }

    #[inline(always)]
    pub fn world_velocity(&self) -> Vec2 {
        self.forward_vector() * self.velocity_local.x + self.right_vector() * self.velocity_local.y
    }

    #[inline(always)]
    pub fn forward_speed(&self) -> f32 {
        self.velocity_local.x
    }

    #[inline(always)]
    pub fn lateral_speed(&self) -> f32 {
        self.velocity_local.y
    }

    pub fn bounding_box_corners(&self) -> [Vec2; 4] {
        let fwd = self.forward_vector() * (self.config.length * 0.5);
        let right = self.right_vector() * (self.config.width * 0.5);

        [
            self.position + fwd - right,
            self.position + fwd + right,
            self.position - fwd + right,
            self.position - fwd - right,
        ]
    }

    pub fn bounding_box_edges(&self) -> [LineSegment; 4] {
        let pts = self.bounding_box_corners();
        [
            LineSegment::new(pts[0], pts[1]),
            LineSegment::new(pts[1], pts[2]),
            LineSegment::new(pts[2], pts[3]),
            LineSegment::new(pts[3], pts[0]),
        ]
    }

    pub fn current_sensor_range(&self) -> f32 {
        let speed = self.velocity_local.x.abs();
        self.config.ray_sensor_base_range + speed * 4.5
    }

    pub fn update_sensors(&mut self, track: &Track) {
        let fwd_angle = self.heading_angle;
        let sensor_origin = self.position + self.forward_vector() * (self.config.length * 0.4);
        let ray_range = self.current_sensor_range();

        for (i, &rel_angle) in self.config.ray_sensor_angles.iter().enumerate() {
            let ray_angle = fwd_angle + rel_angle;
            let ray_dir = Vec2::from_angle(ray_angle);
            let ray = Ray2::new(sensor_origin, ray_dir, ray_range);

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

        // Lookahead 1 & 2 Checkpoint Angles
        let num_checkpoints = track.checkpoints.len();
        if num_checkpoints > 0 {
            let next_idx_1 = (self.current_checkpoint_idx + 1) % num_checkpoints;
            let next_idx_2 = (self.current_checkpoint_idx + 3) % num_checkpoints;

            let target_center_1 = track.checkpoints[next_idx_1].center;
            let to_target_1 = (target_center_1 - self.position).normalize();
            let mut diff_1 = to_target_1.to_angle() - self.heading_angle;
            while diff_1 > PI { diff_1 -= 2.0 * PI; }
            while diff_1 < -PI { diff_1 += 2.0 * PI; }
            self.target_apex_angle_1 = (diff_1 / PI).clamp(-1.0, 1.0);

            let target_center_2 = track.checkpoints[next_idx_2].center;
            let to_target_2 = (target_center_2 - target_center_1).normalize();
            let mut diff_2 = to_target_2.to_angle() - self.heading_angle;
            while diff_2 > PI { diff_2 -= 2.0 * PI; }
            while diff_2 < -PI { diff_2 += 2.0 * PI; }
            self.target_apex_angle_2 = (diff_2 / PI).clamp(-1.0, 1.0);
        }
    }

    /// Build rich 14-dimensional neural network input vector:
    /// [Ray0..Ray8 (9 dynamic distance readings), Forward Speed, Lateral Speed, Steering Angle, Apex Angle 1, Lookahead Curvature 2].
    pub fn get_network_inputs(&self) -> [f32; 14] {
        let norm_speed = (self.velocity_local.x / 40.0).clamp(-0.5, 1.5);
        let norm_lat_speed = (self.velocity_local.y / 15.0).clamp(-1.0, 1.0);
        let norm_steer = (self.steer_angle / self.config.max_steer_angle.max(0.1)).clamp(-1.0, 1.0);

        [
            self.sensor_readings[0].clamp(0.0, 1.0),
            self.sensor_readings[1].clamp(0.0, 1.0),
            self.sensor_readings[2].clamp(0.0, 1.0),
            self.sensor_readings[3].clamp(0.0, 1.0),
            self.sensor_readings[4].clamp(0.0, 1.0),
            self.sensor_readings[5].clamp(0.0, 1.0),
            self.sensor_readings[6].clamp(0.0, 1.0),
            self.sensor_readings[7].clamp(0.0, 1.0),
            self.sensor_readings[8].clamp(0.0, 1.0),
            norm_speed,
            norm_lat_speed,
            norm_steer,
            self.target_apex_angle_1,
            self.target_apex_angle_2,
        ]
    }

    pub fn apply_controls(&mut self, steer: f32, throttle: f32, brake: f32) {
        self.steer_input = if steer.is_nan() { 0.0 } else { steer.clamp(-1.0, 1.0) };
        self.throttle_input = if throttle.is_nan() { 0.0 } else { throttle.clamp(0.0, 1.0) };
        self.brake_input = if brake.is_nan() { 0.0 } else { brake.clamp(0.0, 1.0) };
    }

    pub fn physics_step(&mut self, dt: f32) {
        if !self.is_alive {
            return;
        }

        let g = 9.81f32;
        let mass = self.config.mass.max(100.0);
        let a = self.config.dist_cg_front.max(0.1);
        let b = self.config.dist_cg_rear.max(0.1);
        let l = (a + b).max(0.2);
        let h_cg = self.config.cg_height.max(0.05);
        let iz = self.config.inertia.max(100.0);

        // 1. Actuate Steering with smooth actuator speed
        let target_steer = self.steer_input * self.config.max_steer_angle;
        let steer_diff = target_steer - self.steer_angle;
        let max_steer_step = self.config.steer_speed * dt;
        self.steer_angle += steer_diff.clamp(-max_steer_step, max_steer_step);

        let mut u = if self.velocity_local.x.is_nan() { 0.0 } else { self.velocity_local.x };
        let mut v = if self.velocity_local.y.is_nan() { 0.0 } else { self.velocity_local.y };
        let mut omega = if self.yaw_rate.is_nan() { 0.0 } else { self.yaw_rate };

        // 2. Normal axle loads with dynamic weight transfer
        let weight_front_static = (b / l) * mass * g;
        let weight_rear_static = (a / l) * mass * g;

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
            let drive_force = self.throttle_input * self.config.max_engine_force * 0.35
                - self.brake_input * self.config.max_brake_force * 0.45;
            u += (drive_force / mass) * dt;
            u = u.clamp(-6.0, 48.0);
            v *= 0.85;

            let target_omega = (u / l) * self.steer_angle.tan();
            omega = target_omega;
            self.slip_angle_front = 0.0;
            self.slip_angle_rear = 0.0;
            self.is_skidding = false;
        } else {
            let safe_u = u.abs().max(1.5) * if u >= 0.0 { 1.0 } else { -1.0 };

            let alpha_f = ((v + a * omega) / safe_u).atan() - self.steer_angle;
            let alpha_r = ((v - b * omega) / safe_u).atan();

            self.slip_angle_front = if alpha_f.is_nan() { 0.0 } else { alpha_f };
            self.slip_angle_rear = if alpha_r.is_nan() { 0.0 } else { alpha_r };

            let max_lat_f = (self.config.friction_coeff * fz_front).max(10.0);
            let max_lat_r = (self.config.friction_coeff * fz_rear).max(10.0);

            let fy_f = -max_lat_f * (self.config.cornering_stiffness_front * alpha_f / max_lat_f).tanh();
            let fy_r = -max_lat_r * (self.config.cornering_stiffness_rear * alpha_r / max_lat_r).tanh();

            self.lateral_force_front = if fy_f.is_nan() { 0.0 } else { fy_f };
            self.lateral_force_rear = if fy_r.is_nan() { 0.0 } else { fy_r };

            self.is_skidding = alpha_f.abs() > 0.14 || alpha_r.abs() > 0.12;

            let engine_force = self.throttle_input * self.config.max_engine_force;
            let brake_force = if u > 0.1 {
                self.brake_input * self.config.max_brake_force
            } else if self.throttle_input < 0.05 && self.brake_input > 0.1 {
                -self.config.max_reverse_force
            } else {
                0.0
            };

            let aero_drag = self.config.drag_coeff * u * u.abs();
            let rolling_res = self.config.rolling_resistance * mass * g * 0.001 * if u >= 0.0 { 1.0 } else { -1.0 };

            let fx_rear = engine_force - brake_force * 0.5;
            let fx_front = -brake_force * 0.5;

            let total_fx = fx_rear + fx_front * self.steer_angle.cos()
                - fy_f * self.steer_angle.sin()
                - aero_drag
                - rolling_res;
            let total_fy = fy_r + fy_f * self.steer_angle.cos() + fx_front * self.steer_angle.sin();
            let yaw_torque = a * (fy_f * self.steer_angle.cos() + fx_front * self.steer_angle.sin()) - b * fy_r;

            let du_dt = total_fx / mass + v * omega;
            let dv_dt = total_fy / mass - u * omega;
            let domega_dt = yaw_torque / iz;

            if !du_dt.is_nan() {
                u += du_dt * dt;
            }
            if !dv_dt.is_nan() {
                v += dv_dt * dt;
            }
            if !domega_dt.is_nan() {
                omega += domega_dt * dt;
            }

            u = u.clamp(-8.0, 55.0);
            v *= 0.98;
            omega *= 0.96;
        }

        self.velocity_local = Vec2::new(u, v);
        self.yaw_rate = omega;

        // 4. Integrate Orientation & World Position
        self.heading_angle += self.yaw_rate * dt;

        if self.heading_angle > PI {
            self.heading_angle -= 2.0 * PI;
        } else if self.heading_angle < -PI {
            self.heading_angle += 2.0 * PI;
        }

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

    pub fn check_checkpoints(&mut self, track: &Track) {
        if !self.is_alive {
            return;
        }

        let num_checkpoints = track.checkpoints.len();
        if num_checkpoints == 0 {
            return;
        }

        if self.time_since_last_checkpoint > self.config.checkpoint_timeout {
            self.is_alive = false;
            self.death_reason = DeathReason::IdleTimeout;
            return;
        }

        let next_idx = (self.current_checkpoint_idx + 1) % num_checkpoints;
        let gate = &track.checkpoints[next_idx];

        let car_center_prev = self.position - self.world_velocity() * 0.04;
        let car_motion_seg = LineSegment::new(car_center_prev, self.position);

        let crossed = car_motion_seg.intersect_segment(&gate.gate_line).is_some()
            || self.position.distance_sq(gate.center) < (track.track_width * 0.55).powi(2);

        if crossed {
            let fwd_alignment = self.forward_vector().dot(gate.forward_normal);
            if fwd_alignment > -0.15 {
                self.current_checkpoint_idx = next_idx;
                self.checkpoints_hit += 1;
                self.time_since_last_checkpoint = 0.0;

                // Centerline precision reward
                let dist_from_center = self.position.distance(gate.center);
                let center_quality = (1.0 - (dist_from_center / (track.track_width * 0.5))).clamp(0.0, 1.0);
                self.clean_racing_line_score += center_quality * 50.0;

                if next_idx == 0 && self.checkpoints_hit > num_checkpoints / 2 {
                    self.laps_completed += 1;
                }
            } else {
                self.is_alive = false;
                self.death_reason = DeathReason::WrongWay;
                return;
            }
        }

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

        let avg_speed = if self.time_alive > 0.1 {
            self.distance_traveled / self.time_alive
        } else {
            0.0
        };

        let base_fitness = (self.checkpoints_hit as f32) * 1200.0
            + (self.laps_completed as f32) * 35000.0
            + progress_frac * 800.0
            + avg_speed * 3.5
            + self.clean_racing_line_score;

        self.fitness = if base_fitness.is_nan() { 0.0 } else { base_fitness.max(0.0) };
    }

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
        assert_eq!(inputs.len(), 14);
        for &r in &inputs[0..9] {
            assert!(r >= 0.0 && r <= 1.0);
        }
    }
}
