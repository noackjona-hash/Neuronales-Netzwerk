//! Track representation, procedural spline generation, boundary walls,
//! and checkpoint reward gates built from scratch.

#![allow(dead_code)]

use crate::math::{LineSegment, Vec2};
use std::f32::consts::PI;

/// A checkpoint reward gate spanning across the track width.
#[derive(Debug, Clone)]
pub struct CheckpointGate {
    pub index: usize,
    pub center: Vec2,
    pub inner_pt: Vec2,
    pub outer_pt: Vec2,
    pub gate_line: LineSegment,
    pub forward_normal: Vec2,
    pub distance_from_start: f32,
}

/// Racetrack containing boundaries, centerline, walls, and checkpoints.
#[derive(Debug, Clone)]
pub struct Track {
    pub name: String,
    pub centerline: Vec<Vec2>,
    pub inner_boundary: Vec<Vec2>,
    pub outer_boundary: Vec<Vec2>,
    pub wall_segments: Vec<LineSegment>,
    pub checkpoints: Vec<CheckpointGate>,
    pub start_position: Vec2,
    pub start_angle: f32,
    pub total_length: f32,
    pub track_width: f32,
}

impl Track {
    /// Build a track from a closed loop of centerline waypoints and constant track width.
    pub fn from_centerline(name: &str, centerline: Vec<Vec2>, track_width: f32) -> Self {
        assert!(
            centerline.len() >= 4,
            "Track requires at least 4 centerline points"
        );

        let n = centerline.len();
        let half_w = track_width * 0.5;

        let mut inner_boundary = Vec::with_capacity(n);
        let mut outer_boundary = Vec::with_capacity(n);
        let mut checkpoints = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);

        let mut accum_dist = 0.0f32;
        distances.push(0.0);

        // Compute cumulative distances along centerline
        for i in 0..n {
            let next_i = (i + 1) % n;
            accum_dist += centerline[i].distance(centerline[next_i]);
            if i < n - 1 {
                distances.push(accum_dist);
            }
        }
        let total_length = accum_dist;

        // Compute inner and outer boundaries using smoothed vertex normals
        for i in 0..n {
            let prev_i = (i + n - 1) % n;
            let next_i = (i + 1) % n;

            let pt = centerline[i];
            let dir_in = (pt - centerline[prev_i]).normalize();
            let dir_out = (centerline[next_i] - pt).normalize();
            let avg_dir = (dir_in + dir_out).normalize();

            // Normal pointing left (inward/outward depending on winding)
            let normal = avg_dir.perpendicular();

            let inner_pt = pt + normal * half_w;
            let outer_pt = pt - normal * half_w;

            inner_boundary.push(inner_pt);
            outer_boundary.push(outer_pt);

            let gate_line = LineSegment::new(inner_pt, outer_pt);
            checkpoints.push(CheckpointGate {
                index: i,
                center: pt,
                inner_pt,
                outer_pt,
                gate_line,
                forward_normal: avg_dir,
                distance_from_start: distances[i],
            });
        }

        // Build all wall line segments (inner loop + outer loop)
        let mut wall_segments = Vec::with_capacity(2 * n);
        for i in 0..n {
            let next_i = (i + 1) % n;
            wall_segments.push(LineSegment::new(inner_boundary[i], inner_boundary[next_i]));
            wall_segments.push(LineSegment::new(outer_boundary[i], outer_boundary[next_i]));
        }

        // Determine starting position and heading angle
        let start_position = (centerline[0] + centerline[1]) * 0.5;
        let start_dir = (centerline[1] - centerline[0]).normalize();
        let start_angle = start_dir.to_angle();

        Self {
            name: name.to_string(),
            centerline,
            inner_boundary,
            outer_boundary,
            wall_segments,
            checkpoints,
            start_position,
            start_angle,
            total_length,
            track_width,
        }
    }

    /// Catmull-Rom spline interpolation between p1 and p2 using control points p0, p1, p2, p3.
    fn catmull_rom_spline(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
        let t2 = t * t;
        let t3 = t2 * t;

        let v0 = (p2 - p0) * 0.5;
        let v1 = (p3 - p1) * 0.5;

        p1 * (2.0 * t3 - 3.0 * t2 + 1.0)
            + p2 * (-2.0 * t3 + 3.0 * t2)
            + v0 * (t3 - 2.0 * t2 + t)
            + v1 * (t3 - t2)
    }

    /// Subdivide and smooth coarse control points into a smooth dense centerline.
    pub fn smooth_spline_loop(control_points: &[Vec2], subdivisions_per_segment: usize) -> Vec<Vec2> {
        let n = control_points.len();
        let mut smoothed = Vec::with_capacity(n * subdivisions_per_segment);

        for i in 0..n {
            let p0 = control_points[(i + n - 1) % n];
            let p1 = control_points[i];
            let p2 = control_points[(i + 1) % n];
            let p3 = control_points[(i + 2) % n];

            for step in 0..subdivisions_per_segment {
                let t = step as f32 / subdivisions_per_segment as f32;
                smoothed.push(Self::catmull_rom_spline(p0, p1, p2, p3, t));
            }
        }

        smoothed
    }

    // ==========================================
    // TRACK PRESETS
    // ==========================================

    /// Preset 1: Grand Prix Circuit (Formula-style technical course with straights, S-turns & hairpins).
    pub fn preset_grand_prix() -> Self {
        let raw_pts = vec![
            Vec2::new(180.0, 520.0),  // Start / Main straight
            Vec2::new(450.0, 520.0),
            Vec2::new(750.0, 520.0),
            Vec2::new(980.0, 560.0),  // Turn 1 entrance
            Vec2::new(1150.0, 480.0), // Turn 1 apex
            Vec2::new(1120.0, 320.0),
            Vec2::new(920.0, 260.0),  // Technical chicane
            Vec2::new(980.0, 160.0),
            Vec2::new(1120.0, 110.0),
            Vec2::new(920.0, 70.0),   // Top sweep
            Vec2::new(650.0, 90.0),
            Vec2::new(520.0, 200.0),  // Infield hairpin
            Vec2::new(400.0, 130.0),
            Vec2::new(260.0, 110.0),  // Left hairpin
            Vec2::new(110.0, 240.0),
            Vec2::new(140.0, 380.0),  // Final corner entry
        ];

        let smoothed = Self::smooth_spline_loop(&raw_pts, 8);
        Self::from_centerline("Grand Prix Circuit", smoothed, 76.0)
    }

    /// Preset 2: Super Speedway (High-speed banked oval with sweeping bends).
    pub fn preset_super_speedway() -> Self {
        let raw_pts = vec![
            Vec2::new(300.0, 600.0),  // Bottom straight
            Vec2::new(650.0, 600.0),
            Vec2::new(950.0, 600.0),
            Vec2::new(1160.0, 520.0), // Turn 1
            Vec2::new(1220.0, 360.0), // Turn 2
            Vec2::new(1160.0, 200.0),
            Vec2::new(950.0, 120.0),  // Top straight
            Vec2::new(650.0, 120.0),
            Vec2::new(300.0, 120.0),
            Vec2::new(110.0, 200.0),  // Turn 3
            Vec2::new(60.0, 360.0),   // Turn 4
            Vec2::new(110.0, 520.0),
        ];

        let smoothed = Self::smooth_spline_loop(&raw_pts, 10);
        Self::from_centerline("Super Speedway", smoothed, 92.0)
    }

    /// Preset 3: Hairpin & Chicane Challenge (Demanding apex precision and heavy braking).
    pub fn preset_hairpin_chicane() -> Self {
        let raw_pts = vec![
            Vec2::new(150.0, 580.0),
            Vec2::new(550.0, 580.0),
            Vec2::new(950.0, 580.0),
            Vec2::new(1120.0, 470.0), // Sweeper
            Vec2::new(900.0, 410.0),  // Sharp hairpin 1
            Vec2::new(750.0, 480.0),
            Vec2::new(600.0, 420.0),  // Chicane entry
            Vec2::new(650.0, 300.0),  // Chicane exit
            Vec2::new(980.0, 280.0),
            Vec2::new(1140.0, 190.0), // Top right turn
            Vec2::new(850.0, 100.0),  // Top straight
            Vec2::new(450.0, 100.0),
            Vec2::new(200.0, 140.0),  // Double apex left
            Vec2::new(120.0, 280.0),
            Vec2::new(280.0, 340.0),
            Vec2::new(120.0, 440.0),
        ];

        let smoothed = Self::smooth_spline_loop(&raw_pts, 8);
        Self::from_centerline("Hairpin & Chicane", smoothed, 72.0)
    }

    /// Preset 4: Figure-Eight Ribbon.
    pub fn preset_figure_eight() -> Self {
        let raw_pts = vec![
            Vec2::new(200.0, 520.0),
            Vec2::new(480.0, 520.0),
            Vec2::new(700.0, 400.0), // Center crossing right
            Vec2::new(950.0, 220.0),
            Vec2::new(1120.0, 150.0),
            Vec2::new(1180.0, 300.0),
            Vec2::new(1050.0, 460.0),
            Vec2::new(820.0, 520.0),
            Vec2::new(650.0, 360.0), // Center crossing left
            Vec2::new(420.0, 180.0),
            Vec2::new(200.0, 130.0),
            Vec2::new(100.0, 260.0),
            Vec2::new(110.0, 410.0),
        ];

        let smoothed = Self::smooth_spline_loop(&raw_pts, 8);
        Self::from_centerline("Figure-Eight Ribbon", smoothed, 74.0)
    }

    /// Preset 5: Procedural Randomized Track.
    pub fn preset_procedural(seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::Rng;

        let mut rng = StdRng::seed_from_u64(seed);
        let num_nodes = rng.gen_range(10..15);
        let center = Vec2::new(640.0, 360.0);
        let base_rx = 480.0f32;
        let base_ry = 260.0f32;

        let mut pts = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let angle = (i as f32 / num_nodes as f32) * 2.0 * PI;
            let radius_jitter: f32 = rng.gen_range(0.70..1.25);
            let angle_jitter: f32 = rng.gen_range(-0.10..0.10);

            let eff_angle = angle + angle_jitter;
            let x = center.x + eff_angle.cos() * base_rx * radius_jitter;
            let y = center.y + eff_angle.sin() * base_ry * radius_jitter;
            pts.push(Vec2::new(x, y));
        }

        let smoothed = Self::smooth_spline_loop(&pts, 10);
        let track_name = format!("Procedural Circuit #{}", seed % 1000);
        Self::from_centerline(&track_name, smoothed, 78.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_generation() {
        let track = Track::preset_grand_prix();
        assert!(track.centerline.len() > 50);
        assert_eq!(track.wall_segments.len(), track.centerline.len() * 2);
        assert_eq!(track.checkpoints.len(), track.centerline.len());
        assert!(track.total_length > 1000.0);
    }
}
