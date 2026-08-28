//! Autonomous 2D Car Racing Simulation with Neural Networks and Genetic Evolution.
//! Built entirely from scratch in pure Rust with Macroquad hardware-accelerated rendering.

mod car;
mod evolution;
mod math;
mod nn;
mod track;

use ::rand::Rng;
use car::Car;
use evolution::{EvolutionConfig, Population};
use macroquad::prelude::*;
use math::Vec2;
use nn::NeuralNetwork;
use track::Track;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    FollowBest,
    TrackOverview,
    FreePan,
}

pub struct AppState {
    pub track_idx: usize,
    pub track: Track,
    pub population: Population,
    pub player_car: Option<Car>,
    pub manual_drive: bool,
    pub paused: bool,
    pub sim_speed: usize, // 1x, 2x, 5x, 10x, 25x, 50x
    pub camera_mode: CameraMode,
    pub cam_pos: Vec2,
    pub cam_target: Vec2,
    pub cam_zoom: f32,
    pub last_mouse_pos: Vec2,
    pub show_nn_hud: bool,
    pub show_graph: bool,
    pub show_help: bool,
    pub show_sensors: bool,
    pub toast_message: Option<(String, f32)>, // message, display timer
}

fn window_conf() -> Conf {
    Conf {
        window_title: "NeuroRacer - Neural Network & Evolution Simulation from Scratch".to_string(),
        window_width: 1280,
        window_height: 720,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut state = init_app_state();
    let dt_fixed = 1.0 / 60.0;

    loop {
        let frame_dt = get_frame_time().min(0.1);

        // Handle Toast notifications
        if let Some((_, timer)) = &mut state.toast_message {
            *timer -= frame_dt;
            if *timer <= 0.0 {
                state.toast_message = None;
            }
        }

        // Process User Inputs & Hotkeys
        handle_input(&mut state);

        // Simulation Physics Step (with speed multiplier)
        if !state.paused {
            let steps = state.sim_speed;
            for _ in 0..steps {
                // Step Population
                state.population.step(dt_fixed, &state.track);

                // Step Manual Player Car if active
                if state.manual_drive {
                    if let Some(player) = &mut state.player_car {
                        handle_player_drive(player);
                        player.update(dt_fixed, &state.track);
                    }
                }

                // Check Generation Conclusion
                if state.population.is_generation_over() {
                    state.population.advance_generation(&state.track);
                    if let Some(player) = &mut state.player_car {
                        player.reset(state.track.start_position, state.track.start_angle);
                    }
                }
            }
        }

        // Update Camera
        update_camera(&mut state, frame_dt);

        // Render Frame
        clear_background(Color::new(0.08, 0.10, 0.13, 1.0));

        // 1. World Rendering (in Camera Space)
        draw_world(&state);

        // 2. HUD & UI Overlay (in Screen Space)
        draw_hud(&state);

        next_frame().await;
    }
}

fn init_app_state() -> AppState {
    let track = Track::preset_grand_prix();
    let config = EvolutionConfig {
        population_size: 70,
        elitism_count: 5,
        mutation_rate: 0.10,
        mutation_strength: 0.25,
        tournament_size: 4,
        max_generation_time: 45.0,
        ..Default::default()
    };

    let start_pos = track.start_position;
    let start_angle = track.start_angle;
    let population = Population::new(config, &track, 1337);

    AppState {
        track_idx: 0,
        track,
        population,
        player_car: Some(Car::new(start_pos, start_angle)),
        manual_drive: false,
        paused: false,
        sim_speed: 1,
        camera_mode: CameraMode::FollowBest,
        cam_pos: start_pos,
        cam_target: start_pos,
        cam_zoom: 1.0,
        last_mouse_pos: Vec2::ZERO,
        show_nn_hud: true,
        show_graph: true,
        show_help: false,
        show_sensors: true,
        toast_message: Some(("Welcome to NeuroRacer! Press [H] for controls.".to_string(), 4.0)),
    }
}

fn handle_input(state: &mut AppState) {
    // Space: Pause/Resume
    if is_key_pressed(KeyCode::Space) {
        state.paused = !state.paused;
    }

    // Number keys: Simulation Speed
    if is_key_pressed(KeyCode::Key1) {
        state.sim_speed = 1;
    }
    if is_key_pressed(KeyCode::Key2) {
        state.sim_speed = 2;
    }
    if is_key_pressed(KeyCode::Key3) {
        state.sim_speed = 5;
    }
    if is_key_pressed(KeyCode::Key4) {
        state.sim_speed = 10;
    }
    if is_key_pressed(KeyCode::Key5) {
        state.sim_speed = 25;
    }
    if is_key_pressed(KeyCode::Key6) {
        state.sim_speed = 50;
    }

    // T: Next Track Preset
    if is_key_pressed(KeyCode::T) {
        state.track_idx = (state.track_idx + 1) % 5;
        state.track = match state.track_idx {
            0 => Track::preset_grand_prix(),
            1 => Track::preset_super_speedway(),
            2 => Track::preset_hairpin_chicane(),
            3 => Track::preset_figure_eight(),
            _ => {
                let seed = state.population.rng.gen::<u64>() % 10000;
                Track::preset_procedural(seed)
            }
        };
        state.population.reset_to_track(&state.track);
        if let Some(player) = &mut state.player_car {
            player.reset(state.track.start_position, state.track.start_angle);
        }
        state.toast_message = Some((format!("Switched to: {}", state.track.name), 2.5));
    }

    // C: Cycle Camera Mode
    if is_key_pressed(KeyCode::C) {
        state.camera_mode = match state.camera_mode {
            CameraMode::FollowBest => CameraMode::TrackOverview,
            CameraMode::TrackOverview => CameraMode::FreePan,
            CameraMode::FreePan => CameraMode::FollowBest,
        };
        let mode_name = match state.camera_mode {
            CameraMode::FollowBest => "Follow Leader",
            CameraMode::TrackOverview => "Track Overview",
            CameraMode::FreePan => "Free Pan & Zoom",
        };
        state.toast_message = Some((format!("Camera: {}", mode_name), 1.5));
    }

    // N: Toggle Neural Network HUD
    if is_key_pressed(KeyCode::N) {
        state.show_nn_hud = !state.show_nn_hud;
    }

    // G: Toggle Fitness Graph
    if is_key_pressed(KeyCode::G) {
        state.show_graph = !state.show_graph;
    }

    // H: Toggle Help
    if is_key_pressed(KeyCode::H) {
        state.show_help = !state.show_help;
    }

    // D / V: Toggle Sensor Rays
    if is_key_pressed(KeyCode::V) || is_key_pressed(KeyCode::D) {
        state.show_sensors = !state.show_sensors;
    }

    // M: Toggle Manual Drive
    if is_key_pressed(KeyCode::M) {
        state.manual_drive = !state.manual_drive;
        if state.manual_drive {
            if let Some(player) = &mut state.player_car {
                player.reset(state.track.start_position, state.track.start_angle);
            }
            state.toast_message = Some(("Manual Drive: Use [W/A/S/D] or Arrows to drive!".to_string(), 3.0));
        } else {
            state.toast_message = Some(("Manual Drive Disabled".to_string(), 2.0));
        }
    }

    // K: Kill current generation (advance immediately)
    if is_key_pressed(KeyCode::K) {
        state.population.advance_generation(&state.track);
        if let Some(player) = &mut state.player_car {
            player.reset(state.track.start_position, state.track.start_angle);
        }
        state.toast_message = Some(("Skipped to next generation".to_string(), 2.0));
    }

    // R: Reset all training
    if is_key_pressed(KeyCode::R) {
        state.population.reset_to_track(&state.track);
        state.toast_message = Some(("Reset simulation to Generation 1".to_string(), 2.0));
    }

    // S: Save best brain to JSON
    if is_key_pressed(KeyCode::S) {
        if let Some(brain) = &state.population.best_ever_brain {
            if let Ok(json) = brain.to_json() {
                if std::fs::write("best_car_nn.json", json).is_ok() {
                    state.toast_message = Some(("Saved champion brain to best_car_nn.json!".to_string(), 3.0));
                }
            }
        }
    }

    // L: Load best brain from JSON
    if is_key_pressed(KeyCode::L) {
        if let Ok(json) = std::fs::read_to_string("best_car_nn.json") {
            if let Ok(brain) = NeuralNetwork::from_json(&json) {
                state.population.inject_champion_brain(brain, &state.track);
                state.toast_message = Some(("Loaded and injected champion brain from best_car_nn.json!".to_string(), 3.0));
            }
        } else {
            state.toast_message = Some(("No best_car_nn.json file found to load".to_string(), 2.5));
        }
    }

    // Free Pan Mouse Dragging
    let cur_mouse = Vec2::new(mouse_position().0, mouse_position().1);
    if is_mouse_button_down(MouseButton::Right) || is_mouse_button_down(MouseButton::Middle) {
        let delta = cur_mouse - state.last_mouse_pos;
        state.camera_mode = CameraMode::FreePan;
        state.cam_pos -= delta / state.cam_zoom;
        state.cam_target = state.cam_pos;
    }
    state.last_mouse_pos = cur_mouse;

    // Mouse zoom in/out
    let mouse_wheel = mouse_wheel().1;
    if mouse_wheel != 0.0 {
        state.cam_zoom = (state.cam_zoom * (1.0 + mouse_wheel * 0.1)).clamp(0.15, 4.0);
    }
}

fn handle_player_drive(car: &mut Car) {
    let mut steer = 0.0f32;
    let mut throttle = 0.0f32;
    let mut brake = 0.0f32;

    if is_key_down(KeyCode::A) || is_key_down(KeyCode::Left) {
        steer -= 1.0;
    }
    if is_key_down(KeyCode::D) || is_key_down(KeyCode::Right) {
        steer += 1.0;
    }
    if is_key_down(KeyCode::W) || is_key_down(KeyCode::Up) {
        throttle = 1.0;
    }
    if is_key_down(KeyCode::S) || is_key_down(KeyCode::Down) {
        brake = 1.0;
    }

    car.apply_controls(steer, throttle, brake);
}

fn update_camera(state: &mut AppState, dt: f32) {
    let scr_w = screen_width();
    let scr_h = screen_height();

    match state.camera_mode {
        CameraMode::FollowBest => {
            let leader_idx = state.population.leader_idx();
            let leader_pos = if state.manual_drive && state.player_car.as_ref().map_or(false, |c| c.is_alive) {
                state.player_car.as_ref().unwrap().position
            } else {
                state.population.agents[leader_idx].car.position
            };

            state.cam_target = leader_pos;
            state.cam_pos = state.cam_pos.lerp(state.cam_target, (dt * 8.0).min(1.0));
            state.cam_zoom = 1.15;
        }
        CameraMode::TrackOverview => {
            // Find track bounding box
            let mut min_x = f32::MAX;
            let mut min_y = f32::MAX;
            let mut max_x = f32::MIN;
            let mut max_y = f32::MIN;

            for pt in &state.track.outer_boundary {
                min_x = min_x.min(pt.x);
                min_y = min_y.min(pt.y);
                max_x = max_x.max(pt.x);
                max_y = max_y.max(pt.y);
            }

            let center = Vec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
            let span_x = (max_x - min_x) + 160.0;
            let span_y = (max_y - min_y) + 160.0;

            let zoom_x = scr_w / span_x;
            let zoom_y = scr_h / span_y;
            let target_zoom = zoom_x.min(zoom_y).clamp(0.2, 1.5);

            state.cam_target = center;
            state.cam_pos = state.cam_pos.lerp(state.cam_target, (dt * 5.0).min(1.0));
            state.cam_zoom += (target_zoom - state.cam_zoom) * (dt * 5.0).min(1.0);
        }
        CameraMode::FreePan => {
            // Smooth free pan target
            state.cam_pos = state.cam_pos.lerp(state.cam_target, (dt * 15.0).min(1.0));
        }
    }
}

// ==========================================
// WORLD DRAWING (Track, Walls, Cars, Sensors)
// ==========================================

fn draw_world(state: &AppState) {
    let scr_w = screen_width();
    let scr_h = screen_height();

    // Setup 2D Camera
    let zoom_factor_x = (2.0 / scr_w) * state.cam_zoom;
    let zoom_factor_y = -(2.0 / scr_h) * state.cam_zoom;

    let camera = Camera2D {
        target: vec2(state.cam_pos.x, state.cam_pos.y),
        zoom: vec2(zoom_factor_x, zoom_factor_y),
        ..Default::default()
    };

    set_camera(&camera);

    // 1. Draw Asphalt Track Surface
    draw_track_surface(&state.track);

    // 2. Draw Checkpoint Gates (subtle)
    for gate in &state.track.checkpoints {
        draw_line(
            gate.inner_pt.x,
            gate.inner_pt.y,
            gate.outer_pt.x,
            gate.outer_pt.y,
            1.5,
            Color::new(0.3, 0.4, 0.5, 0.15),
        );
    }

    // 3. Draw Track Outer/Inner Barrier Walls and Kerbs
    draw_track_boundaries(&state.track);

    // 4. Draw Start/Finish Line
    let start_gate = &state.track.checkpoints[0];
    draw_line(
        start_gate.inner_pt.x,
        start_gate.inner_pt.y,
        start_gate.outer_pt.x,
        start_gate.outer_pt.y,
        4.0,
        Color::new(1.0, 1.0, 1.0, 0.9),
    );

    // 5. Draw Population Cars
    let leader_idx = state.population.leader_idx();

    // First draw dead cars (transparent)
    for agent in &state.population.agents {
        if !agent.car.is_alive {
            draw_car_body(&agent.car, Color::new(0.4, 0.4, 0.4, 0.25), false);
        }
    }

    // Then draw alive cars
    for (i, agent) in state.population.agents.iter().enumerate() {
        if agent.car.is_alive {
            let is_leader = i == leader_idx;
            let c = agent.color_rgba;
            let car_color = Color::from_rgba(c[0], c[1], c[2], c[3]);
            draw_car_body(&agent.car, car_color, is_leader);
        }
    }

    // Draw Manual Player Car if enabled
    if state.manual_drive {
        if let Some(player) = &state.player_car {
            let player_col = if player.is_alive {
                Color::new(0.2, 0.9, 1.0, 1.0)
            } else {
                Color::new(0.8, 0.2, 0.2, 0.5)
            };
            draw_car_body(player, player_col, true);
        }
    }

    // 6. Draw Sensor Rays for the Leader
    if state.show_sensors && !state.population.agents.is_empty() {
        let leader_car = if state.manual_drive && state.player_car.as_ref().map_or(false, |c| c.is_alive) {
            state.player_car.as_ref().unwrap()
        } else {
            &state.population.agents[leader_idx].car
        };

        if leader_car.is_alive {
            draw_car_sensors(leader_car);
        }
    }

    set_default_camera();
}

fn draw_track_surface(track: &Track) {
    let n = track.centerline.len();

    // Draw road segments between inner and outer boundary as quads
    for i in 0..n {
        let next_i = (i + 1) % n;

        let in0 = track.inner_boundary[i];
        let out0 = track.outer_boundary[i];
        let in1 = track.inner_boundary[next_i];
        let out1 = track.outer_boundary[next_i];

        // Asphalt Dark Gray Quad
        let asphalt_col = Color::new(0.18, 0.20, 0.23, 1.0);

        // Triangle 1: in0, out0, out1
        draw_triangle(
            vec2(in0.x, in0.y),
            vec2(out0.x, out0.y),
            vec2(out1.x, out1.y),
            asphalt_col,
        );
        // Triangle 2: in0, out1, in1
        draw_triangle(
            vec2(in0.x, in0.y),
            vec2(out1.x, out1.y),
            vec2(in1.x, in1.y),
            asphalt_col,
        );

        // Centerline dashed line
        if i % 2 == 0 {
            let c0 = track.centerline[i];
            let c1 = track.centerline[next_i];
            draw_line(c0.x, c0.y, c1.x, c1.y, 2.0, Color::new(0.9, 0.9, 0.9, 0.35));
        }
    }
}

fn draw_track_boundaries(track: &Track) {
    let n = track.centerline.len();

    for i in 0..n {
        let next_i = (i + 1) % n;

        // Inner kerb (alternating red/white)
        let kerb_col = if (i / 2) % 2 == 0 {
            Color::new(0.9, 0.2, 0.2, 1.0)
        } else {
            Color::new(0.95, 0.95, 0.95, 1.0)
        };

        let in0 = track.inner_boundary[i];
        let in1 = track.inner_boundary[next_i];
        draw_line(in0.x, in0.y, in1.x, in1.y, 3.5, kerb_col);

        let out0 = track.outer_boundary[i];
        let out1 = track.outer_boundary[next_i];
        draw_line(out0.x, out0.y, out1.x, out1.y, 3.5, kerb_col);

        // Outer barrier line
        draw_line(in0.x, in0.y, in1.x, in1.y, 1.5, Color::new(0.1, 0.1, 0.1, 0.8));
        draw_line(out0.x, out0.y, out1.x, out1.y, 1.5, Color::new(0.1, 0.1, 0.1, 0.8));
    }
}

fn draw_car_body(car: &Car, color: Color, is_highlighted: bool) {
    let corners = car.bounding_box_corners();

    // 1. Leader Glow Aura
    if is_highlighted && car.is_alive {
        draw_poly(car.position.x, car.position.y, 16, 24.0, 0.0, Color::new(1.0, 0.84, 0.0, 0.25));
    }

    // 2. Car Chassis (2 Triangles forming OBB)
    draw_triangle(
        vec2(corners[0].x, corners[0].y),
        vec2(corners[1].x, corners[1].y),
        vec2(corners[2].x, corners[2].y),
        color,
    );
    draw_triangle(
        vec2(corners[0].x, corners[0].y),
        vec2(corners[2].x, corners[2].y),
        vec2(corners[3].x, corners[3].y),
        color,
    );

    // 3. Chassis Outline
    let outline_col = if is_highlighted {
        Color::new(1.0, 0.9, 0.2, 1.0)
    } else {
        Color::new(0.1, 0.1, 0.1, 0.9)
    };

    for i in 0..4 {
        let p1 = corners[i];
        let p2 = corners[(i + 1) % 4];
        draw_line(p1.x, p1.y, p2.x, p2.y, if is_highlighted { 2.5 } else { 1.5 }, outline_col);
    }

    // 4. Windshield & Hood Arrow indicator
    let nose = car.position + car.forward_vector() * (car.config.length * 0.45);
    let front_hood = car.position + car.forward_vector() * (car.config.length * 0.15);
    let glass_l = front_hood - car.right_vector() * (car.config.width * 0.3);
    let glass_r = front_hood + car.right_vector() * (car.config.width * 0.3);

    draw_triangle(
        vec2(nose.x, nose.y),
        vec2(glass_l.x, glass_l.y),
        vec2(glass_r.x, glass_r.y),
        Color::new(0.1, 0.15, 0.2, 0.9),
    );

    // Headlights
    let fl = corners[0];
    let fr = corners[1];
    draw_circle(fl.x * 0.9 + nose.x * 0.1, fl.y * 0.9 + nose.y * 0.1, 2.0, Color::new(1.0, 1.0, 0.6, 0.9));
    draw_circle(fr.x * 0.9 + nose.x * 0.1, fr.y * 0.9 + nose.y * 0.1, 2.0, Color::new(1.0, 1.0, 0.6, 0.9));
}

fn draw_car_sensors(car: &Car) {
    let sensor_origin = car.position + car.forward_vector() * (car.config.length * 0.4);

    for (i, hit_opt) in car.sensor_hits.iter().enumerate() {
        let frac = car.sensor_readings[i];

        // Color gradient from green (far/safe) to red (close/critical)
        let ray_color = if frac > 0.65 {
            Color::new(0.2, 0.9, 0.3, 0.65)
        } else if frac > 0.30 {
            Color::new(0.95, 0.85, 0.2, 0.75)
        } else {
            Color::new(0.95, 0.2, 0.2, 0.90)
        };

        let end_pt = match hit_opt {
            Some(h) => h.point,
            None => {
                let angle = car.heading_angle + car.config.ray_sensor_angles[i];
                sensor_origin + Vec2::from_angle(angle) * car.config.ray_sensor_range
            }
        };

        // Draw ray line
        draw_line(
            sensor_origin.x,
            sensor_origin.y,
            end_pt.x,
            end_pt.y,
            1.5,
            ray_color,
        );

        // Draw collision impact point circle
        if let Some(h) = hit_opt {
            draw_circle(h.point.x, h.point.y, 3.5, Color::new(1.0, 0.2, 0.2, 0.95));
        }
    }
}

// ==========================================
// SCREEN HUD & NEURAL NETWORK VISUALIZER
// ==========================================

fn draw_hud(state: &AppState) {
    let scr_w = screen_width();
    let scr_h = screen_height();

    // 1. Top Dashboard Banner
    draw_rectangle(0.0, 0.0, scr_w, 54.0, Color::new(0.06, 0.08, 0.11, 0.92));
    draw_line(0.0, 54.0, scr_w, 54.0, 1.5, Color::new(0.2, 0.25, 0.35, 0.8));

    let alive = state.population.alive_count();
    let total = state.population.agents.len();
    let gen = state.population.generation;
    let time = state.population.generation_time;
    let best_fit = state.population.current_best_fitness();
    let record_fit = state.population.best_ever_fitness;

    let font_size = 20.0;
    let y_text = 33.0;

    // Title / Logo
    draw_text("NEURO-RACER", 18.0, y_text, 24.0, Color::new(0.2, 0.85, 1.0, 1.0));

    // Stats Cards
    draw_text(&format!("GEN: {}", gen), 210.0, y_text, font_size, WHITE);
    draw_text(
        &format!("ALIVE: {}/{}", alive, total),
        310.0,
        y_text,
        font_size,
        if alive > 5 { Color::new(0.3, 0.9, 0.4, 1.0) } else { Color::new(1.0, 0.3, 0.3, 1.0) },
    );
    draw_text(&format!("TIME: {:.1}s", time), 450.0, y_text, font_size, Color::new(0.8, 0.8, 0.9, 1.0));
    draw_text(&format!("BEST FIT: {:.0}", best_fit), 580.0, y_text, font_size, Color::new(1.0, 0.85, 0.2, 1.0));
    draw_text(&format!("RECORD: {:.0}", record_fit), 750.0, y_text, font_size, Color::new(0.4, 1.0, 0.6, 1.0));

    // Track & Speed Info
    let speed_text = if state.paused { "PAUSED" } else { &format!("{}x", state.sim_speed) };
    draw_text(&format!("SPEED: {}", speed_text), 920.0, y_text, font_size, Color::new(0.9, 0.6, 1.0, 1.0));
    draw_text(&format!("TRACK: {}", state.track.name), 1040.0, y_text, font_size, Color::new(0.7, 0.8, 0.9, 1.0));

    // 2. Neural Network Visualizer HUD (Bottom-Right)
    if state.show_nn_hud && !state.population.agents.is_empty() {
        let leader_idx = state.population.leader_idx();
        let leader_agent = &state.population.agents[leader_idx];
        draw_neural_network_hud(leader_agent, scr_w - 380.0, scr_h - 280.0, 365.0, 265.0);
    }

    // 3. Fitness Progression Graph (Bottom-Left)
    if state.show_graph && state.population.stats_history.len() >= 2 {
        draw_fitness_graph(&state.population.stats_history, 15.0, scr_h - 180.0, 280.0, 165.0);
    }

    // 4. Toast Notification Overlay
    if let Some((msg, _)) = &state.toast_message {
        let tw = measure_text(msg, None, 20, 1.0).width;
        let px = (scr_w - tw) * 0.5 - 15.0;
        let py = 65.0;
        draw_rectangle(px, py, tw + 30.0, 34.0, Color::new(0.1, 0.12, 0.18, 0.95));
        draw_rectangle_lines(px, py, tw + 30.0, 34.0, 1.5, Color::new(0.3, 0.6, 1.0, 0.9));
        draw_text(msg, px + 15.0, py + 23.0, 20.0, Color::new(0.9, 0.95, 1.0, 1.0));
    }

    // 5. Help Overlay Window (if toggled)
    if state.show_help {
        draw_help_overlay(scr_w, scr_h);
    } else {
        // Subtle hint in bottom left
        draw_text("[H] Controls & Help", 18.0, scr_h - 10.0, 15.0, Color::new(0.5, 0.6, 0.7, 0.8));
    }
}

fn draw_neural_network_hud(agent: &evolution::Agent, x: f32, y: f32, w: f32, h: f32) {
    // Background panel
    draw_rectangle(x, y, w, h, Color::new(0.05, 0.07, 0.10, 0.92));
    draw_rectangle_lines(x, y, w, h, 1.5, Color::new(0.2, 0.4, 0.6, 0.7));

    draw_text("BRAIN TOPOLOGY (LEADER)", x + 12.0, y + 20.0, 15.0, Color::new(0.2, 0.8, 1.0, 0.95));

    let inputs = agent.car.get_network_inputs();
    let (outputs, layer_activations) = agent.brain.forward_with_cache(&inputs);

    // Layout layer columns
    let num_layers = agent.brain.layer_sizes.len();
    let col_spacing = (w - 60.0) / (num_layers - 1) as f32;

    let layer_x = |layer_idx: usize| -> f32 { x + 30.0 + layer_idx as f32 * col_spacing };

    let node_y = |_layer_idx: usize, node_idx: usize, total_nodes: usize| -> f32 {
        let avail_h = h - 60.0;
        let spacing = avail_h / (total_nodes as f32 + 1.0);
        y + 35.0 + (node_idx + 1) as f32 * spacing
    };

    // 1. Draw Synaptic Weights (Lines between layers)
    for l in 0..agent.brain.layers.len() {
        let layer = &agent.brain.layers[l];
        let in_count = layer.in_features;
        let out_count = layer.out_features;

        let x1 = layer_x(l);
        let x2 = layer_x(l + 1);

        for out_idx in 0..out_count {
            let y2 = node_y(l + 1, out_idx, out_count);
            for in_idx in 0..in_count {
                let y1 = node_y(l, in_idx, in_count);
                let weight = layer.weights.get(out_idx, in_idx);

                let alpha = (weight.abs() / 2.0).clamp(0.05, 0.85);
                let line_color = if weight > 0.0 {
                    Color::new(0.2, 0.6, 1.0, alpha) // Positive weight (Blue)
                } else {
                    Color::new(1.0, 0.3, 0.2, alpha) // Negative weight (Red)
                };

                let thickness = (weight.abs() * 1.2).clamp(0.5, 2.5);
                draw_line(x1, y1, x2, y2, thickness, line_color);
            }
        }
    }

    // 2. Draw Neuron Nodes with Activation Colors
    for (l_idx, &layer_size) in agent.brain.layer_sizes.iter().enumerate() {
        let px = layer_x(l_idx);
        let acts = &layer_activations[l_idx];

        for n_idx in 0..layer_size {
            let py = node_y(l_idx, n_idx, layer_size);
            let act = acts.get(n_idx).copied().unwrap_or(0.0);

            // Node Color based on activation
            let node_color = if act > 0.0 {
                let intensity = act.clamp(0.0, 1.0);
                Color::new(0.2 + 0.1 * intensity, 0.4 + 0.6 * intensity, 0.3 + 0.3 * intensity, 1.0)
            } else {
                let intensity = (-act).clamp(0.0, 1.0);
                Color::new(0.4 + 0.6 * intensity, 0.2, 0.2, 1.0)
            };

            draw_circle(px, py, 4.5, node_color);
            draw_circle_lines(px, py, 4.5, 1.0, Color::new(0.9, 0.9, 0.9, 0.8));
        }
    }

    // 3. Draw Output Meters (Steering & Gas)
    let steer = outputs[0];
    let gas = outputs[1];

    let meter_y = y + h - 22.0;
    draw_text(&format!("STR: {:+.2}", steer), x + 15.0, meter_y, 13.0, Color::new(0.9, 0.9, 0.9, 0.9));
    draw_rectangle(x + 95.0, meter_y - 9.0, 60.0, 10.0, Color::new(0.15, 0.2, 0.25, 1.0));
    let steer_w = (steer * 30.0).clamp(-30.0, 30.0);
    draw_rectangle(x + 125.0, meter_y - 9.0, steer_w, 10.0, Color::new(0.2, 0.8, 1.0, 1.0));

    draw_text(&format!("GAS: {:+.2}", gas), x + 185.0, meter_y, 13.0, Color::new(0.9, 0.9, 0.9, 0.9));
    draw_rectangle(x + 260.0, meter_y - 9.0, 60.0, 10.0, Color::new(0.15, 0.2, 0.25, 1.0));
    let gas_w = (gas * 60.0).clamp(0.0, 60.0);
    draw_rectangle(x + 260.0, meter_y - 9.0, gas_w, 10.0, Color::new(0.3, 1.0, 0.4, 1.0));
}

fn draw_fitness_graph(history: &[evolution::GenerationStats], x: f32, y: f32, w: f32, h: f32) {
    draw_rectangle(x, y, w, h, Color::new(0.05, 0.07, 0.10, 0.92));
    draw_rectangle_lines(x, y, w, h, 1.5, Color::new(0.2, 0.4, 0.6, 0.7));

    draw_text("FITNESS EVOLUTION", x + 10.0, y + 18.0, 14.0, Color::new(1.0, 0.85, 0.2, 0.95));

    let max_fit = history
        .iter()
        .map(|s| s.best_fitness)
        .fold(1.0f32, |acc, f| acc.max(f));

    let n = history.len();
    let graph_x = x + 15.0;
    let graph_y = y + 25.0;
    let graph_w = w - 30.0;
    let graph_h = h - 35.0;

    for i in 0..n - 1 {
        let x1 = graph_x + (i as f32 / (n - 1) as f32) * graph_w;
        let x2 = graph_x + ((i + 1) as f32 / (n - 1) as f32) * graph_w;

        // Best fitness line (Gold)
        let y1_best = graph_y + graph_h - (history[i].best_fitness / max_fit) * graph_h;
        let y2_best = graph_y + graph_h - (history[i + 1].best_fitness / max_fit) * graph_h;
        draw_line(x1, y1_best, x2, y2_best, 2.0, Color::new(1.0, 0.85, 0.2, 1.0));

        // Avg fitness line (Cyan)
        let y1_avg = graph_y + graph_h - (history[i].avg_fitness / max_fit) * graph_h;
        let y2_avg = graph_y + graph_h - (history[i + 1].avg_fitness / max_fit) * graph_h;
        draw_line(x1, y1_avg, x2, y2_avg, 1.5, Color::new(0.2, 0.8, 1.0, 0.8));
    }
}

fn draw_help_overlay(scr_w: f32, scr_h: f32) {
    let pw = 520.0;
    let ph = 380.0;
    let px = (scr_w - pw) * 0.5;
    let py = (scr_h - ph) * 0.5;

    draw_rectangle(px, py, pw, ph, Color::new(0.04, 0.06, 0.09, 0.97));
    draw_rectangle_lines(px, py, pw, ph, 2.0, Color::new(0.2, 0.6, 1.0, 0.9));

    draw_text("SIMULATION CONTROLS & SHORTCUTS", px + 25.0, py + 35.0, 20.0, Color::new(0.2, 0.85, 1.0, 1.0));

    let items = [
        ("[Space]", "Pause / Resume simulation"),
        ("[1] - [6]", "Simulation Speed (1x, 2x, 5x, 10x, 25x, 50x)"),
        ("[T]", "Cycle Track Presets (Monaco, Speedway, Hairpin, Figure-8, Procedural)"),
        ("[C]", "Cycle Camera (Follow Best <-> Track Overview <-> Free Pan)"),
        ("[Right Click Drag]", "Free Pan Camera across racetrack"),
        ("[Mouse Wheel]", "Zoom Camera in and out"),
        ("[N]", "Toggle Neural Network HUD Visualizer"),
        ("[G]", "Toggle Fitness Progression History Chart"),
        ("[V] / [D]", "Toggle Sensor Ray Visualization"),
        ("[M]", "Toggle Manual Player Car (Drive with W/A/S/D or Arrows)"),
        ("[K]", "Kill current generation & advance immediately"),
        ("[R]", "Reset entire simulation to Generation 1"),
        ("[S]", "Save Champion Brain to best_car_nn.json"),
        ("[L]", "Load Champion Brain from best_car_nn.json"),
        ("[H]", "Close this Help Dialog"),
    ];

    let mut cur_y = py + 70.0;
    for (key, desc) in items {
        draw_text(key, px + 30.0, cur_y, 16.0, Color::new(1.0, 0.85, 0.2, 1.0));
        draw_text(desc, px + 120.0, cur_y, 16.0, Color::new(0.85, 0.9, 0.95, 0.9));
        cur_y += 22.0;
    }
}
