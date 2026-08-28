# 🏎️ NeuroRacer: Autonomous 2D Neural Network & Genetic Evolution Racing Engine

A complete, high-performance, modular 2D Top-Down Car Racing Simulation with a **Custom Deep Neural Network** and **Evolutionary/Genetic Algorithm** written **entirely from scratch in Rust** with zero external ML, matrix, or physics libraries.

---

## 🌟 Key Architecture & Highlights

```
NeuronalesNetzwerk/
├── Cargo.toml                     # Optimized release profile (LTO, codegen-units=1)
├── src/
│   ├── lib.rs                     # Re-exports core modules for library & test usage
│   ├── math.rs                    # 2D Vector Math, Raycasting, Line Intersections, Matrix Linear Algebra
│   ├── nn.rs                      # Deep Feedforward Neural Net, Activations, Xavier/He Init, Mutation, Crossover
│   ├── car.rs                     # 2D Vehicle Physics, Lateral Grip/Drift, Raycast Sensors, Fitness Tracking
│   ├── track.rs                   # Spline-Interpolated Tracks, Boundaries, Reward Checkpoint Gates, 5 Presets
│   ├── evolution.rs               # Population Management, Tournament/Roulette Selection, Elitism, Historical Stats
│   └── main.rs                    # Macroquad 60+ FPS Renderer, Live Neural HUD, Fitness Charts, Camera & Controls
└── tests/
    └── evolution_convergence.rs   # Automated Genetic Evolution Convergence Test
```

---

## 🔬 Mathematical Foundations & Algorithms

### 1. Vector & Raycast Geometry (`src/math.rs`)
- **Ray-Line Segment Intersection**:
  Given a ray $R(t) = \mathbf{O} + t \mathbf{D}$ ($t \in [0, d_{\max}]$) and track barrier segment $S(u) = \mathbf{A} + u (\mathbf{B} - \mathbf{A})$ ($u \in [0, 1]$):
  $$\mathbf{O} + t \mathbf{D} = \mathbf{A} + u (\mathbf{B} - \mathbf{A})$$
  Solving using the 2D cross product (wedge product $\mathbf{a} \times \mathbf{b} = a_x b_y - a_y b_x$):
  $$t = \frac{(\mathbf{A} - \mathbf{O}) \times (\mathbf{B} - \mathbf{A})}{\mathbf{D} \times (\mathbf{B} - \mathbf{A})}, \quad u = \frac{(\mathbf{A} - \mathbf{O}) \times \mathbf{D}}{\mathbf{D} \times (\mathbf{B} - \mathbf{A})}$$
  A valid collision hit occurs when $0 \le t \le d_{\max}$ and $0 \le u \le 1$.

- **Gaussian Random Sampling via Box-Muller Transform**:
  $$Z = \sqrt{-2 \ln U_1} \cos(2\pi U_2), \quad U_1, U_2 \sim \mathcal{U}(0, 1)$$

### 1. Realistic Vehicle Dynamics (`src/car.rs`)
- **Non-Linear Bicycle Dynamics Model**:
  - **Slip Angles**:
    $$\alpha_f = \arctan\left(\frac{v + a \omega}{u}\right) - \delta, \quad \alpha_r = \arctan\left(\frac{v - b \omega}{u}\right)$$
  - **Dynamic Weight Transfer**:
    $$F_{z,f} = W_f - \frac{h_{\text{cg}}}{L} m a_x, \quad F_{z,r} = W_r + \frac{h_{\text{cg}}}{L} m a_x$$
  - **Pacejka/Tanh Non-Linear Lateral Tire Forces**:
    $$F_{y,f} = - \mu F_{z,f} \cdot \tanh\left(\frac{C_f \alpha_f}{\mu F_{z,f}}\right), \quad F_{y,r} = - \mu F_{z,r} \cdot \tanh\left(\frac{C_r \alpha_r}{\mu F_{z,r}}\right)$$
  - **Yaw Torque & Accelerations**:
    $$\tau_z = a (F_{y,f} \cos\delta + F_{x,f} \sin\delta) - b F_{y,r}, \quad \dot{\omega} = \frac{\tau_z}{I_z}$$
  - **Skid Marks**: Emits persistent asphalt tire rubber trails during hard drifts or heavy braking!
  - **Physically Steered Front Wheels**: Rendered with exact steering angle $\delta$ on the vehicle chassis.

### 2. Deep Multi-Layer Neural Network (`src/nn.rs`)
- **6-Layer Deep Architecture (`[11 -> 32 -> 24 -> 16 -> 12 -> 2]`)**:
  - **11 Inputs**: 7 Raycast distances, Longitudinal Speed $u$, Lateral Slip $v$, Yaw Rate $\omega$, Steer Angle $\delta$.
  - **Hidden Layer 1 (32 Neurons, LeakyReLU)**: Spatial environment & wall perception.
  - **Hidden Layer 2 (24 Neurons, ReLU)**: Apex identification & corner curvature recognition.
  - **Hidden Layer 3 (16 Neurons, Tanh)**: Vehicle slip limit & lateral grip modeling.
  - **Hidden Layer 4 (12 Neurons, Tanh)**: High-level racing line strategy & control blending.
  - **Output Layer (2 Neurons, Tanh)**: Continuous Steering $[-1, 1]$ and Gas/Brake $[-1, 1]$.
- **Dynamic Neural Visualizer HUD**: Dynamically scales and renders any number of deep layers with weight-coded synaptic lines and live neuron firing rates.

### 3. Vehicle Dynamics & Sensor Array (`src/car.rs`)
- **Arcade/Semi-Realistic Physics**:
  - Forward engine acceleration and dynamic braking with reverse gear.
  - Speed-dependent turning response.
  - Lateral tire friction damping: preserves forward momentum while realistically penalizing uncontrolled lateral slide.
  - Air drag proportional to $v^2$ and rolling resistance.
- **Sensor Array**:
  - 7 Raycast sensors radiating at $[-75^\circ, -45^\circ, -20^\circ, 0^\circ, +20^\circ, +45^\circ, +75^\circ]$.
  - Normalized distance readings $[0.0, 1.0]$.
- **Oriented Bounding Box (OBB) Collision**:
  - 4-corner chassis representation tested against all inner and outer track walls.

### 4. Genetic Evolution Engine (`src/evolution.rs`)
- **Multi-Factor Fitness Function**:
  $$\text{Fitness} = (\text{Checkpoints Hit} \times 1000) + (\text{Laps Completed} \times 20000) + (\text{Segment Progress} \times 500) + (\text{Average Speed} \times 1.5)$$
- **Genetic Operators**:
  - **Elitism**: Top $N$ champions copied untouched into the next generation.
  - **Selection**: Tournament Selection ($k=4$), Roulette Wheel, or Rank-based selection.
  - **Crossover**: Blended uniform arithmetic crossover between parent gene weights.
  - **Mutation**: Gaussian perturbation $\mathcal{N}(0, \sigma^2)$ applied per gene.

---

## 🎮 Interactive Simulation Controls

| Key / Input | Action |
|---|---|
| **`[Space]`** | Pause / Resume simulation |
| **`[1]` – `[6]`** | Adjust Simulation Speed ($1\times, 2\times, 5\times, 10\times, 25\times, 50\times$) |
| **`[T]`** | Cycle Racetrack Presets (Grand Prix, Super Speedway, Hairpin & Chicane, Figure-8, Procedural) |
| **`[C]`** | Cycle Camera Mode (Follow Leading Car $\leftrightarrow$ Track Overview $\leftrightarrow$ Free Pan) |
| **`[Right Click Drag]`** | Free Pan Camera across the track |
| **`[Mouse Scroll]`** | Zoom Camera in / out |
| **`[N]`** | Toggle Live Neural Network Brain HUD (weights & activations) |
| **`[G]`** | Toggle Fitness Evolution History Graph |
| **`[V]` / `[D]`** | Toggle Sensor Ray visualization |
| **`[M]`** | Toggle Manual Player Car (Drive with **W/A/S/D** or **Arrow Keys** alongside AI) |
| **`[K]`** | Kill current generation and trigger next epoch immediately |
| **`[R]`** | Reset simulation to Generation 1 |
| **`[S]`** | Save Champion Brain to `best_car_nn.json` |
| **`[L]`** | Load and inject Champion Brain from `best_car_nn.json` |
| **`[H]`** | Toggle On-Screen Controls & Help Overlay |

---

## 🚀 Building & Running

### Run in Development Mode:
```bash
cargo run
```

### Run in High-Performance Release Mode (Recommended):
```bash
cargo run --release
```

### Execute All Mathematical & Evolutionary Unit/Integration Tests:
```bash
cargo test
```
