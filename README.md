# Phonon Garden

Phonon Garden is a real-time, interactive physics simulation that explores the emergent properties of wave propagation on a large-scale, discrete 2D space. The project aims to create a visually striking "digital petri dish" where complex wave phenomena like reflection, refraction, and interference arise naturally from a simple set of underlying rules.

## The Core Concept

At its heart, Phonon Garden simulates a 2D "drum head" or "aether" made not of a continuous surface, but of millions of interconnected points (nodes). When this medium is "plucked" by a user's click, a wave disturbance is created. The simulation calculates how this wave evolves over time, spreading through the random network of nodes.

The primary goal is to build a system where **isotropy** (the property of being uniform in all directions) emerges from a locally **anisotropic** (directionally-biased) and random substrate. In other words, we want to see perfect circular waves expanding from a chaotic mesh of points, demonstrating how a continuous, Euclidean-like space can be approximated by a discrete graph.

## Technical Implementation

The simulation is built in Rust for high performance, memory safety, and the ability to leverage modern multi-core CPUs.

### 1. The Space: A Random Geometric Graph (RGG)

The discrete space is modeled as a **Random Geometric Graph**.
- **Generation:** A large number of nodes (e.g., 10,000 to 1,000,000+) are placed at random positions within a 2D plane.
- **Connectivity:** Two nodes are connected with an edge if and only if their Euclidean distance is less than a specified connection radius `R`.
- **Properties:** This method creates a graph with a "natural" structure. The number of connections per node fluctuates based on local density, which is crucial for achieving statistical isotropy and avoiding the artifacts common in regular grid-based simulations.

### 2. The Physics: A Discretized Wave Equation

To model wave behavior correctly (as opposed to simple diffusion), the simulation must account for inertia. This is achieved by tracking two state variables for each node:
- **`u` (Displacement):** The scalar value representing the node's "height" or displacement from its resting state.
- **`v` (Velocity):** The rate of change of the displacement (`du/dt`).

The simulation evolves in discrete time steps (`Δt`) using a second-order numerical integration scheme. The core of the physics engine is the **Graph Laplacian**, which serves as a discrete analog to the second spatial derivative (`∇²u`).

- **Force Calculation:** The "restoring force" on a node is proportional to its Laplacian value, which measures how much its displacement deviates from the average of its neighbors. `Acceleration ∝ Laplacian(u)`.
- **Update Rule:** At each time step, the acceleration is used to update the velocity, and the new velocity is used to update the displacement.

### 3. Key Features

- **High-Performance Simulation:** Written in Rust and leveraging the `ndarray` crate for efficient numerical operations and `rayon` for multi-threading the core simulation loop, allowing for a massive number of nodes in real-time.
- **Interactive "Plucking":** Users can click anywhere on the screen to introduce a new wave into the system, creating a dynamic and engaging experience.
- **Emergent Isotropy:** The primary visual goal. Despite the random, irregular connections at the micro-level, waves will be observed to propagate in near-perfect circles at the macro-level.
- **Advanced Visualization:** The state of each node is mapped to a unique color based on *both* its displacement (`u`) and its velocity (`v`). This creates a rich, informative, and visually stunning representation of the wave's phase, showing not just its height but also its direction of travel.
- **Tunable Parameters:** The simulation will allow for real-time or configuration-based tuning of key physical constants:
    - **Connection Radius `R`:** To control the density and connectivity of the graph.
    - **Wave Speed `c`:** To change the propagation speed.
    - **Damping:** A small, configurable damping factor is included to allow waves to naturally fade over time, creating a more realistic and visually clean experience. It can be set to zero to simulate a perfect, energy-conserving system.
- **Graph Relaxation (Pre-computation):** An initial "relaxation" step adjusts node positions to create a more uniform, "blue noise" distribution, which improves the visual quality of the simulation.

## Project Vision

Phonon Garden aims to be more than just a technical demo. It is an artistic exploration of the beauty of emergent complexity. By visualizing the fundamental mathematics of waves in a novel way, it seeks to create a mesmerizing and contemplative digital artwork.