### **Phonon Garden: Project Requirements, Constraints, and Restrictions**

**Document Purpose:** This document serves as the definitive specification for the `phonon_garden` project. All development must adhere strictly to these guidelines. Deviations are not permitted without explicit new instructions. The goal is to prevent ambiguity and ensure the final product matches the intended design precisely.

---

### **Part 1: Core Functional Requirements (What It MUST Do)**

1.  **Simulation Space:**
    *   **FR1.1:** The simulation space MUST be a **Random Geometric Graph (RGG)**.
    *   **FR1.2:** The graph generation process MUST be: 1) Place N nodes at random 2D positions. 2) Connect any two nodes if their Euclidean distance is less than a configurable radius `R`.
    *   **FR1.3:** The number of connections per node MUST be an emergent property of the RGG process. It is NOT to be a fixed number (i.e., do not implement a k-Nearest Neighbors or k-Regular graph).

2.  **Physics Model:**
    *   **FR2.1:** The simulation MUST model a **second-order wave equation**, not a first-order diffusion equation.
    *   **FR2.2:** To achieve this, each node MUST have two state variables: `u` (displacement) and `v` (velocity).
    *   **FR2.3:** The update logic MUST be based on the **Graph Laplacian** (`Laplacian(u_i) = Σ(u_j - u_i)`). The acceleration of a node must be proportional to this value.
    *   **FR2.4:** A configurable `damping` parameter MUST be included in the velocity update step. It must be possible to set this to `0.0` to achieve a perfectly energy-conserving system.

3.  **Interaction and Visualization:**
    *   **FR3.1:** The application MUST open a graphical window and render the simulation in real-time.
    *   **FR3.2:** A mouse click MUST introduce a disturbance into the simulation by setting the `u` value of the nearest node to a positive constant (e.g., `1.0`).
    *   **FR3.3:** The color of each node on screen MUST be determined by a function that takes *both* its `u` and `v` values as input. The mapping must follow this specific scheme:
        *   Neutral State (`u=0, v=0`): A neutral grey.
        *   Positive `u`: Contributes a yellow/magenta component.
        *   Negative `u`: Contributes a magenta/yellow component.
        *   Positive `v`: Contributes a red/cyan component.
        *   Negative `v`: Contributes a cyan/red component.
    *   **FR3.4:** The rendering method MUST be high-performance, suitable for 100,000+ nodes. This implies rendering to an off-screen image or texture buffer that is then drawn to the screen once per frame, rather than using individual draw calls per node.

---

### **Part 2: Technical Constraints (HOW It Must Be Built)**

1.  **Language and Ecosystem:**
    *   **TC1.1:** The project MUST be written in **Rust** (latest stable edition, e.g., 2021).
    *   **TC1.2:** The project MUST be managed by **Cargo**. All dependencies must be explicitly listed in `Cargo.toml`.

2.  **Core Libraries:**
    *   **TC2.1:** Graphics and windowing MUST be handled by **`macroquad`**. Do not use any other graphics library like `winit`, `minifb`, `pixels`, or `bevy`.
    *   **TC2.2:** All primary simulation state arrays (`u`, `v`, node positions) MUST use the **`ndarray`** crate for high-performance numerical computation. Standard `Vec<T>` should only be used where `ndarray` is not appropriate (e.g., the adjacency list).
    *   **TC2.3:** The graph generation's neighbor search MUST be accelerated using the **`kdtree`** crate (or a similar, high-performance spatial index crate) to avoid O(N²) complexity.
    *   **TC2.4:** The main simulation loop MUST be parallelized using the **`rayon`** crate to leverage multi-core CPUs.

---

### **Part 3: Explicit Restrictions (What It MUST NOT Do)**

1.  **Physics and Simulation:**
    *   **R1.1:** DO NOT use a regular grid (e.g., a 2D array) for the simulation space. The graph **must** be irregular and random.
    *   **R1.2:** DO NOT implement a diffusion/heat-flow model (a first-order equation). The presence of both `u` and `v` state variables is non-negotiable.
    *   **R1.3:** The simulation logic MUST NOT physically move the nodes. The graph's structure and node positions are static after an optional initial relaxation phase. The wave is a change in node *state*, not position.

2.  **Implementation:**
    *   **R2.1:** DO NOT use `unsafe` Rust code unless it is absolutely unavoidable and comes from a trusted, well-established crate.
    *   **R2.2:** DO NOT hard-code values that should be configurable. Key parameters like `NUM_NODES`, `CONNECTION_RADIUS`, `WAVE_SPEED`, and `DAMPING` should be defined as `const` or configurable variables at the top of the file for easy tuning.
    *   **R2.3:** DO NOT implement a naive O(N²) neighbor search for graph generation. The use of a spatial index is a firm requirement.
    *   **R2.4:** DO NOT use individual `draw_circle` or `draw_pixel` calls in a loop for the final rendering. The high-performance texture-based approach is required.
