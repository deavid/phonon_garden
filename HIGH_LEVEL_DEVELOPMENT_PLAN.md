### **Phonon Garden: High-Level Development Plan**

**Objective:** To incrementally build the `phonon_garden` simulation, ensuring each stage results in a functional and verifiable state.

---

### **Milestone 1: Static Graph Generation and Visualization**

**Goal:** Create and display the static "drum head" of our simulation. The world should exist, but it won't do anything yet.

*   **Step 1.1: Implement Graph Generation.**
    *   **Task:** Flesh out the `create_random_geometric_graph` function.
    *   **Details:**
        *   Accept `num_nodes`, `width`, and `height` to create `num_nodes` with random `(x, y)` positions.
        *   Use the `kdtree` crate to build a spatial index of these nodes.
        *   For each node, use the k-d tree's `within` method to efficiently find all neighbors within the `radius`.
        *   Populate the `Graph` struct with the `nodes` and the `adj` (adjacency list).
    *   **Verification:** The function should return a `Graph` struct. We can print the number of nodes and the average degree to the console to verify it's working as expected.

*   **Step 1.2: Basic Node Visualization.**
    *   **Task:** In the `main` loop, draw the generated graph.
    *   **Details:**
        *   Call `create_random_geometric_graph` once during setup.
        *   In the main loop, iterate through `graph.nodes`.
        *   For each node, draw a small circle or a single pixel at its position using a static color (e.g., `WHITE`).
    *   **Verification:** Running the application should display a field of static white dots representing the nodes.

*   **Step 1.3: (Optional but Recommended) Implement Graph Relaxation.**
    *   **Task:** Create a new function `relax_graph_positions` that runs after generation but before the main loop.
    *   **Details:**
        *   Implement the spring/repulsion force logic we discussed.
        *   Run this simulation for a fixed number of iterations (e.g., 100).
        *   This function will modify the `pos` field of the nodes in the `Graph` struct.
    *   **Verification:** The visual output from Step 1.2 should now appear much more uniform and less "clumpy" than before.

---

### **Milestone 2: Implementing the Core Wave Simulation**

**Goal:** Make the waves propagate. The visualization will still be basic, but the underlying physics will be functional.

*   **Step 2.1: Initialize Simulation State.**
    *   **Task:** After creating the graph, initialize the `SimState` struct.
    *   **Details:**
        *   The `u` and `v` arrays in `SimState` should be created using `Array1::zeros(num_nodes)`.
    *   **Verification:** The simulation state exists and is correctly sized.

*   **Step 2.2: Implement the Simulation Step.**
    *   **Task:** Flesh out the `run_simulation_step` function.
    *   **Details:**
        *   Implement the two-pass update:
            1.  **First Pass:** Create a temporary `accelerations` `ndarray`. Loop through all nodes, calculate the `Laplacian(u_i)` using the `graph.adj` list, and store the resulting acceleration.
            2.  **Second Pass:** Loop through all nodes again. Update `v_i` using the stored acceleration, `dt`, and the `damping` factor. Then update `u_i` using the new `v_i`.
    *   **Verification:** This is the hardest part to verify visually yet, but the code should be logically complete.

*   **Step 2.3: Implement Mouse Interaction ("Plucking").**
    *   **Task:** In the `main` loop's input handling block, find the closest node to the mouse click and modify its state.
    *   **Details:**
        *   On click, iterate through `graph.nodes` to find the node with the minimum distance to the mouse coordinates. (A k-d tree could also be used here for efficiency if needed).
        *   Set the `u` value of that node in `state.u` to `1.0`.
    *   **Verification:** Clicking should successfully modify the `u` array. We can verify with a `println!`.

---

### **Milestone 3: Advanced Visualization and Final Touches**

**Goal:** Bring the simulation to life with dynamic, informative coloring and make it feel like a finished product.

*   **Step 3.1: Implement `u,v` to Color Mapping.**
    *   **Task:** Flesh out the `get_color_for_node` function.
    *   **Details:**
        *   Implement the logic we designed: `(u, v)` -> `(r, g, b)`. A neutral grey base, with additive color components for positive/negative `u` and `v`.
        *   Remember to clamp the final color values to the valid `[0.0, 1.0]` range.

*   **Step 3.2: High-Performance Rendering.**
    *   **Task:** Refactor the drawing code to be more efficient.
    *   **Details:**
        *   Instead of calling `draw_circle` thousands of times, create a `macroquad::Image`.
        *   In the main loop, iterate through the nodes and set the corresponding pixel in the image's byte buffer using the color from `get_color_for_node`.
        *   Create a `Texture2D` from this image and draw it once to the screen. This is vastly faster.
    *   **Verification:** The simulation now runs smoothly and displays beautiful, colorful wave patterns. The waves are clearly visible.

*   **Step 3.3: Parallelize the Simulation Logic.**
    *   **Task:** Integrate the `rayon` crate to speed up the simulation step.
    *   **Details:**
        *   The first pass of the simulation loop (calculating accelerations) is a prime candidate for parallelization.
        *   Convert the loop over nodes (`for i in 0..num_nodes`) to a parallel iterator using Rayon (e.g., `(0..num_nodes).into_par_iter().for_each(...)`).
    *   **Verification:** The simulation should handle a much larger number of nodes while maintaining a high frame rate. CPU usage should be distributed across multiple cores.
