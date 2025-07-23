use macroquad::prelude::*;
use ndarray::Array1;

// --- Data Structures ---
// These will define the components of our simulation space.

// Represents a single point in our 2D space.
struct Node {
    pos: Vec2,
}

// Represents the entire discrete space: a collection of nodes and their connections.
struct Graph {
    nodes: Vec<Node>,
    // Adjacency list: for each node, a list of indices of its neighbors.
    adj: Vec<Vec<usize>>,
}

// Holds the dynamic state of the wave simulation.
struct SimState {
    // Displacement ('u') for each node.
    u: Array1<f32>,
    // Velocity ('v') for each node.
    v: Array1<f32>,
}

// --- Core Functions ---
// These are the placeholders for the main logic that the agent will implement.

/// TODO: Implement graph generation.
/// Creates a Random Geometric Graph by connecting nodes within a given radius.
/// Should use the kdtree crate for efficiency.
fn create_random_geometric_graph(num_nodes: usize, radius: f32, width: f32, height: f32) -> Graph {
    // Placeholder implementation
    println!("TODO: Implement create_random_geometric_graph");
    Graph {
        nodes: vec![],
        adj: vec![],
    }
}

/// TODO: Implement the simulation logic.
/// Runs one time-step of the wave simulation.
fn run_simulation_step(graph: &Graph, state: &mut SimState, c: f32, dt: f32, damping: f32) {
    // Placeholder implementation
    // This is where the Laplacian calculation and state updates will happen.
}

/// TODO: Implement the visualization logic.
/// Maps the simulation state (u, v) to a color for rendering.
fn get_color_for_node(u: f32, v: f32) -> Color {
    // Placeholder: just a simple grey for now.
    // The final version will have the complex u,v -> color mapping.
    GRAY
}

// --- Main Application Loop ---

fn window_conf() -> Conf {
    Conf {
        window_title: "Phonon Garden".to_string(),
        window_width: 1000,
        window_height: 1000,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    println!("Welcome to the Phonon Garden. Initializing...");

    // --- Configuration ---
    const NUM_NODES: usize = 10_000; // Start with a smaller number for faster prototyping
    const CONNECTION_RADIUS: f32 = 15.0;
    const WAVE_SPEED: f32 = 1.0;
    const DAMPING: f32 = 0.005;

    // --- Setup Phase ---
    // This is where the agent will call the generation and setup functions.
    // let graph = create_random_geometric_graph(NUM_NODES, ...);
    // let mut state = SimState { u: ..., v: ... };
    println!("TODO: Call graph generation and initialize simulation state.");

    // --- Main Loop ---
    loop {
        // --- Input Handling ---
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            println!(
                "Mouse clicked at ({}, {}). TODO: Implement wave pluck.",
                mx, my
            );
        }

        // --- Simulation Step ---
        let dt = get_frame_time();
        // TODO: Call run_simulation_step(&graph, &mut state, ...);

        // --- Drawing ---
        clear_background(BLACK);

        draw_text("Phonon Garden - Initial Setup", 20.0, 20.0, 30.0, WHITE);
        draw_text(
            "Next step: Implement graph generation and simulation logic.",
            20.0,
            50.0,
            20.0,
            LIGHTGRAY,
        );

        // TODO: Loop through all nodes and draw them using their state color.
        // for i in 0..graph.nodes.len() {
        //     let node = &graph.nodes[i];
        //     let color = get_color_for_node(state.u[i], state.v[i]);
        //     draw_circle(node.pos.x, node.pos.y, 1.0, color);
        // }

        next_frame().await
    }
}
