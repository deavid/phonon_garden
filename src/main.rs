use macroquad::prelude::{
    clear_background, draw_circle, draw_text, is_mouse_button_pressed, mouse_position, next_frame,
    screen_height, screen_width, Color, Conf, MouseButton, Vec2, BLACK, GRAY, WHITE,
};
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

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use rand::{thread_rng, Rng};

/// Creates a Random Geometric Graph by connecting nodes within a given radius.
/// Uses a k-d tree for efficient neighbor search.
fn create_random_geometric_graph(num_nodes: usize, radius: f32, width: f32, height: f32) -> Graph {
    let mut rng = thread_rng();

    // 1. Create nodes with random positions
    let mut nodes = Vec::with_capacity(num_nodes);
    let mut points = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let x = rng.gen_range(0.0..width);
        let y = rng.gen_range(0.0..height);
        nodes.push(Node {
            pos: Vec2::new(x, y),
        });
        points.push([x as f64, y as f64]);
    }

    // 2. Build the k-d tree for efficient spatial queries
    let mut kdtree = KdTree::new(2);
    for (i, point) in points.iter().enumerate() {
        kdtree.add(*point, i).unwrap();
    }

    // 3. Find neighbors for each node
    let mut adj = vec![vec![]; num_nodes];
    let radius_squared = (radius as f64).powi(2);

    for (i, point) in points.iter().enumerate() {
        let neighbors = kdtree
            .within(point, radius_squared, &squared_euclidean)
            .unwrap();

        for &(_, &neighbor_index) in neighbors.iter() {
            if i != neighbor_index {
                adj[i].push(neighbor_index);
            }
        }
    }

    println!(
        "Graph generated: {} nodes, average degree: {:.2}",
        num_nodes,
        adj.iter().map(|n| n.len()).sum::<usize>() as f32 / num_nodes as f32
    );

    Graph { nodes, adj }
}

/// Performs a single step of graph relaxation.
fn relax_graph_step(
    graph: &mut Graph,
    repulsion_strength: f32,
    spring_strength: f32,
    ideal_distance: f32,
    width: f32,
    height: f32,
) {
    let mut forces = vec![Vec2::ZERO; graph.nodes.len()];
    let containment_strength = 0.01;

    for i in 0..graph.nodes.len() {
        // Repulsion from all other nodes
        for j in 0..graph.nodes.len() {
            if i == j {
                continue;
            }
            let d = graph.nodes[i].pos - graph.nodes[j].pos;
            let dist_sq = d.length_squared();
            if dist_sq > 1e-6 {
                let force = d / dist_sq;
                forces[i] += force * repulsion_strength;
            }
        }

        // Spring force with connected neighbors
        for &neighbor_index in &graph.adj[i] {
            let d = graph.nodes[neighbor_index].pos - graph.nodes[i].pos;
            let dist = d.length();
            let displacement = dist - ideal_distance;
            let force = d.normalize_or_zero() * displacement * spring_strength;
            forces[i] += force;
        }


        // Containment force from window boundaries
        let pos = graph.nodes[i].pos;
        forces[i].x -= (pos.x - width * 0.5) * containment_strength;
        forces[i].y -= (pos.y - height * 0.5) * containment_strength;
    }

    for i in 0..graph.nodes.len() {
        graph.nodes[i].pos += forces[i];
    }
}

/// TODO: Implement the simulation logic.
/// Runs one time-step of the wave simulation.
fn run_simulation_step(_graph: &Graph, _state: &mut SimState, _c: f32, _dt: f32, _damping: f32) {
    // Placeholder implementation
    // This is where the Laplacian calculation and state updates will happen.
}

/// TODO: Implement the visualization logic.
/// Maps the simulation state (u, v) to a color for rendering.
fn get_color_for_node(_u: f32, _v: f32) -> Color {
    // Placeholder: just a simple grey for now.
    // The final version will have the complex u,v -> color mapping.
    GRAY
}

enum AppPhase {
    Relaxing,
    Simulating,
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
    const RELAXATION_ITERATIONS: usize = 100;
    const REPULSION_STRENGTH: f32 = 0.1;

    // --- Setup Phase ---
    let (width, height) = (screen_width(), screen_height());
    let mut graph = create_random_geometric_graph(NUM_NODES, CONNECTION_RADIUS, width, height);

    let mut phase = AppPhase::Relaxing;
    let mut relaxation_step = 0;

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

        // --- Phase-specific Logic ---
        match phase {
            AppPhase::Relaxing => {
                if relaxation_step < RELAXATION_ITERATIONS {
                    relax_graph_step(
                        &mut graph,
                        REPULSION_STRENGTH,
                        0.01, // spring_strength
                        CONNECTION_RADIUS,
                        width,
                        height,
                    );
                    relaxation_step += 1;
                } else {
                    phase = AppPhase::Simulating;
                }
            }
            AppPhase::Simulating => {
                // let dt = get_frame_time();
                // run_simulation_step(&graph, &mut state, WAVE_SPEED, dt, DAMPING);
            }
        }

        // --- Drawing ---
        clear_background(BLACK);

        // Draw nodes
        for node in &graph.nodes {
            draw_circle(node.pos.x, node.pos.y, 1.5, WHITE);
        }

        // Draw phase-specific text
        match phase {
            AppPhase::Relaxing => {
                let progress = (relaxation_step as f32 / RELAXATION_ITERATIONS as f32) * 100.0;
                let text = format!("Relaxing: {:.0}%", progress);
                draw_text(&text, 20.0, 20.0, 30.0, WHITE);
            }
            AppPhase::Simulating => {
                draw_text("Simulating", 20.0, 20.0, 30.0, WHITE);
            }
        }

        next_frame().await
    }
}
