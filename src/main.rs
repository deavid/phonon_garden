use macroquad::prelude::{
    clear_background, draw_text, get_frame_time, is_mouse_button_pressed, mouse_position,
    next_frame, screen_height, screen_width, Color, Conf, MouseButton, Vec2, BLACK, WHITE,
};
use macroquad::texture::draw_texture;
use ndarray::Array1;

// --- Data Structures ---
// These will define the components of our simulation space.

use ndarray::Array2;

// Represents the entire discrete space: a collection of nodes and their connections.
struct Graph {
    // Using Array2 for efficient row-based access to node positions.
    // Each row is a `[x, y]` coordinate.
    nodes: Array2<f32>,
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
use rand::Rng;

/// Creates a Random Geometric Graph by connecting nodes within a given radius.
/// Uses a k-d tree for efficient neighbor search.
fn create_random_geometric_graph(
    num_nodes: usize,
    radius: f32,
    width: f32,
    height: f32,
) -> (Graph, KdTree<f64, usize, [f64; 2]>) {
    let mut rng = rand::rngs::ThreadRng::default();

    // 1. Create nodes with random positions
    let mut nodes = Array2::<f32>::zeros((num_nodes, 2));
    let mut points = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let x = rng.random_range(0.0..width);
        let y = rng.random_range(0.0..height);
        nodes[[i, 0]] = x;
        nodes[[i, 1]] = y;
        points.push([x as f64, y as f64]);
    }

    // 2. Build the k-d tree for efficient spatial queries
    let mut kdtree = KdTree::new(2);
    for (i, point) in points.iter().enumerate() {
        kdtree.add(*point, i).unwrap();
    }

    // 3. Find neighbors for each node and ensure symmetry
    let mut adj = vec![vec![]; num_nodes];
    let radius_squared = (radius as f64).powi(2);

    for i in 0..num_nodes {
        let neighbors = kdtree
            .within(&points[i], radius_squared, &squared_euclidean)
            .unwrap();

        for &(_, &neighbor_index) in neighbors.iter() {
            if i != neighbor_index {
                // Add edge from i to neighbor_index
                if !adj[i].contains(&neighbor_index) {
                    adj[i].push(neighbor_index);
                }
                // Add edge from neighbor_index to i to ensure symmetry
                if !adj[neighbor_index].contains(&i) {
                    adj[neighbor_index].push(i);
                }
            }
        }
    }

    println!(
        "Graph generated: {} nodes, average degree: {:.2}",
        num_nodes,
        adj.iter().map(|n| n.len()).sum::<usize>() as f32 / num_nodes as f32
    );

    (Graph { nodes, adj }, kdtree)
}

/// Performs a single step of graph relaxation.
fn relax_graph_step(
    graph: &mut Graph,
    kdtree: &KdTree<f64, usize, [f64; 2]>,
    repulsion_strength: f32,
    spring_strength: f32,
    ideal_distance: f32,
    width: f32,
    height: f32,
) {
    let mut forces = Array2::<f32>::zeros((graph.nodes.nrows(), 2));
    let containment_strength = 0.01;

    for i in 0..graph.nodes.nrows() {
        // Repulsion from nearby nodes
        let pos_i = Vec2::new(graph.nodes[[i, 0]], graph.nodes[[i, 1]]);
        let pos_f64 = [pos_i.x as f64, pos_i.y as f64];
        let neighbors = kdtree
            .within(&pos_f64, ideal_distance.powi(2) as f64, &squared_euclidean)
            .unwrap();

        for &(_, &j) in neighbors.iter() {
            if i == j {
                continue;
            }
            let pos_j = Vec2::new(graph.nodes[[j, 0]], graph.nodes[[j, 1]]);
            let d = pos_i - pos_j;
            let dist_sq = d.length_squared();
            if dist_sq > 1e-6 {
                let force = d / dist_sq;
                forces[[i, 0]] += force.x * repulsion_strength;
                forces[[i, 1]] += force.y * repulsion_strength;
            }
        }

        // Spring force with connected neighbors
        for &neighbor_index in &graph.adj[i] {
            let pos_neighbor = Vec2::new(
                graph.nodes[[neighbor_index, 0]],
                graph.nodes[[neighbor_index, 1]],
            );
            let d = pos_neighbor - pos_i;
            let dist = d.length();
            let displacement = dist - ideal_distance;
            let force = d.normalize_or_zero() * displacement * spring_strength;
            forces[[i, 0]] += force.x;
            forces[[i, 1]] += force.y;
        }

        // Containment force from window boundaries
        let margin = 50.0;
        if pos_i.x < margin {
            forces[[i, 0]] += (margin - pos_i.x) * containment_strength;
        }
        if pos_i.x > width - margin {
            forces[[i, 0]] -= (pos_i.x - (width - margin)) * containment_strength;
        }
        if pos_i.y < margin {
            forces[[i, 1]] += (margin - pos_i.y) * containment_strength;
        }
        if pos_i.y > height - margin {
            forces[[i, 1]] -= (pos_i.y - (height - margin)) * containment_strength;
        }
    }

    graph.nodes += &forces;
}

/// Runs one time-step of the wave simulation.
fn run_simulation_step(graph: &Graph, state: &mut SimState, c: f32, dt: f32, damping: f32) {
    let mut acceleration = Array1::<f32>::zeros(graph.nodes.nrows());

    // First pass: calculate accelerations
    for i in 0..graph.nodes.nrows() {
        let u_i = state.u[i];
        let mut laplacian = 0.0;
        for &neighbor_index in &graph.adj[i] {
            laplacian += state.u[neighbor_index] - u_i;
        }
        acceleration[i] = c * c * laplacian;
    }

    // Second pass: update velocities and displacements
    for i in 0..graph.nodes.nrows() {
        state.v[i] += acceleration[i] * dt;
        state.v[i] *= 1.0 - damping; // Apply damping
        state.u[i] += state.v[i] * dt;
    }
}

use macroquad::texture::{Image, Texture2D};

/// Maps the simulation state (u, v) to a color for rendering.
fn get_color_for_node(u: f32, v: f32) -> Color {
    let r = (0.5 + u).clamp(0.0, 1.0);
    let g = (0.5 + v).clamp(0.0, 1.0);
    let b = (0.5 - u).clamp(0.0, 1.0);
    Color::new(r, g, b, 1.0)
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
    const IDEAL_SPRING_LENGTH_MULTIPLIER: f32 = 1.0;

    // --- Setup Phase ---
    let (width, height) = (screen_width(), screen_height());
    let (mut graph, kdtree) =
        create_random_geometric_graph(NUM_NODES, CONNECTION_RADIUS, width, height);

    let mut phase = AppPhase::Relaxing;
    let mut relaxation_step = 0;

    let mut state = SimState {
        u: Array1::zeros(NUM_NODES),
        v: Array1::zeros(NUM_NODES),
    };

    let mut image = Image::gen_image_color(width as u16, height as u16, BLACK);
    let texture = Texture2D::from_image(&image);

    // --- Main Loop ---
    loop {
        // --- Input Handling ---
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let mouse_pos_f64 = [mx as f64, my as f64];
            if let Ok(neighbors) = kdtree.nearest(&mouse_pos_f64, 1, &squared_euclidean) {
                if let Some(&(_, &nearest_node_index)) = neighbors.first() {
                    state.u[nearest_node_index] = 1.0;
                    println!("Plucked node {}", nearest_node_index);
                }
            }
        }

        // --- Phase-specific Logic ---
        match phase {
            AppPhase::Relaxing => {
                if relaxation_step < RELAXATION_ITERATIONS {
                    relax_graph_step(
                        &mut graph,
                        &kdtree,
                        REPULSION_STRENGTH,
                        0.01, // spring_strength
                        CONNECTION_RADIUS * IDEAL_SPRING_LENGTH_MULTIPLIER,
                        width,
                        height,
                    );
                    relaxation_step += 1;
                } else {
                    phase = AppPhase::Simulating;
                }
            }
            AppPhase::Simulating => {
                let dt = get_frame_time();
                run_simulation_step(&graph, &mut state, WAVE_SPEED, dt, DAMPING);
            }
        }

        // --- Drawing ---
        // Clear the image buffer
        image.bytes.chunks_exact_mut(4).for_each(|pixel| {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            pixel[3] = 255;
        });

        // Draw nodes to the image buffer
        for i in 0..graph.nodes.nrows() {
            let x = graph.nodes[[i, 0]] as u32;
            let y = graph.nodes[[i, 1]] as u32;
            if x < width as u32 && y < height as u32 {
                let color = get_color_for_node(state.u[i], state.v[i]);
                image.set_pixel(x, y, color);
            }
        }
        texture.update(&image);

        clear_background(BLACK);
        draw_texture(&texture, 0.0, 0.0, WHITE);

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
