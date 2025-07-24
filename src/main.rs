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

    // 3. Find potential neighbors and connect with distance-dependent probability
    let mut adj = vec![vec![]; num_nodes];
    let radius_squared = (radius as f64).powi(2);

    for i in 0..num_nodes {
        let neighbors = kdtree
            .within(&points[i], radius_squared, &squared_euclidean)
            .unwrap();

        for &(distance_sq, &neighbor_index) in neighbors.iter() {
            if i != neighbor_index {
                // Calculate distance-dependent connection probability
                let distance = (distance_sq as f32).sqrt();
                let normalized_distance = distance / radius; // 0.0 to 1.0

                // Probability function: higher chance for closer nodes
                // Using exponential decay: p = e^(-k * normalized_distance)
                // where k controls how quickly probability drops with distance
                let decay_factor = 3.0; // Higher values = steeper falloff
                let connection_probability = (-decay_factor * normalized_distance).exp();

                // Only add connection if we haven't already processed this pair
                // and the random chance succeeds
                let should_connect = rng.random::<f32>() < connection_probability;

                if should_connect && i < neighbor_index {
                    // Add bidirectional edges (only process each pair once)
                    adj[i].push(neighbor_index);
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
    let containment_strength = 0.001;

    // Calculate current average edge length for uniform target
    let mut total_edge_length = 0.0;
    let mut edge_count = 0;

    for i in 0..graph.nodes.nrows() {
        let pos_i = Vec2::new(graph.nodes[[i, 0]], graph.nodes[[i, 1]]);
        for &neighbor_index in &graph.adj[i] {
            if neighbor_index > i {
                // Only count each edge once
                let pos_neighbor = Vec2::new(
                    graph.nodes[[neighbor_index, 0]],
                    graph.nodes[[neighbor_index, 1]],
                );
                let dist = (pos_neighbor - pos_i).length();
                total_edge_length += dist;
                edge_count += 1;
            }
        }
    }

    let target_edge_length = if edge_count > 0 {
        total_edge_length / edge_count as f32
    } else {
        ideal_distance // Fallback to the provided ideal distance
    };

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

        // Spring force with connected neighbors - now using uniform target length
        for &neighbor_index in &graph.adj[i] {
            let pos_neighbor = Vec2::new(
                graph.nodes[[neighbor_index, 0]],
                graph.nodes[[neighbor_index, 1]],
            );
            let d = pos_neighbor - pos_i;
            let dist = d.length();
            let displacement = dist - target_edge_length;
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

/// Runs one time-step of the wave simulation with biharmonic stiffness term.
fn run_simulation_step(
    graph: &Graph,
    state: &mut SimState,
    c: f32,
    dt: f32,
    damping: f32,
    beta: f32,
) {
    let mut laplacian = Array1::<f32>::zeros(graph.nodes.nrows());
    let mut biharmonic = Array1::<f32>::zeros(graph.nodes.nrows());

    // First pass: calculate Laplacian (∇²u)
    for i in 0..graph.nodes.nrows() {
        let u_i = state.u[i];
        let mut lap = 0.0;
        for &neighbor_index in &graph.adj[i] {
            lap += state.u[neighbor_index] - u_i;
        }
        // Normalize by degree for consistency
        if !graph.adj[i].is_empty() {
            lap /= graph.adj[i].len() as f32;
        }
        laplacian[i] = lap;
    }

    // Second pass: calculate biharmonic (∇⁴u = ∇²(∇²u))
    for i in 0..graph.nodes.nrows() {
        let lap_i = laplacian[i];
        let mut bilap = 0.0;
        for &neighbor_index in &graph.adj[i] {
            bilap += laplacian[neighbor_index] - lap_i;
        }
        // Normalize by degree for consistency
        if !graph.adj[i].is_empty() {
            bilap /= graph.adj[i].len() as f32;
        }
        biharmonic[i] = bilap;
    }

    // Calculate acceleration: tension + bending stiffness
    let alpha = c * c; // tension coefficient (old c²)

    for i in 0..graph.nodes.nrows() {
        let acceleration = alpha * laplacian[i] - beta * biharmonic[i];
        let damping = damping * (state.v[i].powi(2) + state.u[i].powi(2) + 0.00001).sqrt();
        // Update velocity and displacement
        state.v[i] += acceleration * dt;
        state.v[i] *= 1.0 - damping / 300.0;
        state.u[i] += state.v[i] * dt;
        state.u[i] *= 1.0 - damping;
    }
}

use macroquad::texture::{Image, Texture2D};

/// Maps the simulation state (u, v) to a color for rendering.
fn get_color_for_node(u: f32, v: f32) -> Color {
    const S: f32 = 5.0;
    const K: f32 = 5.0;
    let u = (u * S * K).cbrt() * 1.0;
    let v = (v * S).cbrt() * 0.5;
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
        window_width: 500,
        window_height: 500,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    println!("Welcome to the Phonon Garden. Initializing...");

    // --- Configuration ---
    const NUM_NODES: usize = 50_000;
    const CONNECTION_RADIUS: f32 = 20.0;
    const WAVE_SPEED: f32 = 0.5;
    const SUB_ITERATIONS: usize = 3;
    const SPEEDUP: f32 = 5.0;
    const DAMPING: f32 = 0.2;
    const MAX_FRAME_TIME: f32 = 1.0 / 30.0;
    const RELAXATION_ITERATIONS: usize = 1;
    const REPULSION_STRENGTH: f32 = 0.0001;
    const IDEAL_SPRING_LENGTH_MULTIPLIER: f32 = 1.0;
    const BETA: f32 = 4.5; // bending stiffness coefficient (new parameter)

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

    // Energy monitoring - will display on screen continuously

    // --- Main Loop ---
    loop {
        // --- Input Handling ---
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let mouse_pos_f64 = [mx as f64, my as f64];

            // Find the nearest node to start the graph-based plucking
            if let Ok(neighbors) = kdtree.nearest(&mouse_pos_f64, 1, &squared_euclidean) {
                if let Some(&(_, &start_node)) = neighbors.first() {
                    // Graph-based Gaussian pluck parameters
                    let max_hops = 5; // Maximum number of graph hops to consider
                    let pluck_strength = 1.0; // Maximum displacement at center
                    let sigma = 0.3; // Standard deviation in terms of graph hops
                    let sigma_sq = sigma * sigma;

                    // Use BFS to find nodes within max_hops and their distances
                    use std::collections::{HashMap, VecDeque};
                    let mut queue = VecDeque::new();
                    let mut distances = HashMap::new();
                    let mut plucked_count = 0;

                    // Initialize BFS
                    queue.push_back((start_node, 0));
                    distances.insert(start_node, 0);

                    while let Some((current_node, hop_distance)) = queue.pop_front() {
                        // Apply Gaussian displacement based on hop distance
                        let gaussian_weight =
                            (-(hop_distance as f32).powi(2) / (2.0 * sigma_sq)).exp();
                        let displacement = pluck_strength * gaussian_weight;
                        state.u[current_node] += displacement;
                        plucked_count += 1;

                        // Add neighbors to queue if within max_hops
                        if hop_distance < max_hops {
                            for &neighbor_index in &graph.adj[current_node] {
                                use std::collections::hash_map::Entry;
                                if let Entry::Vacant(e) = distances.entry(neighbor_index) {
                                    e.insert(hop_distance + 1);
                                    queue.push_back((neighbor_index, hop_distance + 1));
                                }
                            }
                        }
                    }

                    println!(
                        "Plucked {} nodes with graph-based Gaussian distribution (max hops: {})",
                        plucked_count, max_hops
                    );
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
                let dt = get_frame_time().min(MAX_FRAME_TIME) * SPEEDUP / SUB_ITERATIONS as f32;
                for _ in 0..SUB_ITERATIONS {
                    // Run wave simulation
                    run_simulation_step(&graph, &mut state, WAVE_SPEED, dt, DAMPING, BETA);
                }
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

                // Calculate and display energy in real-time
                let kinetic_energy: f32 = state.v.iter().map(|&v| 0.5 * v * v).sum();
                let potential_energy: f32 = state.u.iter().map(|&u| 0.5 * u * u).sum();
                let total_energy = kinetic_energy + potential_energy;

                let energy_text = format!("Total Energy: {:.4}", total_energy);
                let kinetic_text = format!("Kinetic: {:.4}", kinetic_energy);
                let potential_text = format!("Potential: {:.4}", potential_energy);

                draw_text(&energy_text, 20.0, 50.0, 20.0, WHITE);
                draw_text(&kinetic_text, 20.0, 75.0, 20.0, WHITE);
                draw_text(&potential_text, 20.0, 100.0, 20.0, WHITE);
            }
        }

        next_frame().await
    }
}
