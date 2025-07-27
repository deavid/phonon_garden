use macroquad::prelude::{
    clear_background, draw_text, get_frame_time, is_key_pressed, is_mouse_button_pressed,
    mouse_position, next_frame, screen_height, screen_width, Color, Conf, KeyCode, MouseButton,
    BLACK, WHITE,
};
use macroquad::texture::draw_texture;
use ndarray::Array1;

// --- Configuration Constants ---
// These control the behavior of the phonon garden simulation.

const NUM_NODES_SQRT: usize = 600;

/// Number of nodes in the random geometric graph
const NUM_NODES: usize = NUM_NODES_SQRT * NUM_NODES_SQRT;

/// Maximum connection radius for graph edges (pixels)
const CONNECTION_RADIUS: f32 = 4000.0 / (NUM_NODES_SQRT as f32);

/// Maximum number of connections per node
const MAX_CONNECTIONS: usize = 60;

/// Probability of keeping each potential neighbor (0.0 to 1.0)
const NEIGHBOR_KEEP_PROBABILITY: f32 = 0.15;

/// Wave propagation speed coefficient
const WAVE_SPEED: f32 = 0.2;

/// Number of simulation sub-iterations per frame for temporal stability
const SUB_ITERATIONS: usize = 3;

/// Time acceleration factor for the simulation
const SPEEDUP: f32 = 20.0;

/// Base damping coefficient for wave attenuation
const DAMPING: f32 = 0.9;

/// Maximum frame time to prevent large simulation steps
const MAX_FRAME_TIME: f32 = 1.0 / 30.0;

/// Biharmonic stiffness coefficient for bending resistance
const BETA: f32 = 0.2;

/// Spring constant for restoring force that pulls displacement back to zero
/// Higher values create stronger restoring force, making waves return to equilibrium faster
const SPRING_CONSTANT: f32 = 0.005;

/// Influence radius for pixel rendering interpolation
const PIXEL_INFLUENCE_RADIUS: f32 = 6.0;

/// Boundary zone percentage for calculating boundary damping (as fraction from border)
const BOUNDARY_ZONE_PERCENTAGE: f32 = 0.05;

/// Whether to use asymmetric connectivity (directed graph)
/// When true, connections are directional and not guaranteed to be bidirectional
const USE_ASYMMETRIC_CONNECTIVITY: bool = false;

/// Probability that a reverse connection exists when using asymmetric connectivity
/// Only applies when USE_ASYMMETRIC_CONNECTIVITY is true
const REVERSE_CONNECTION_PROBABILITY: f32 = 0.8;

// --- Data Structures ---
// These will define the components of our simulation space.

use ndarray::Array2;
use sprs::CsMat;

// Represents the entire discrete space: a collection of nodes and their connections.
struct Graph {
    // Using Array2 for efficient row-based access to node positions.
    // Each row is a `[x, y]` coordinate.
    _nodes: Array2<f32>,
    // Adjacency list: for each node, a list of indices of its neighbors.
    adj: Vec<Vec<usize>>,
    // Boundary damping coefficient for each node (0.0 = no damping, 1.0 = full damping)
    boundary_damping: Array1<f32>,
    // Pre-computed normalized Laplacian matrix for efficient computation
    laplacian_matrix: CsMat<f32>,
}

// Holds the dynamic state of the wave simulation.
struct SimState {
    // Displacement ('u') for each node.
    u: Array1<f32>,
    // Velocity ('v') for each node.
    v: Array1<f32>,
}

// Stores pre-computed pixel influence data for smooth rendering
struct PixelInfluences {
    // For each pixel, stores (node_index, weight) pairs for nodes that influence it
    influences: Vec<Vec<(usize, f32)>>,
    width: u32,
    height: u32,
}

// Handles viewport scaling and aspect ratio management
struct Viewport {
    // Original pixel influences dimensions
    original_width: u32,
    original_height: u32,
    // Current screen dimensions
    screen_width: u32,
    screen_height: u32,
    // Viewport area within screen (for aspect ratio preservation)
    viewport_x: u32,
    viewport_y: u32,
    viewport_width: u32,
    viewport_height: u32,
    // Scale factors for mapping between original and viewport
    scale_x: f32,
    scale_y: f32,
}

impl Viewport {
    fn new(original_width: u32, original_height: u32) -> Self {
        Self {
            original_width,
            original_height,
            screen_width: original_width,
            screen_height: original_height,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: original_width,
            viewport_height: original_height,
            scale_x: 1.0,
            scale_y: 1.0,
        }
    }

    fn update_screen_size(&mut self, new_width: u32, new_height: u32) {
        self.screen_width = new_width;
        self.screen_height = new_height;

        // Calculate aspect ratios
        let original_aspect = self.original_width as f32 / self.original_height as f32;
        let screen_aspect = new_width as f32 / new_height as f32;

        if screen_aspect > original_aspect {
            // Screen is wider than original - pillarbox (black bars on sides)
            self.viewport_height = new_height;
            self.viewport_width = (new_height as f32 * original_aspect) as u32;
            self.viewport_x = (new_width - self.viewport_width) / 2;
            self.viewport_y = 0;
        } else {
            // Screen is taller than original - letterbox (black bars top/bottom)
            self.viewport_width = new_width;
            self.viewport_height = (new_width as f32 / original_aspect) as u32;
            self.viewport_x = 0;
            self.viewport_y = (new_height - self.viewport_height) / 2;
        }

        self.scale_x = self.viewport_width as f32 / self.original_width as f32;
        self.scale_y = self.viewport_height as f32 / self.original_height as f32;
    }

    // Convert screen coordinates to original pixel influence coordinates
    fn screen_to_original(&self, screen_x: u32, screen_y: u32) -> Option<(u32, u32)> {
        if screen_x < self.viewport_x
            || screen_x >= self.viewport_x + self.viewport_width
            || screen_y < self.viewport_y
            || screen_y >= self.viewport_y + self.viewport_height
        {
            return None; // Outside viewport
        }

        let viewport_x = screen_x - self.viewport_x;
        let viewport_y = screen_y - self.viewport_y;

        let original_x = (viewport_x as f32 / self.scale_x) as u32;
        let original_y = (viewport_y as f32 / self.scale_y) as u32;

        if original_x < self.original_width && original_y < self.original_height {
            Some((original_x, original_y))
        } else {
            None
        }
    }

    // Convert mouse position to simulation coordinates
    fn screen_mouse_to_sim(
        &self,
        mouse_x: f32,
        mouse_y: f32,
        sim_width: f32,
        sim_height: f32,
    ) -> Option<(f32, f32)> {
        if let Some((orig_x, orig_y)) = self.screen_to_original(mouse_x as u32, mouse_y as u32) {
            let sim_x = (orig_x as f32 / self.original_width as f32) * sim_width;
            let sim_y = (orig_y as f32 / self.original_height as f32) * sim_height;
            Some((sim_x, sim_y))
        } else {
            None
        }
    }
}

// Timing metrics for simulation steps
#[derive(Debug, Clone)]
struct SimulationMetrics {
    laplacian_time: f32,  // Time for Laplacian calculation (ms)
    biharmonic_time: f32, // Time for biharmonic calculation (ms)
    update_time: f32,     // Time for state updates (ms)
    total_time: f32,      // Total simulation step time (ms)
    frame_count: u64,     // Frame counter for temporal LOD
}

impl SimulationMetrics {
    fn new() -> Self {
        Self {
            laplacian_time: 0.0,
            biharmonic_time: 0.0,
            update_time: 0.0,
            total_time: 0.0,
            frame_count: 0,
        }
    }

    // Apply IIR filter for smoothing: new_value = alpha * current + (1-alpha) * new
    fn smooth_update(&mut self, new_metrics: &SimulationMetrics, alpha: f32) {
        self.laplacian_time =
            alpha * self.laplacian_time + (1.0 - alpha) * new_metrics.laplacian_time;
        self.biharmonic_time =
            alpha * self.biharmonic_time + (1.0 - alpha) * new_metrics.biharmonic_time;
        self.update_time = alpha * self.update_time + (1.0 - alpha) * new_metrics.update_time;
        self.total_time = alpha * self.total_time + (1.0 - alpha) * new_metrics.total_time;
        self.frame_count = new_metrics.frame_count; // Don't smooth frame count
    }
}

// --- Core Functions ---
// These are the placeholders for the main logic that the agent will implement.

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use rand::Rng;
use sprs::TriMat;
use std::time::Instant;

/// Creates a normalized Laplacian matrix from the adjacency list
fn create_laplacian_matrix(adj: &[Vec<usize>], num_nodes: usize) -> CsMat<f32> {
    let mut triplets = TriMat::new((num_nodes, num_nodes));

    for (i, neighbors) in adj.iter().enumerate().take(num_nodes) {
        let degree = neighbors.len() as f32;

        if degree > 0.0 {
            // For discrete Laplacian: L[i] = (1/degree) * sum(u[neighbors] - u[i])
            // This translates to: L[i] = (1/degree) * sum(u[neighbors]) - u[i]
            // In matrix form: L[i] = (1/degree) * sum over neighbors + (-1) * u[i]

            // Diagonal entry: -1 (coefficient of u[i])
            triplets.add_triplet(i, i, -1.0);

            // Off-diagonal entries: 1/degree for each neighbor
            let weight = 1.0 / degree;
            for &neighbor in neighbors {
                triplets.add_triplet(i, neighbor, weight);
            }
        }
        // If degree is 0, the row remains all zeros (isolated node)
    }

    triplets.to_csr()
}

/// Creates a Random Geometric Graph by connecting nodes within a given radius.
/// Uses a k-d tree for efficient neighbor search.
/// Can create either symmetric or asymmetric connectivity based on USE_ASYMMETRIC_CONNECTIVITY flag.
fn create_random_geometric_graph(
    num_nodes: usize,
    radius: f32,
    width: f32,
    height: f32,
) -> (Graph, KdTree<f64, usize, [f64; 2]>) {
    let total_start = Instant::now();
    let mut rng = rand::rngs::ThreadRng::default();

    // 1. Create nodes with random positions
    let step1_start = Instant::now();
    let mut nodes = Array2::<f32>::zeros((num_nodes, 2));
    let mut points = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let x = rng.random_range(0.0..width);
        let y = rng.random_range(0.0..height);
        nodes[[i, 0]] = x;
        nodes[[i, 1]] = y;
        points.push([x as f64, y as f64]);
    }
    let step1_duration = step1_start.elapsed();
    println!(
        "Step 1 (node generation) completed in {:.2?}",
        step1_duration
    );

    // 2. Build the k-d tree for efficient spatial queries
    let step2_start = Instant::now();
    let mut kdtree = KdTree::new(2);
    for (i, point) in points.iter().enumerate() {
        kdtree.add(*point, i).unwrap();
    }
    let step2_duration = step2_start.elapsed();
    println!(
        "Step 2 (k-d tree construction) completed in {:.2?}",
        step2_duration
    );

    // 3. Find potential neighbors and connect them (symmetric or asymmetric)
    let step3_start = Instant::now();
    let mut adj = vec![vec![]; num_nodes];
    let radius_squared = (radius as f64).powi(2);

    // Step 3 sub-timings
    let mut kdtree_query_total = std::time::Duration::ZERO;
    let mut potential_connections_total = std::time::Duration::ZERO;
    let mut adj_construction_total = std::time::Duration::ZERO;

    if USE_ASYMMETRIC_CONNECTIVITY {
        println!("Creating asymmetric (directed) connectivity...");

        for i in 0..num_nodes {
            // Time the kdtree query
            let query_start = Instant::now();
            let neighbors = kdtree
                .nearest(
                    &points[i],
                    (MAX_CONNECTIONS * 2 + 1).min(num_nodes),
                    &squared_euclidean,
                )
                .unwrap();
            kdtree_query_total += query_start.elapsed();

            // Time the potential connections construction
            let connections_start = Instant::now();
            let mut potential_connections: Vec<(usize, f64)> = neighbors
                .iter()
                .filter_map(|&(distance_sq, &neighbor_index)| {
                    if i != neighbor_index && distance_sq <= radius_squared {
                        if rng.random::<f32>() < NEIGHBOR_KEEP_PROBABILITY {
                            Some((neighbor_index, distance_sq))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            potential_connections.truncate(MAX_CONNECTIONS);
            potential_connections_total += connections_start.elapsed();

            // Time the adjacency list construction - ASYMMETRIC VERSION
            let adj_start = Instant::now();
            for (neighbor_index, _) in potential_connections {
                // Add directed edge from i to neighbor_index
                adj[i].push(neighbor_index);

                // Optionally add reverse connection with some probability
                if rng.random::<f32>() < REVERSE_CONNECTION_PROBABILITY {
                    adj[neighbor_index].push(i);
                }
            }
            adj_construction_total += adj_start.elapsed();
        }
    } else {
        println!("Creating symmetric (undirected) connectivity...");

        for i in 0..num_nodes {
            // Time the kdtree query - get more neighbors than we need to account for probabilistic filtering
            let query_start = Instant::now();
            let neighbors = kdtree
                .nearest(
                    &points[i],
                    (MAX_CONNECTIONS * 2 + 1).min(num_nodes),
                    &squared_euclidean,
                ) // Get extra neighbors for probabilistic selection
                .unwrap();
            kdtree_query_total += query_start.elapsed();

            // Time the potential connections construction
            let connections_start = Instant::now();
            let mut potential_connections: Vec<(usize, f64)> = neighbors
                .iter()
                .filter_map(|&(distance_sq, &neighbor_index)| {
                    // Filter out self and nodes beyond radius
                    if i != neighbor_index && distance_sq <= radius_squared {
                        // Apply probabilistic selection
                        if rng.random::<f32>() < NEIGHBOR_KEEP_PROBABILITY {
                            Some((neighbor_index, distance_sq))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            // No need to sort - nearest() already returns sorted results
            // Just truncate to ensure we don't exceed MAX_CONNECTIONS
            potential_connections.truncate(MAX_CONNECTIONS);
            potential_connections_total += connections_start.elapsed();

            // Time the adjacency list construction - SYMMETRIC VERSION
            let adj_start = Instant::now();
            for (neighbor_index, _) in potential_connections {
                if i < neighbor_index {
                    // Add bidirectional edges (only process each pair once)
                    adj[i].push(neighbor_index);
                    adj[neighbor_index].push(i);
                }
            }
            adj_construction_total += adj_start.elapsed();
        }
    }
    let step3_duration = step3_start.elapsed();
    println!(
        "Step 3 (neighbor finding & graph construction) completed in {:.2?}",
        step3_duration
    );
    println!(
        "  - kdtree queries: {:.2?} ({:.1}%)",
        kdtree_query_total,
        kdtree_query_total.as_secs_f64() / step3_duration.as_secs_f64() * 100.0
    );
    println!(
        "  - potential connections: {:.2?} ({:.1}%)",
        potential_connections_total,
        potential_connections_total.as_secs_f64() / step3_duration.as_secs_f64() * 100.0
    );
    println!(
        "  - adjacency construction: {:.2?} ({:.1}%)",
        adj_construction_total,
        adj_construction_total.as_secs_f64() / step3_duration.as_secs_f64() * 100.0
    );

    // 4. Calculate boundary damping coefficients
    let step4_start = Instant::now();
    let mut boundary_damping = Array1::<f32>::zeros(num_nodes);
    let boundary_zone_width = width.min(height) * BOUNDARY_ZONE_PERCENTAGE;

    for i in 0..num_nodes {
        let x = nodes[[i, 0]];
        let y = nodes[[i, 1]];

        // Calculate distance to nearest boundary
        let dist_to_boundary = f32::min(
            f32::min(x, width - x),  // distance to left/right edges
            f32::min(y, height - y), // distance to top/bottom edges
        );

        // If within boundary zone, calculate damping coefficient
        if dist_to_boundary < boundary_zone_width {
            // Smooth transition from 0 at boundary_zone_width to 1 at boundary
            let normalized_distance = dist_to_boundary / boundary_zone_width;
            // Use cubic function for smooth transition: (1 - t)³
            let damping_factor = (1.0 - normalized_distance).powi(3);
            boundary_damping[i] = damping_factor;
        }
        // else remains 0.0 (no boundary damping)
    }
    let step4_duration = step4_start.elapsed();
    println!(
        "Step 4 (boundary damping calculation) completed in {:.2?}",
        step4_duration
    );

    // 5. Create the normalized Laplacian matrix
    let step5_start = Instant::now();
    let laplacian_matrix = create_laplacian_matrix(&adj, num_nodes);
    let step5_duration = step5_start.elapsed();
    println!(
        "Step 5 (Laplacian matrix creation) completed in {:.2?}",
        step5_duration
    );

    let total_duration = total_start.elapsed();

    // Calculate connectivity statistics
    let total_edges: usize = adj.iter().map(|n| n.len()).sum();
    let avg_out_degree = total_edges as f32 / num_nodes as f32;

    // For asymmetric graphs, calculate in-degree statistics
    let connectivity_info = if USE_ASYMMETRIC_CONNECTIVITY {
        let mut in_degrees = vec![0; num_nodes];
        for neighbors in &adj {
            for &neighbor in neighbors {
                in_degrees[neighbor] += 1;
            }
        }
        let avg_in_degree = in_degrees.iter().sum::<usize>() as f32 / num_nodes as f32;
        format!(
            "asymmetric connectivity - avg out-degree: {:.2}, avg in-degree: {:.2}",
            avg_out_degree, avg_in_degree
        )
    } else {
        format!("symmetric connectivity - avg degree: {:.2}", avg_out_degree)
    };

    println!(
        "Graph generation completed in {:.2?}: {} nodes, {}, boundary damped nodes: {}, matrix nnz: {}",
        total_duration,
        num_nodes,
        connectivity_info,
        boundary_damping.iter().filter(|&&d| d > 0.0).count(),
        laplacian_matrix.nnz()
    );

    (
        Graph {
            _nodes: nodes,
            adj,
            boundary_damping,
            laplacian_matrix,
        },
        kdtree,
    )
}

/// Pre-computes pixel influence data for smooth interpolated rendering
fn create_pixel_influences(
    kdtree: &KdTree<f64, usize, [f64; 2]>,
    width: u32,
    height: u32,
    influence_radius: f32,
) -> PixelInfluences {
    let mut influences = Vec::with_capacity((width * height) as usize);
    let epsilon_sq = 0.2;
    let radius_sq = (influence_radius as f64).powi(2);

    println!("Pre-computing pixel influences...");

    for y in 0..height {
        for x in 0..width {
            let pixel_pos = [x as f64, y as f64];
            let mut pixel_influences = Vec::new();

            // Find all nodes within influence radius of this pixel
            if let Ok(neighbors) = kdtree.within(&pixel_pos, radius_sq, &squared_euclidean) {
                for &(distance_sq, &node_index) in neighbors.iter() {
                    // Calculate weight using squared inverse distance with epsilon
                    let weight = 1.0 / (distance_sq as f32 + epsilon_sq);
                    pixel_influences.push((node_index, weight));
                }

                // Keep only the top 4 strongest influences
                if pixel_influences.len() > 4 {
                    // Sort by weight (descending) and keep only top 4
                    pixel_influences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    pixel_influences.truncate(4);
                }
            }

            influences.push(pixel_influences);
        }

        // Progress indicator
        if y % 200 == 100 {
            println!("  Progress: {}/{} rows", y, height);
        }
    }

    let total_influences: usize = influences.iter().map(|p| p.len()).sum();
    println!(
        "Pixel influences computed: {} pixels, average {} influences per pixel",
        influences.len(),
        total_influences as f32 / influences.len() as f32
    );

    PixelInfluences {
        influences,
        width,
        height,
    }
}

/// Runs one time-step of the wave simulation with biharmonic stiffness term.
/// Uses parallel operations and optimized memory access patterns.
fn run_simulation_step(
    graph: &Graph,
    state: &mut SimState,
    c: f32,
    dt: f32,
    damping: f32,
    beta: f32,
    frame_count: u64,
) -> SimulationMetrics {
    let step_start = Instant::now();

    // First pass: calculate Laplacian (∇²u) using sparse matrix multiplication
    let laplacian_start = Instant::now();
    let laplacian = &graph.laplacian_matrix * &state.u;
    let laplacian_time = laplacian_start.elapsed().as_secs_f32() * 1000.0;

    // Second pass: calculate biharmonic (∇⁴u) = Laplacian of Laplacian
    let biharmonic_start = Instant::now();
    let biharmonic = &graph.laplacian_matrix * &laplacian;
    let biharmonic_time = biharmonic_start.elapsed().as_secs_f32() * 1000.0;

    // Update velocities and positions
    let update_start = Instant::now();
    let alpha = c * c;

    // Calculate acceleration: a = α∇²u - β∇⁴u - k*u (spring restoring force)
    let spring_force = &state.u * (-SPRING_CONSTANT);
    let acceleration = &laplacian * alpha - &biharmonic * beta + spring_force;

    // Update velocity: v(t+dt) = v(t) + a*dt
    let velocity_update = &acceleration * dt;
    state.v += &velocity_update;

    // Apply adaptive damping based on biharmonic magnitude
    // Higher curvature (biharmonic) leads to more damping
    let adaptive_damping = biharmonic.mapv(|b| {
        let biharmonic_magnitude = b.abs();
        damping * biharmonic_magnitude
    });

    // Apply the adaptive damping
    state
        .v
        .iter_mut()
        .zip(adaptive_damping.iter())
        .for_each(|(v, &adaptive_damp)| {
            *v *= 1.0 - adaptive_damp.min(0.4);
        });

    // Apply boundary damping
    state
        .v
        .iter_mut()
        .zip(graph.boundary_damping.iter())
        .for_each(|(v, &boundary)| {
            if boundary > 0.0 {
                *v *= 1.0 - boundary * 0.5;
            }
        });

    // Update position: u(t+dt) = u(t) + v*dt
    let displacement_update = &state.v * dt;
    state.u += &displacement_update;

    let update_time = update_start.elapsed().as_secs_f32() * 1000.0;
    let total_time = step_start.elapsed().as_secs_f32() * 1000.0;

    SimulationMetrics {
        laplacian_time,
        biharmonic_time,
        update_time,
        total_time,
        frame_count,
    }
}
use macroquad::texture::{Image, Texture2D};

/// Maps the simulation state (u, v) to a color for rendering.
fn get_color_for_node(u: f32, v: f32) -> Color {
    const S: f32 = 50.0;

    // Treat (u, v) as a vector and calculate its magnitude
    let magnitude = (u * u + v * v).sqrt();
    let intensity = ((magnitude * S).sqrt() / 2.0).clamp(0.0, 0.5);

    // Use the vector components for color direction, magnitude for intensity
    let u_normalized = if magnitude > 1e-6 { u / magnitude } else { 0.0 };
    let v_normalized = if magnitude > 1e-6 { v / magnitude } else { 0.0 };

    let r = (0.5 + u_normalized * intensity).clamp(0.0, 1.0);
    let g = (0.5 + v_normalized * intensity).clamp(0.0, 1.0);
    let b = (0.5 - u_normalized * intensity).clamp(0.0, 1.0);

    Color::new(r, g, b, 1.0)
}

/// Renders the simulation using smooth pixel interpolation
fn render_with_pixel_influences(
    image: &mut Image,
    pixel_influences: &PixelInfluences,
    state: &SimState,
) {
    // Clear the image buffer
    image.bytes.chunks_exact_mut(4).for_each(|pixel| {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        pixel[3] = 255;
    });

    // Render each pixel using weighted interpolation
    for y in 0..pixel_influences.height {
        for x in 0..pixel_influences.width {
            let pixel_index = (y * pixel_influences.width + x) as usize;
            let influences = &pixel_influences.influences[pixel_index];

            if !influences.is_empty() {
                let mut total_weight = 0.0;
                let mut weighted_u = 0.0;
                let mut weighted_v = 0.0;

                // Calculate weighted average of nearby nodes
                for &(node_index, weight) in influences {
                    total_weight += weight;
                    weighted_u += weight * state.u[node_index];
                    weighted_v += weight * state.v[node_index];
                }

                // Normalize and get color
                if total_weight > 0.0 {
                    let final_u = weighted_u / total_weight;
                    let final_v = weighted_v / total_weight;
                    let color = get_color_for_node(final_u, final_v);
                    image.set_pixel(x, y, color);
                }
            }
        }
    }
}

/// Renders the simulation with viewport scaling and aspect ratio preservation
fn render_with_viewport_scaling(
    image: &mut Image,
    pixel_influences: &PixelInfluences,
    viewport: &Viewport,
    state: &SimState,
) {
    // Clear the entire image buffer to black
    image.bytes.chunks_exact_mut(4).for_each(|pixel| {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        pixel[3] = 255;
    });

    // Only render within the viewport area
    for screen_y in viewport.viewport_y..(viewport.viewport_y + viewport.viewport_height) {
        for screen_x in viewport.viewport_x..(viewport.viewport_x + viewport.viewport_width) {
            // Map screen coordinates to original pixel influence coordinates
            if let Some((orig_x, orig_y)) = viewport.screen_to_original(screen_x, screen_y) {
                let pixel_index = (orig_y * pixel_influences.width + orig_x) as usize;
                if pixel_index < pixel_influences.influences.len() {
                    let influences = &pixel_influences.influences[pixel_index];

                    if !influences.is_empty() {
                        let mut total_weight = 0.0;
                        let mut weighted_u = 0.0;
                        let mut weighted_v = 0.0;

                        // Calculate weighted average of nearby nodes
                        for &(node_index, weight) in influences {
                            total_weight += weight;
                            weighted_u += weight * state.u[node_index];
                            weighted_v += weight * state.v[node_index];
                        }

                        // Normalize and get color
                        if total_weight > 0.0 {
                            let final_u = weighted_u / total_weight;
                            let final_v = weighted_v / total_weight;
                            let color = get_color_for_node(final_u, final_v);

                            // Make sure we're within image bounds
                            if screen_x < viewport.screen_width && screen_y < viewport.screen_height
                            {
                                image.set_pixel(screen_x, screen_y, color);
                            }
                        }
                    }
                }
            }
        }
    }
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

    // --- Setup Phase ---
    let (sim_width, sim_height) = (screen_width(), screen_height());
    let (graph, kdtree) =
        create_random_geometric_graph(NUM_NODES, CONNECTION_RADIUS, sim_width, sim_height);

    // Pre-compute pixel influences for smooth rendering
    let pixel_influences = create_pixel_influences(
        &kdtree,
        sim_width as u32,
        sim_height as u32,
        PIXEL_INFLUENCE_RADIUS,
    );

    // Initialize viewport for scaling and aspect ratio management
    let mut viewport = Viewport::new(sim_width as u32, sim_height as u32);

    let mut state = SimState {
        u: Array1::zeros(NUM_NODES),
        v: Array1::zeros(NUM_NODES),
    };

    let mut image = Image::gen_image_color(sim_width as u16, sim_height as u16, BLACK);
    let mut texture = Texture2D::from_image(&image);

    // Initialize smoothed simulation metrics
    let mut smoothed_metrics = SimulationMetrics::new();
    let smoothing_factor = 0.2; // IIR filter coefficient (higher = more smoothing)

    // Timing reporting
    let mut last_report_time = Instant::now();
    let report_interval = std::time::Duration::from_secs(5);

    // Frame counter for temporal level-of-detail
    let mut frame_count = 0u64;

    // --- Main Loop ---
    loop {
        // Check for window resize and update viewport
        let current_width = screen_width() as u32;
        let current_height = screen_height() as u32;

        if current_width != viewport.screen_width || current_height != viewport.screen_height {
            println!(
                "Window resized to {}x{}, updating viewport with aspect ratio preservation...",
                current_width, current_height
            );
            viewport.update_screen_size(current_width, current_height);

            // Recreate image and texture for new size
            image = Image::gen_image_color(current_width as u16, current_height as u16, BLACK);
            texture = Texture2D::from_image(&image);
        }

        // --- Input Handling ---
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();

            // Transform mouse coordinates to simulation space using viewport
            if let Some((sim_mx, sim_my)) =
                viewport.screen_mouse_to_sim(mx, my, sim_width, sim_height)
            {
                let mouse_pos_f64 = [sim_mx as f64, sim_my as f64];

                // Find the nearest node to start the graph-based plucking
                if let Ok(neighbors) = kdtree.nearest(&mouse_pos_f64, 1, &squared_euclidean) {
                    if let Some(&(_, &start_node)) = neighbors.first() {
                        // Graph-based Gaussian pluck parameters
                        let max_hops = 8; // Maximum number of graph hops to consider
                        let pluck_strength = 1.0; // Maximum displacement at center
                        let sigma = 0.6; // Standard deviation in terms of graph hops
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
        }

        // Reset simulation when 'R' key is pressed
        if is_key_pressed(KeyCode::R) {
            state.u.fill(0.0);
            state.v.fill(0.0);
            println!("Simulation reset!");
        }

        // --- Phase-specific Logic ---
        let dt = get_frame_time().min(MAX_FRAME_TIME) * SPEEDUP / SUB_ITERATIONS as f32;
        let mut frame_metrics = SimulationMetrics::new();
        for _ in 0..SUB_ITERATIONS {
            // Run wave simulation and collect metrics
            let step_metrics = run_simulation_step(
                &graph,
                &mut state,
                WAVE_SPEED,
                dt,
                DAMPING,
                BETA,
                frame_count,
            );
            // Accumulate metrics over sub-iterations
            frame_metrics.laplacian_time += step_metrics.laplacian_time / SUB_ITERATIONS as f32;
            frame_metrics.biharmonic_time += step_metrics.biharmonic_time / SUB_ITERATIONS as f32;
            frame_metrics.update_time += step_metrics.update_time / SUB_ITERATIONS as f32;
            frame_metrics.total_time += step_metrics.total_time / SUB_ITERATIONS as f32;
            frame_count += 1;
        }
        frame_metrics.frame_count = frame_count;

        // Apply smoothing to the accumulated frame metrics
        smoothed_metrics.smooth_update(&frame_metrics, smoothing_factor);

        // Report timing metrics to console every 5 seconds
        if last_report_time.elapsed() >= report_interval {
            println!(
                "Performance Report - Laplacian: {:.2}ms, Biharmonic: {:.2}ms, Updates: {:.2}ms, Total: {:.2}ms",
                smoothed_metrics.laplacian_time,
                smoothed_metrics.biharmonic_time,
                smoothed_metrics.update_time,
                smoothed_metrics.total_time
            );
            last_report_time = Instant::now();
        }

        // --- Drawing ---
        // Use viewport-scaled rendering if window size changed, otherwise use original
        if viewport.screen_width != viewport.original_width
            || viewport.screen_height != viewport.original_height
        {
            render_with_viewport_scaling(&mut image, &pixel_influences, &viewport, &state);
        } else {
            render_with_pixel_influences(&mut image, &pixel_influences, &state);
        }
        texture.update(&image);

        clear_background(BLACK);
        draw_texture(&texture, 0.0, 0.0, WHITE);

        // Draw phase-specific text
        draw_text("Simulating", 20.0, 20.0, 30.0, WHITE);
        draw_text(
            "Click to pluck • Press R to reset",
            20.0,
            viewport.screen_height as f32 - 30.0,
            20.0,
            WHITE,
        );

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

        // Display simulation timing metrics
        let timing_text = format!("Sim Total: {:.2}ms", smoothed_metrics.total_time);
        let laplacian_text = format!("Laplacian: {:.2}ms", smoothed_metrics.laplacian_time);
        let biharmonic_text = format!("Biharmonic: {:.2}ms", smoothed_metrics.biharmonic_time);
        let update_text = format!("Updates: {:.2}ms", smoothed_metrics.update_time);

        draw_text(&timing_text, 20.0, 140.0, 20.0, WHITE);
        draw_text(&laplacian_text, 20.0, 165.0, 20.0, WHITE);
        draw_text(&biharmonic_text, 20.0, 190.0, 20.0, WHITE);
        draw_text(&update_text, 20.0, 215.0, 20.0, WHITE);

        next_frame().await
    }
}
