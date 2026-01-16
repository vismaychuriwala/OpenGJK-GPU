#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

// ============================================================================
// SHARED SIMULATION CONFIGURATION
// ============================================================================

#define NUM_OBJECTS 2000           // Number of physics objects
#define MAX_PAIRS (NUM_OBJECTS * 50)  // Preallocated pair buffer (avg 50 neighbors per object with broad-phase)

// Kernel launch configuration
#define BLOCK_SIZE 256

// Spatial grid configuration for broad-phase culling
#define SPATIAL_GRID_SIZE 30              // Grid divisions per axis
#define MAX_OBJECTS_PER_CELL 256          // Max objects per cell

// Spatial configuration (scales with number of objects)
#define BOUNDARY_SCALE_FACTOR 0.08f  // Boundary = num_objects * scale_factor
#define MIN_BOUNDARY_SIZE 10.0f
#define MAX_BOUNDARY_SIZE 100.0f  // Increased to accommodate more objects

// Physics parameters
#define GRAVITY_Y -9.80665f
#define FPS 60.0f
#define DELTA_TIME (1.0f / FPS)
#define DAMPING_COEFF 0.95f
#define COLLISION_EPSILON 0.1f
#define RESTITUTION 0.8f
#define ANGULAR_DAMPING 0.98f       // Air resistance for rotation
#define INERTIA_ICOSAHEDRON 0.4f    // I = 0.4 * mass * radius^2 (approximate)

// Helper macro to compute boundary size based on object count
#define COMPUTE_BOUNDARY(num_objects) \
    fminf(fmaxf(MIN_BOUNDARY_SIZE, (num_objects) * BOUNDARY_SCALE_FACTOR), MAX_BOUNDARY_SIZE)

// Helper macro to compute cell size based on boundary
// Cell size = (2 * boundary) / grid_size  (boundary is half-extent, so full size is 2*boundary)
#define COMPUTE_CELL_SIZE(boundary) \
    ((2.0f * (boundary)) / SPATIAL_GRID_SIZE)

#endif // SIM_CONFIG_H
