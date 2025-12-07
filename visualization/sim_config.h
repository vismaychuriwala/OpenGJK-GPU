#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

// ============================================================================
// SHARED SIMULATION CONFIGURATION
// ============================================================================

// Physics parameters
#define GRAVITY_Y -9.8f
#define FPS 60.0f
#define DELTA_TIME (1.0f / FPS)
#define DAMPING_COEFF 0.95f
#define COLLISION_EPSILON 0.1f
#define RESTITUTION 0.8f

// Spatial configuration (scales with number of objects)
#define BOUNDARY_SCALE_FACTOR 0.3f  // Boundary = num_objects * scale_factor
#define MIN_BOUNDARY_SIZE 10.0f
#define MAX_BOUNDARY_SIZE 100.0f

// Kernel launch configuration
#define BLOCK_SIZE 256

// Helper macro to compute boundary size based on object count
#define COMPUTE_BOUNDARY(num_objects) \
    fminf(fmaxf(MIN_BOUNDARY_SIZE, (num_objects) * BOUNDARY_SCALE_FACTOR), MAX_BOUNDARY_SIZE)

#endif // SIM_CONFIG_H
