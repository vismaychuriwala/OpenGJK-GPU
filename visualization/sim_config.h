#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

// ============================================================================
// SHARED SIMULATION CONFIGURATION
// ============================================================================

#define NUM_OBJECTS 1000           // Number of physics objects
#define MAX_PAIRS (NUM_OBJECTS * 50)  // Preallocated pair buffer (avg 50 neighbors per object with broad-phase)

// Kernel launch configuration
#define BLOCK_SIZE 256

// Spatial grid configuration for broad-phase culling
#define MAX_SPATIAL_GRID_SIZE 30         // Max grid divisions per axis (for allocation)
#define MAX_OBJECTS_PER_CELL 512          // Max objects per cell

// Set to 1 to load OBJ files at startup (slow for dense meshes until hull builder is improved)
#define LOAD_OBJS 0

// Random convex hull shape generation
#define MAX_HULL_VERTS    32     // maximum GJK vertices per random hull
#define MIN_HULL_VERTS     8     // minimum GJK vertices per random hull
#define HULL_SHAPE_RATIO 0.70f   // fraction of objects that are random convex hulls

// Object scale range: per-object base size and per-axis variation
#define SCALE_BASE_MIN   0.3f
#define SCALE_BASE_MAX   2.5f
#define SCALE_AXIS_MIN   0.6f
#define SCALE_AXIS_MAX   1.6f

// Spatial configuration (scales with number of objects)
#define BOUNDARY_SCALE_FACTOR 0.08f  // Boundary = num_objects * scale_factor
#define MIN_BOUNDARY_SIZE 10.0f
#define MAX_BOUNDARY_SIZE 100.0f  // Increased to accommodate more objects

// Physics parameters
#define GRAVITY_Y -9.80665f
#define FPS 60.0f
#define DELTA_TIME (1.0f / FPS)
#define COLLISION_EPSILON 0.1f
#define RESTITUTION 0.9f
#define FRICTION_COEFF 0.2f
#define ANGULAR_DAMPING 0.999f  // multiplied each physics step
#define BAUMGARTE_BETA  0.2f    // position correction fraction per step for penetrating pairs

// Helper macro to compute boundary size based on object count
#define COMPUTE_BOUNDARY(num_objects) \
    fminf(fmaxf(MIN_BOUNDARY_SIZE, (num_objects) * BOUNDARY_SCALE_FACTOR), MAX_BOUNDARY_SIZE)

#endif // SIM_CONFIG_H
