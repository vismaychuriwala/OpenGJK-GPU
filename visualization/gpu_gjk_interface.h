// gpu_gjk_interface.h
#pragma once
#include "gjk_integration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration only - no struct definition
typedef struct GPU_GJK_Context GPU_GJK_Context;

// Physics parameters for GPU simulation
typedef struct {
    Vector3f gravity;          // Gravity acceleration (e.g., {0, -9.8, 0})
    float deltaTime;           // Time step
    float dampingCoeff;        // Bounce damping (0-1)
    float boundarySize;        // World boundary size
    float collisionEpsilon;    // Distance threshold for collision
} GPU_PhysicsParams;

// Render data structure - only what CPU needs for drawing
typedef struct {
    Vector3f* positions;       // Array of positions [num_objects]
    bool* is_colliding;        // Array of collision flags [num_objects]
    int num_objects;
} GPU_RenderData;

// NEW API: GPU-owned simulation
bool gpu_gjk_init(GPU_GJK_Context** context, int max_objects, int max_pairs);
void gpu_gjk_cleanup(GPU_GJK_Context** context);

// Initialize simulation with physics objects
bool gpu_gjk_register_object(GPU_GJK_Context* context, int object_id,
                             const GJK_Shape* shape,
                             Vector3f position, Vector3f velocity,
                             float mass, float radius);

// Set collision pairs (call once at startup)
bool gpu_gjk_set_collision_pairs(GPU_GJK_Context* context, int* pairs, int num_pairs);

// Run full physics + collision step on GPU
bool gpu_gjk_step_simulation(GPU_GJK_Context* context, const GPU_PhysicsParams* params);

// Get rendering data (copies minimal data from GPU to CPU)
bool gpu_gjk_get_render_data(GPU_GJK_Context* context, GPU_RenderData* data);

// Reset simulation to initial state
bool gpu_gjk_reset_simulation(GPU_GJK_Context* context);

#ifdef __cplusplus
}
#endif