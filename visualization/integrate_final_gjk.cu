// integrate_final_gjk.cu - GPU-owned physics simulation
#include "gpu_gjk_interface.h"
#include "sim_config.h"
#include "../GJK/gpu/openGJK.h"
#include "utils/scan_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA error checking function
void checkCUDAError(const char *msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// QUATERNION MATH DEVICE FUNCTIONS
// ============================================================================

__device__ Quaternion quat_normalize(Quaternion q) {
    float len = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (len < 1e-8f) {
        Quaternion identity = {1.0f, 0.0f, 0.0f, 0.0f};
        return identity;
    }
    Quaternion result = {q.w/len, q.x/len, q.y/len, q.z/len};
    return result;
}

__device__ Quaternion quat_multiply(Quaternion a, Quaternion b) {
    Quaternion result = {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
    return result;
}

__device__ Vector3f quat_rotate_vector(Quaternion q, Vector3f v) {
    // v' = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
    Vector3f u = {q.x, q.y, q.z};
    float s = q.w;

    // First cross: cross(u, v)
    Vector3f uv = {
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
    };

    // Second term: cross(u, uv) + s*v
    Vector3f uuv = {
        u.y*uv.z - u.z*uv.y,
        u.z*uv.x - u.x*uv.z,
        u.x*uv.y - u.y*uv.x
    };

    Vector3f result = {
        v.x + 2.0f * (uuv.x + s*uv.x),
        v.y + 2.0f * (uuv.y + s*uv.y),
        v.z + 2.0f * (uuv.z + s*uv.z)
    };
    return result;
}

__device__ Vector3f vector_cross(Vector3f a, Vector3f b) {
    Vector3f result = {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
    return result;
}

// GPU Physics Object (mirrored on device)
// Removed initial_position and initial_velocity to save memory (24 bytes per object)
typedef struct {
    Vector3f position;
    Vector3f velocity;
    Quaternion orientation;        // Rotation state (unit quaternion)
    Vector3f angular_velocity;     // Angular velocity (radians/sec)
    float mass;
    float radius;
    float inertia;                 // Moment of inertia
} GPU_PhysicsObject;

// Spatial grid cell - stores object IDs in this cell
typedef struct {
    int objects[MAX_OBJECTS_PER_CELL];    // Object IDs in this cell
    int count;                             // Number of objects in cell (0-32)
} GridCell;

// Persistent GPU buffer pool
struct GPU_Buffer_Pool {
    int max_pairs;

    // Coordinate buffers
    gkFloat* d_local_coords;   // Flat array for local-space vertex offsets [max_objects * 12 * 3]
    gkFloat* d_all_coords;     // Flat array for world-space coordinates [max_objects * 12 * 3]

    // Polytope, simplex, and distance buffers for parallel collision checks
    gkPolytope* d_polytopes1;  // Device array [max_pairs] - first polytope of each pair
    gkPolytope* d_polytopes2;  // Device array [max_pairs] - second polytope of each pair
    gkSimplex* d_simplices;    // Device array [max_pairs]
    gkFloat* d_distances;      // Device array [max_pairs]

    // EPA witness point buffers for rotation/torque
    gkFloat* d_witness1;       // Device array [max_pairs * 3] - contact points on body 1
    gkFloat* d_witness2;       // Device array [max_pairs * 3] - contact points on body 2
    gkFloat* d_contact_normals;// Device array [max_pairs * 3] - collision normals
};

// GPU GJK Context - owns all simulation state
struct GPU_GJK_Context {
    bool initialized;
    int device_id;
    int max_objects;
    int num_objects;
    int max_pairs;
    int num_pairs;

    // GPU-owned simulation state
    GPU_PhysicsObject* d_objects;     // Device array of physics objects
    GPU_PhysicsObject* h_objects;     // Host copy for initialization only (freed after sync)

    // Collision pair indices
    int* d_collision_pairs;           // Device array [num_pairs * 2]

    // Render data (pinned host memory for fast GPU->CPU transfer)
    Vector3f* h_positions;            // Pinned host array [num_objects]

    // Persistent buffer pool for collision detection
    GPU_Buffer_Pool* buffer_pool;

    // Spatial grid for broad-phase culling
    GridCell* d_grid;                 // Device array [GRID_SIZE^3] = 8000 cells
    int* d_pair_counts;               // Device array [max_objects] - pairs per object
    int* d_pair_offsets;              // Device array [max_objects] - prefix sum of counts
    int h_total_pairs;                // Host copy of total pairs generated

    // Temp buffers for multi-block scan
    int* d_block_sums;                // Per-block sums for scan
    int* d_block_incr;                // Scanned block sums
    int max_scan_blocks;              // Max blocks needed for scan
};

// CUDA kernel: Physics update for all objects
__global__ void physics_update_kernel(GPU_PhysicsObject* objects,
                                     int num_objects,
                                     const GPU_PhysicsParams params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;

    GPU_PhysicsObject* obj = &objects[i];

    // LINEAR MOTION
    // Update velocity with gravity
    obj->velocity.x += params.gravity.x * params.deltaTime;
    obj->velocity.y += params.gravity.y * params.deltaTime;
    obj->velocity.z += params.gravity.z * params.deltaTime;

    // Update position
    obj->position.x += obj->velocity.x * params.deltaTime;
    obj->position.y += obj->velocity.y * params.deltaTime;
    obj->position.z += obj->velocity.z * params.deltaTime;

    // ANGULAR MOTION
    // Integrate angular velocity into orientation: dq/dt = 0.5 * omega_quat * q
    float wx = obj->angular_velocity.x;
    float wy = obj->angular_velocity.y;
    float wz = obj->angular_velocity.z;
    float half_dt = 0.5f * params.deltaTime;

    Quaternion omega_quat = {0.0f, wx, wy, wz};  // Pure quaternion from angular velocity
    Quaternion dq = quat_multiply(omega_quat, obj->orientation);

    // Update orientation: q_new = q + dt * dq/dt
    obj->orientation.w += half_dt * dq.w;
    obj->orientation.x += half_dt * dq.x;
    obj->orientation.y += half_dt * dq.y;
    obj->orientation.z += half_dt * dq.z;

    // Renormalize quaternion to prevent drift
    obj->orientation = quat_normalize(obj->orientation);

    // Apply angular damping
    obj->angular_velocity.x *= ANGULAR_DAMPING;
    obj->angular_velocity.y *= ANGULAR_DAMPING;
    obj->angular_velocity.z *= ANGULAR_DAMPING;

    // Boundary collisions (all axes use same boundary size)
    float boundary = params.boundarySize;
    float ground = -boundary;  // Ground at bottom of boundary box
    float ceiling = boundary;  // Ceiling at top of boundary box

    // Y-axis (ground and ceiling)
    if (obj->position.y - obj->radius < ground) {
        obj->position.y = ground + obj->radius;
        obj->velocity.y = -obj->velocity.y * params.dampingCoeff;
    }
    if (obj->position.y + obj->radius > ceiling) {
        obj->position.y = ceiling - obj->radius;
        obj->velocity.y = -obj->velocity.y * params.dampingCoeff;
    }

    // X-axis
    if (fabsf(obj->position.x) > boundary - obj->radius) {
        obj->velocity.x = -obj->velocity.x * params.dampingCoeff;
        obj->position.x = (obj->position.x > 0.0f) ?
            boundary - obj->radius : -boundary + obj->radius;
    }

    // Z-axis
    if (fabsf(obj->position.z) > boundary - obj->radius) {
        obj->velocity.z = -obj->velocity.z * params.dampingCoeff;
        obj->position.z = (obj->position.z > 0.0f) ?
            boundary - obj->radius : -boundary + obj->radius;
    }
}

// CUDA kernel: Build world-space coordinates from local vertices + positions
__global__ void build_world_coords_kernel(GPU_PhysicsObject* objects,
                                         gkFloat* local_vertices,
                                         gkFloat* world_coords,
                                         int* vertices_per_shape,
                                         int num_objects)
{
    int obj_id = blockIdx.x;
    int vert_id = threadIdx.x;

    if (obj_id >= num_objects) return;

    int num_verts = vertices_per_shape[obj_id];
    if (vert_id >= num_verts) return;

    // Calculate offsets into flat arrays
    int local_offset = obj_id * 12 * 3;  // Assuming max 12 vertices per shape
    int world_offset = obj_id * 12 * 3;

    int local_idx = local_offset + vert_id * 3;
    int world_idx = world_offset + vert_id * 3;

    // Transform: world = local + position
    world_coords[world_idx + 0] = local_vertices[local_idx + 0] + objects[obj_id].position.x;
    world_coords[world_idx + 1] = local_vertices[local_idx + 1] + objects[obj_id].position.y;
    world_coords[world_idx + 2] = local_vertices[local_idx + 2] + objects[obj_id].position.z;
}

// ============================================================================
// SPATIAL GRID KERNELS FOR BROAD-PHASE CULLING
// ============================================================================

// Note: Grid clearing is done with cudaMemset in the host code for better performance

// Kernel 1: Insert objects into spatial grid
__global__ void insert_objects_kernel(GPU_PhysicsObject* objects,
                                      int num_objects,
                                      GridCell* grid,
                                      float cell_size,
                                      float boundary)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    Vector3f pos = objects[obj_id].position;

    // Convert world position to grid cell coordinates
    // Map from [-boundary, boundary] to [0, GRID_SIZE-1]
    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);

    // Clamp to grid bounds
    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    // Calculate 1D cell index
    int cell_idx = cx + cy * SPATIAL_GRID_SIZE + cz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;

    // Atomically insert object into cell
    int slot = atomicAdd(&grid[cell_idx].count, 1);
    if (slot < MAX_OBJECTS_PER_CELL) {
        grid[cell_idx].objects[slot] = obj_id;
    }
    // If slot >= MAX_OBJECTS_PER_CELL, object is silently dropped (cell overflow)
    // This is safe because read loops clamp to min(cell->count, MAX_OBJECTS_PER_CELL)
}

// Kernel 2: Count pairs per object (Pass 1 of two-pass approach)
__global__ void count_pairs_kernel(GPU_PhysicsObject* objects,
                                   int num_objects,
                                   GridCell* grid,
                                   float cell_size,
                                   float boundary,
                                   int* pair_counts)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    Vector3f pos = objects[obj_id].position;
    int count = 0;

    // Get object's cell coordinates
    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);

    // Clamp to grid bounds
    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    // Check 27 neighboring cells (including self)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx;
                int ny = cy + dy;
                int nz = cz + dz;

                // Skip if out of bounds
                if (nx < 0 || nx >= SPATIAL_GRID_SIZE) continue;
                if (ny < 0 || ny >= SPATIAL_GRID_SIZE) continue;
                if (nz < 0 || nz >= SPATIAL_GRID_SIZE) continue;

                // Get neighbor cell
                int cell_idx = nx + ny * SPATIAL_GRID_SIZE + nz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
                GridCell* cell = &grid[cell_idx];

                // Count pairs with objects in this cell
                for (int i = 0; i < cell->count && i < MAX_OBJECTS_PER_CELL; i++) {
                    int other_id = cell->objects[i];

                    // Only count each pair once (avoid duplicates)
                    if (other_id <= obj_id) continue;

                    count++;
                }
            }
        }
    }

    // Store count for this object
    pair_counts[obj_id] = count;
}

// Kernel 3: Generate pairs using offsets from prefix sum (Pass 2 of two-pass approach)
__global__ void generate_pairs_kernel(GPU_PhysicsObject* objects,
                                       int num_objects,
                                      GridCell* grid,
                                      float cell_size,
                                      float boundary,
                                      int* pair_offsets,
                                      int* pair_buffer,
                                      int max_pairs)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    Vector3f pos = objects[obj_id].position;
    int write_offset = pair_offsets[obj_id];  // Where this object writes its pairs

    // Get object's cell coordinates
    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);

    // Clamp to grid bounds
    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    // Check 27 neighboring cells (including self)
    int local_pair_idx = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx;
                int ny = cy + dy;
                int nz = cz + dz;

                // Skip if out of bounds
                if (nx < 0 || nx >= SPATIAL_GRID_SIZE) continue;
                if (ny < 0 || ny >= SPATIAL_GRID_SIZE) continue;
                if (nz < 0 || nz >= SPATIAL_GRID_SIZE) continue;

                // Get neighbor cell
                int cell_idx = nx + ny * SPATIAL_GRID_SIZE + nz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
                GridCell* cell = &grid[cell_idx];

                // Generate pairs with objects in this cell
                for (int i = 0; i < cell->count && i < MAX_OBJECTS_PER_CELL; i++) {
                    int other_id = cell->objects[i];

                    // Only generate each pair once (avoid duplicates)
                    if (other_id <= obj_id) continue;

                    // Write pair to buffer at predetermined offset
                    int global_pair_idx = write_offset + local_pair_idx;

                    // Bounds check to prevent buffer overflow (fixes phantom collision bug)
                    if (global_pair_idx >= max_pairs) {
                        return;  // Stop writing if we exceed buffer capacity
                    }

                    pair_buffer[global_pair_idx * 2 + 0] = obj_id;
                    pair_buffer[global_pair_idx * 2 + 1] = other_id;
                    local_pair_idx++;
                }
            }
        }
    }
}

// ============================================================================
// COLLISION DETECTION AND RESPONSE KERNELS
// ============================================================================

// CUDA kernel: Collision response with rotation and torque using EPA witness points
__global__ void check_and_respond_with_rotation_kernel(
    GPU_PhysicsObject* objects,
    int* pairs,
    gkFloat* distances,
    gkFloat* witness1,      // Contact points on body 1
    gkFloat* witness2,      // Contact points on body 2
    gkFloat* normals,       // Contact normals
    float epsilon,
    int num_pairs,
    int num_objects)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;

    // Check collision - distance near 0 means collision (GJK) or penetration (EPA gives negative)
    // Accept both cases: distance <= epsilon OR distance is negative (penetration)
    if (distances[p] > epsilon) return;

    int idA = pairs[p * 2 + 0];
    int idB = pairs[p * 2 + 1];

    if (idA < 0 || idA >= num_objects || idB < 0 || idB >= num_objects) return;

    GPU_PhysicsObject* obj1 = &objects[idA];
    GPU_PhysicsObject* obj2 = &objects[idB];

    // Get witness points (contact points)
    Vector3f contact1 = {
        witness1[p * 3 + 0],
        witness1[p * 3 + 1],
        witness1[p * 3 + 2]
    };

    Vector3f contact2 = {
        witness2[p * 3 + 0],
        witness2[p * 3 + 1],
        witness2[p * 3 + 2]
    };

    // Get contact normal (from obj1 to obj2)
    Vector3f normal = {
        normals[p * 3 + 0],
        normals[p * 3 + 1],
        normals[p * 3 + 2]
    };

    // Normalize (EPA should provide normalized, but safety check)
    float nlen = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    if (nlen < 1e-6f) return;
    normal.x /= nlen; normal.y /= nlen; normal.z /= nlen;

    // Moment arms (contact point - center of mass)
    Vector3f r1 = {
        contact1.x - obj1->position.x,
        contact1.y - obj1->position.y,
        contact1.z - obj1->position.z
    };

    Vector3f r2 = {
        contact2.x - obj2->position.x,
        contact2.y - obj2->position.y,
        contact2.z - obj2->position.z
    };

    // Relative velocity at contact point
    // v_rel = (v2 + w2 × r2) - (v1 + w1 × r1)
    Vector3f w1_cross_r1 = vector_cross(obj1->angular_velocity, r1);
    Vector3f w2_cross_r2 = vector_cross(obj2->angular_velocity, r2);

    Vector3f v1_at_contact = {
        obj1->velocity.x + w1_cross_r1.x,
        obj1->velocity.y + w1_cross_r1.y,
        obj1->velocity.z + w1_cross_r1.z
    };

    Vector3f v2_at_contact = {
        obj2->velocity.x + w2_cross_r2.x,
        obj2->velocity.y + w2_cross_r2.y,
        obj2->velocity.z + w2_cross_r2.z
    };

    Vector3f v_rel = {
        v2_at_contact.x - v1_at_contact.x,
        v2_at_contact.y - v1_at_contact.y,
        v2_at_contact.z - v1_at_contact.z
    };

    // Velocity along normal
    float v_rel_n = v_rel.x * normal.x + v_rel.y * normal.y + v_rel.z * normal.z;

    // Don't resolve if separating
    if (v_rel_n > 0.0f) return;

    // Compute impulse magnitude with rotation
    // j = -(1 + e) * v_rel_n / (1/m1 + 1/m2 + ((r1 × n)²/I1) + ((r2 × n)²/I2))
    Vector3f r1_cross_n = vector_cross(r1, normal);
    Vector3f r2_cross_n = vector_cross(r2, normal);

    float angular_factor1 = (r1_cross_n.x*r1_cross_n.x + r1_cross_n.y*r1_cross_n.y + r1_cross_n.z*r1_cross_n.z) / obj1->inertia;
    float angular_factor2 = (r2_cross_n.x*r2_cross_n.x + r2_cross_n.y*r2_cross_n.y + r2_cross_n.z*r2_cross_n.z) / obj2->inertia;

    float j = -(1.0f + RESTITUTION) * v_rel_n;
    j /= (1.0f/obj1->mass + 1.0f/obj2->mass + angular_factor1 + angular_factor2);

    // Apply linear impulse
    Vector3f impulse = {j * normal.x, j * normal.y, j * normal.z};

    obj1->velocity.x -= impulse.x / obj1->mass;
    obj1->velocity.y -= impulse.y / obj1->mass;
    obj1->velocity.z -= impulse.z / obj1->mass;

    obj2->velocity.x += impulse.x / obj2->mass;
    obj2->velocity.y += impulse.y / obj2->mass;
    obj2->velocity.z += impulse.z / obj2->mass;

    // Apply angular impulse (torque = r × impulse)
    Vector3f torque1 = vector_cross(r1, impulse);
    Vector3f torque2 = vector_cross(r2, impulse);

    obj1->angular_velocity.x -= torque1.x / obj1->inertia;
    obj1->angular_velocity.y -= torque1.y / obj1->inertia;
    obj1->angular_velocity.z -= torque1.z / obj1->inertia;

    obj2->angular_velocity.x += torque2.x / obj2->inertia;
    obj2->angular_velocity.y += torque2.y / obj2->inertia;
    obj2->angular_velocity.z += torque2.z / obj2->inertia;
}

// ============================================================================
// PUBLIC API IMPLEMENTATION
// ============================================================================

bool gpu_gjk_init(GPU_GJK_Context** context, int max_objects, int max_pairs)
{
    printf("Initializing GPU-owned physics simulation...\n");

    *context = (GPU_GJK_Context*)malloc(sizeof(GPU_GJK_Context));
    if (!*context) {
        fprintf(stderr, "Failed to allocate GPU context\n");
        return false;
    }

    memset(*context, 0, sizeof(GPU_GJK_Context));

    GPU_GJK_Context* ctx = *context;
    ctx->initialized = false;
    ctx->device_id = 0;
    ctx->max_objects = max_objects;
    ctx->max_pairs = max_pairs;
    ctx->num_objects = 0;
    ctx->num_pairs = 0;

    // Allocate host memory
    ctx->h_objects = (GPU_PhysicsObject*)malloc(sizeof(GPU_PhysicsObject) * max_objects);

    // Allocate pinned host memory for fast GPU->CPU transfer
    cudaMallocHost(&ctx->h_positions, sizeof(Vector3f) * max_objects);
    checkCUDAError("cudaMallocHost h_positions");

    // Allocate GPU memory for physics objects
    cudaMalloc(&ctx->d_objects, sizeof(GPU_PhysicsObject) * max_objects);
    checkCUDAError("cudaMalloc d_objects");

    cudaMalloc(&ctx->d_collision_pairs, sizeof(int) * max_pairs * 2);
    checkCUDAError("cudaMalloc d_collision_pairs");

    // Allocate buffer pool
    ctx->buffer_pool = (GPU_Buffer_Pool*)malloc(sizeof(GPU_Buffer_Pool));
    if (!ctx->buffer_pool) {
        fprintf(stderr, "Failed to allocate buffer pool\n");
        return false;
    }
    ctx->buffer_pool->max_pairs = max_pairs;

    // Allocate persistent GPU buffers for collision detection
    cudaMalloc(&ctx->buffer_pool->d_local_coords, sizeof(gkFloat) * max_objects * 12 * 3);
    checkCUDAError("cudaMalloc d_local_coords");

    cudaMalloc(&ctx->buffer_pool->d_all_coords, sizeof(gkFloat) * max_objects * 12 * 3);
    checkCUDAError("cudaMalloc d_all_coords");

    cudaMalloc(&ctx->buffer_pool->d_polytopes1, sizeof(gkPolytope) * max_pairs);
    checkCUDAError("cudaMalloc d_polytopes1");

    cudaMalloc(&ctx->buffer_pool->d_polytopes2, sizeof(gkPolytope) * max_pairs);
    checkCUDAError("cudaMalloc d_polytopes2");

    cudaMalloc(&ctx->buffer_pool->d_simplices, sizeof(gkSimplex) * max_pairs);
    checkCUDAError("cudaMalloc d_simplices");

    cudaMalloc(&ctx->buffer_pool->d_distances, sizeof(gkFloat) * max_pairs);
    checkCUDAError("cudaMalloc d_distances");

    // Allocate EPA witness point buffers for rotation/torque
    cudaMalloc(&ctx->buffer_pool->d_witness1, sizeof(gkFloat) * max_pairs * 3);
    checkCUDAError("cudaMalloc d_witness1");

    cudaMalloc(&ctx->buffer_pool->d_witness2, sizeof(gkFloat) * max_pairs * 3);
    checkCUDAError("cudaMalloc d_witness2");

    cudaMalloc(&ctx->buffer_pool->d_contact_normals, sizeof(gkFloat) * max_pairs * 3);
    checkCUDAError("cudaMalloc d_contact_normals");

    // Allocate spatial grid for broad-phase culling
    int total_cells = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    cudaMalloc(&ctx->d_grid, sizeof(GridCell) * total_cells);
    checkCUDAError("cudaMalloc d_grid");

    // Allocate pair count/offset buffers with power-of-2 size for scan algorithm
    int scan_buffer_size = next_power_of_2(max_objects);
    cudaMalloc(&ctx->d_pair_counts, sizeof(int) * scan_buffer_size);
    checkCUDAError("cudaMalloc d_pair_counts");

    cudaMalloc(&ctx->d_pair_offsets, sizeof(int) * scan_buffer_size);
    checkCUDAError("cudaMalloc d_pair_offsets");
    ctx->h_total_pairs = 0;

    printf("Allocated scan buffers: %d elements (rounded up from %d)\n",
           scan_buffer_size, max_objects);

    ctx->initialized = true;
    printf("GPU simulation context initialized (max_objects=%d, max_pairs=%d)\n",
           max_objects, max_pairs);
    printf("Spatial grid: %dx%dx%d = %d cells (%.2f MB)\n",
           SPATIAL_GRID_SIZE, SPATIAL_GRID_SIZE, SPATIAL_GRID_SIZE, total_cells,
           (sizeof(GridCell) * total_cells) / (1024.0f * 1024.0f));
    printf("Grid cell capacity: %d objects per cell\n", MAX_OBJECTS_PER_CELL);

    return true;
}

void gpu_gjk_cleanup(GPU_GJK_Context** context)
{
    if (!context || !*context) return;

    GPU_GJK_Context* ctx = *context;

    // Free device memory
    if (ctx->d_objects) cudaFree(ctx->d_objects);
    if (ctx->d_collision_pairs) cudaFree(ctx->d_collision_pairs);

    // Free spatial grid
    if (ctx->d_grid) cudaFree(ctx->d_grid);
    if (ctx->d_pair_counts) cudaFree(ctx->d_pair_counts);
    if (ctx->d_pair_offsets) cudaFree(ctx->d_pair_offsets);

    // Free buffer pool
    if (ctx->buffer_pool) {
        cudaFree(ctx->buffer_pool->d_local_coords);
        cudaFree(ctx->buffer_pool->d_all_coords);
        cudaFree(ctx->buffer_pool->d_polytopes1);
        cudaFree(ctx->buffer_pool->d_polytopes2);
        cudaFree(ctx->buffer_pool->d_simplices);
        cudaFree(ctx->buffer_pool->d_distances);
        cudaFree(ctx->buffer_pool->d_witness1);
        cudaFree(ctx->buffer_pool->d_witness2);
        cudaFree(ctx->buffer_pool->d_contact_normals);
        free(ctx->buffer_pool);
    }

    // Free pinned host memory
    if (ctx->h_positions) cudaFreeHost(ctx->h_positions);

    // Free host memory (should already be freed after sync, but check just in case)
    if (ctx->h_objects) {
        free(ctx->h_objects);
        ctx->h_objects = NULL;
    }

    free(ctx);
    *context = NULL;

    printf("GPU simulation context cleaned up\n");
}

bool gpu_gjk_register_object(GPU_GJK_Context* context, int object_id,
                             const GJK_Shape* shape,
                             Vector3f position, Vector3f velocity,
                             Quaternion orientation, Vector3f angular_velocity,
                             float mass, float radius)
{
    if (!context || !shape) return false;
    if (object_id < 0 || object_id >= context->max_objects) return false;

    // Store physics object on host (removed initial_position and initial_velocity)
    context->h_objects[object_id].position = position;
    context->h_objects[object_id].velocity = velocity;
    context->h_objects[object_id].orientation = orientation;
    context->h_objects[object_id].angular_velocity = angular_velocity;
    context->h_objects[object_id].mass = mass;
    context->h_objects[object_id].radius = radius;
    context->h_objects[object_id].inertia = INERTIA_ICOSAHEDRON * mass * radius * radius;

    // Store local-space vertex offsets in consolidated buffer
    int num_verts = shape->num_vertices;
    gkFloat* h_local_verts = (gkFloat*)malloc(sizeof(gkFloat) * num_verts * 3);
    for (int i = 0; i < num_verts; i++) {
        h_local_verts[i * 3 + 0] = shape->vertices[i].x;
        h_local_verts[i * 3 + 1] = shape->vertices[i].y;
        h_local_verts[i * 3 + 2] = shape->vertices[i].z;
    }

    // Copy to the correct offset in d_local_coords (assumes 12 vertices per object)
    int coord_offset = object_id * 12 * 3;
    cudaMemcpy(context->buffer_pool->d_local_coords + coord_offset,
               h_local_verts,
               sizeof(gkFloat) * num_verts * 3,
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy register object vertices", __LINE__);
    free(h_local_verts);

    if (object_id + 1 > context->num_objects) {
        context->num_objects = object_id + 1;
    }

    printf("Registered GPU object %d: pos=(%.2f, %.2f, %.2f), mass=%.2f, radius=%.2f\n",
           object_id, position.x, position.y, position.z, mass, radius);

    return true;
}

// CUDA kernel: Initialize polytopes ONCE at startup (called from gpu_gjk_set_collision_pairs)
// GJK initializes support cache (s[], s_idx) itself - we only set numpoints and coord pointer
__global__ void init_polytopes_once_kernel(int* pairs,
                                           gkFloat* all_coords,
                                           gkPolytope* polytopes1,
                                           gkPolytope* polytopes2,
                                           int num_pairs,
                                           int num_vertices_per_shape)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;

    int idA = pairs[p * 2 + 0];
    int idB = pairs[p * 2 + 1];

    // Setup polytope A (first in pair) - only geometry, GJK handles support cache
    polytopes1[p].numpoints = num_vertices_per_shape;
    polytopes1[p].coord = all_coords + (idA * 12 * 3);

    // Setup polytope B (second in pair)
    polytopes2[p].numpoints = num_vertices_per_shape;
    polytopes2[p].coord = all_coords + (idB * 12 * 3);
}

bool gpu_gjk_set_collision_pairs(GPU_GJK_Context* context, int* pairs, int num_pairs)
{
    if (!context || !pairs) return false;
    if (num_pairs > context->max_pairs) {
        fprintf(stderr, "Too many pairs: %d > max %d\n", num_pairs, context->max_pairs);
        return false;
    }

    context->num_pairs = num_pairs;

    // Copy pairs to GPU
    cudaMemcpy(context->d_collision_pairs, pairs,
               sizeof(int) * num_pairs * 2,
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy collision pairs", __LINE__);

    // Copy initial object states to GPU
    cudaMemcpy(context->d_objects, context->h_objects,
               sizeof(GPU_PhysicsObject) * context->num_objects,
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy object states", __LINE__);

    // Initialize polytopes ONCE (no longer done per-frame!)
    int pairBlocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_polytopes_once_kernel<<<pairBlocks, BLOCK_SIZE>>>(
        context->d_collision_pairs,
        context->buffer_pool->d_all_coords,
        context->buffer_pool->d_polytopes1,
        context->buffer_pool->d_polytopes2,
        num_pairs,
        12);  // All shapes are icosahedrons with 12 vertices
    checkCUDAError("init_polytopes_once_kernel (set_collision_pairs)", __LINE__);
    cudaDeviceSynchronize();
    checkCUDAError("cudaDeviceSynchronize (set_collision_pairs)", __LINE__);

    printf("Set %d collision pairs on GPU (polytopes initialized)\n", num_pairs);
    return true;
}

// CUDA kernel: Transform local vertices to world space
// Recalculates world = rotate(local) + position each frame (keeps vertices in sync with object)
__global__ void transform_to_world_kernel(GPU_PhysicsObject* objects,
                                          gkFloat* local_coords,
                                          gkFloat* world_coords,
                                          int num_objects)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each object has 12 vertices, compute which object and vertex this thread handles
    int obj_id = tid / 12;
    int vert_id = tid % 12;

    if (obj_id >= num_objects) return;
    if (vert_id >= 12) return;  // Icosahedron has 12 vertices

    GPU_PhysicsObject* obj = &objects[obj_id];
    int vertex_offset = (obj_id * 12 + vert_id) * 3;

    // Get local vertex
    Vector3f local_vert = {
        local_coords[vertex_offset + 0],
        local_coords[vertex_offset + 1],
        local_coords[vertex_offset + 2]
    };

    // Apply rotation first, then translation
    Vector3f rotated = quat_rotate_vector(obj->orientation, local_vert);

    // Transform to world space: world = rotate(local) + position
    world_coords[vertex_offset + 0] = rotated.x + obj->position.x;
    world_coords[vertex_offset + 1] = rotated.y + obj->position.y;
    world_coords[vertex_offset + 2] = rotated.z + obj->position.z;
}

bool gpu_gjk_step_simulation(GPU_GJK_Context* context, const GPU_PhysicsParams* params)
{
    if (!context || !context->initialized || !params) return false;

    int num_objs = context->num_objects;
    int num_pairs = context->num_pairs;

    // Step 1: Physics update (parallel per object)
    int objBlocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    physics_update_kernel<<<objBlocks, BLOCK_SIZE>>>(context->d_objects, num_objs, *params);
    checkCUDAError("physics_update_kernel", __LINE__);

    // Step 2: Transform local vertices to world space (world = local + position)
    // Total work: num_objs * 12 vertices per object
    int total_vertices = num_objs * 12;
    int transformBlocks = (total_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transform_to_world_kernel<<<transformBlocks, BLOCK_SIZE>>>(
        context->d_objects,
        context->buffer_pool->d_local_coords,
        context->buffer_pool->d_all_coords,
        num_objs);
    checkCUDAError("transform_to_world_kernel", __LINE__);

    // Step 3: Initialize polytopes for current pairs (pairs change each frame with dynamic culling)
    if (num_pairs > 0) {
        int pairBlocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_polytopes_once_kernel<<<pairBlocks, BLOCK_SIZE>>>(
            context->d_collision_pairs,
            context->buffer_pool->d_all_coords,
            context->buffer_pool->d_polytopes1,
            context->buffer_pool->d_polytopes2,
            num_pairs,
            12);  // All shapes are icosahedrons with 12 vertices
        checkCUDAError("init_polytopes_once_kernel", __LINE__);
    }

    // Step 4: Batched GJK collision detection
    // Each collision uses 16 threads (half-warp), so use 32 threads per block for 2 pairs
    const int threadsPerCollision = 16;
    const int collisionsPerBlock = 2;
    const int threadsPerBlock = threadsPerCollision * collisionsPerBlock;  // 32
    int gjkBlocks = (num_pairs + collisionsPerBlock - 1) / collisionsPerBlock;

    if (num_pairs > 0) {
        compute_minimum_distance<<<gjkBlocks, threadsPerBlock>>>(
            context->buffer_pool->d_polytopes1,  // First polytopes [num_pairs]
            context->buffer_pool->d_polytopes2,  // Second polytopes [num_pairs]
            context->buffer_pool->d_simplices,
            context->buffer_pool->d_distances,
            num_pairs);
        checkCUDAError("compute_minimum_distance (GJK)", __LINE__);

        // Step 5: EPA - Compute witness points and contact normals for rotation
        // EPA uses 32 threads per collision (full warp)
        const int epaThreadsPerCollision = 32;
        int epaBlocks = (num_pairs + epaThreadsPerCollision - 1) / epaThreadsPerCollision;

        compute_epa<<<epaBlocks, epaThreadsPerCollision>>>(
            context->buffer_pool->d_polytopes1,
            context->buffer_pool->d_polytopes2,
            context->buffer_pool->d_simplices,
            context->buffer_pool->d_distances,
            context->buffer_pool->d_witness1,
            context->buffer_pool->d_witness2,
            context->buffer_pool->d_contact_normals,
            num_pairs);
        checkCUDAError("compute_epa (EPA)", __LINE__);

        // DEBUG: Print first few distances to see what EPA is producing
        static int debug_counter = 0;
        if (debug_counter++ % 60 == 0 && num_pairs > 0) {
            gkFloat h_distances[10];
            int check_count = num_pairs < 10 ? num_pairs : 10;
            cudaMemcpy(h_distances, context->buffer_pool->d_distances,
                      sizeof(gkFloat) * check_count, cudaMemcpyDeviceToHost);
            printf("DEBUG: First %d distances after EPA: ", check_count);
            for (int i = 0; i < check_count; i++) {
                printf("%.3f ", h_distances[i]);
            }
            printf("\n");
        }

        // Step 6: Collision response with rotation and torque
        int pairBlocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        check_and_respond_with_rotation_kernel<<<pairBlocks, BLOCK_SIZE>>>(
            context->d_objects,
            context->d_collision_pairs,
            context->buffer_pool->d_distances,
            context->buffer_pool->d_witness1,
            context->buffer_pool->d_witness2,
            context->buffer_pool->d_contact_normals,
            params->collisionEpsilon,
            num_pairs,
            num_objs);
        checkCUDAError("check_and_respond_with_rotation_kernel", __LINE__);
    }

    // No sync needed - GPU->CPU transfer in get_render_data will implicitly sync

    return true;
}

bool gpu_gjk_get_render_data(GPU_GJK_Context* context, GPU_RenderData* data)
{
    if (!context || !data) return false;

    int num_objs = context->num_objects;

    // Use strided memcpy to extract positions directly from GPU_PhysicsObject array
    // cudaMemcpy2D can copy position field (12 bytes) with stride of sizeof(GPU_PhysicsObject)
    cudaMemcpy2D(context->h_positions,                    // dst
                 sizeof(Vector3f),                        // dst pitch (contiguous)
                 context->d_objects,                      // src (position is first field, offset 0)
                 sizeof(GPU_PhysicsObject),               // src pitch (stride between objects)
                 sizeof(Vector3f),                        // width (bytes to copy per row)
                 num_objs,                                // height (number of rows = objects)
                 cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy2D get_render_data", __LINE__);

    data->positions = context->h_positions;
    data->is_colliding = NULL;  // No longer tracking per-object collision flags
    data->num_objects = num_objs;
    return true;
}

// Removed gpu_gjk_reset_simulation to save memory (no initial_position/velocity needed)

bool gpu_gjk_sync_objects_to_device(GPU_GJK_Context* context)
{
    if (!context || !context->initialized) return false;

    cudaMemcpy(context->d_objects, context->h_objects,
               sizeof(GPU_PhysicsObject) * context->num_objects,
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy sync_objects_to_device", __LINE__);

    // Free host copy after syncing - no longer needed (saves 32 bytes × num_objects)
    if (context->h_objects) {
        free(context->h_objects);
        context->h_objects = NULL;
        printf("Freed host object buffer (saved %zu KB)\n",
               (sizeof(GPU_PhysicsObject) * context->num_objects) / 1024);
    }

    return true;
}

int gpu_gjk_update_collision_pairs_dynamic(GPU_GJK_Context* context, const GPU_PhysicsParams* params)
{
    if (!context || !context->initialized || !params) return 0;

    int num_objs = context->num_objects;
    float cell_size = COMPUTE_CELL_SIZE(params->boundarySize);

    // Step 1: Clear spatial grid (use cudaMemset for efficiency)
    int total_cells = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    cudaMemset(context->d_grid, 0, sizeof(GridCell) * total_cells);
    checkCUDAError("cudaMemset d_grid", __LINE__);

    // Step 2: Insert objects into spatial grid
    int objBlocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    insert_objects_kernel<<<objBlocks, BLOCK_SIZE>>>(
        context->d_objects, num_objs, context->d_grid,
        cell_size, params->boundarySize);
    checkCUDAError("insert_objects_kernel", __LINE__);

    // Step 2.5: Clear pair counts (prevents garbage values in tail threads)
    cudaMemset(context->d_pair_counts, 0, sizeof(int) * num_objs);
    checkCUDAError("cudaMemset d_pair_counts", __LINE__);

    // Step 3: Count pairs per object (Pass 1)
    count_pairs_kernel<<<objBlocks, BLOCK_SIZE>>>(
        context->d_objects, num_objs, context->d_grid,
        cell_size, params->boundarySize, context->d_pair_counts);
    checkCUDAError("count_pairs_kernel", __LINE__);

    // Step 4: Prefix sum to get write offsets
    // Note: d_pair_counts and d_pair_offsets were allocated with power-of-2 size at init
    // Use single-block scan for small arrays, multi-block for larger ones
    int n_pow2 = next_power_of_2(num_objs);

    if (num_objs <= 1024) {
        // Single-block scan (fast for small arrays)
        int blockSize = n_pow2 / 2;  // Need n/2 threads for scan
        if (blockSize > 512) blockSize = 512;

        int sharedMemSize = (n_pow2 + CONFLICT_FREE_OFFSET(n_pow2)) * sizeof(int);
        block_scan_kernel<<<1, blockSize, sharedMemSize>>>(
            num_objs, context->d_pair_offsets, context->d_pair_counts, nullptr);
        checkCUDAError("block_scan_kernel (prefix sum)", __LINE__);
    } else {
        // Multi-block recursive scan for large arrays (>1024 elements)
        // Zero-pad the tail if num_objs is not a power of 2
        if (n_pow2 > num_objs) {
            cudaMemset(context->d_pair_counts + num_objs, 0, (n_pow2 - num_objs) * sizeof(int));
            checkCUDAError("cudaMemset pad pair_counts", __LINE__);
        }

        recursive_scan(n_pow2, context->d_pair_offsets, context->d_pair_counts);
        checkCUDAError("recursive_scan (multi-block prefix sum)", __LINE__);
    }

    // Step 5: Get total pair count (last offset + last count)
    int h_last_offset, h_last_count;
    cudaMemcpy(&h_last_offset, context->d_pair_offsets + num_objs - 1, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy h_last_offset", __LINE__);
    cudaMemcpy(&h_last_count, context->d_pair_counts + num_objs - 1, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy h_last_count", __LINE__);
    context->h_total_pairs = h_last_offset + h_last_count;
    context->num_pairs = context->h_total_pairs;

    // Check if we exceed max pairs
    if (context->num_pairs > context->max_pairs) {
        fprintf(stderr, "Warning: Generated %d pairs, but max is %d. Clamping.\n",
                context->num_pairs, context->max_pairs);
        context->num_pairs = context->max_pairs;
    }

    // Step 6: Generate actual pairs (Pass 2)
    if (context->num_pairs > 0) {
        generate_pairs_kernel<<<objBlocks, BLOCK_SIZE>>>(
            context->d_objects, num_objs, context->d_grid,
            cell_size, params->boundarySize,
            context->d_pair_offsets, context->d_collision_pairs,
            context->max_pairs);
        checkCUDAError("generate_pairs_kernel", __LINE__);
    }

    return context->num_pairs;
}

#ifdef __cplusplus
}
#endif
