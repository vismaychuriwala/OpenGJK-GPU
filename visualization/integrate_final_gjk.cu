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

// GPU Physics Object - position, mass, radius only.
// Velocity lives in the ping/pong arrays so it can be swapped without touching this struct.
typedef struct {
    Vector3f position;
    float mass;
    float radius;
} GPU_PhysicsObject;

// Spatial grid cell - stores object IDs in this cell
typedef struct {
    int objects[MAX_OBJECTS_PER_CELL];
    int count;
} GridCell;

// Persistent GPU buffer pool
struct GPU_Buffer_Pool {
    int max_objects;
    int max_pairs;

    gkFloat* d_local_coords;   // [max_objects * 12 * 3]
    gkFloat* d_all_coords;     // [max_objects * 12 * 3]
    gkPolytope* d_polytopes;   // [max_objects]
    gkSimplex* d_simplices;    // [max_pairs]
    gkFloat* d_distances;      // [max_pairs]
};

// GPU GJK Context - owns all simulation state
struct GPU_GJK_Context {
    bool initialized;
    int device_id;
    int max_objects;
    int num_objects;
    int max_pairs;
    int num_pairs;

    GPU_PhysicsObject* d_objects;       // Device array [max_objects] - position/mass/radius
    GPU_PhysicsObject* h_objects;       // Host copy for init (freed after sync)

    // Ping-pong velocity buffers.
    // physics_update writes in-place to vel_ping.
    // Before collision response: cudaMemcpy vel_pong <- vel_ping (D2D, seeds pong with current state).
    // check_and_respond reads vel_ping (stable snapshot), atomicAdds impulse deltas to vel_pong.
    // After response: swap(vel_ping, vel_pong) - free pointer exchange on host.
    Vector3f* d_vel_A;                  // Device array [max_objects] - velocity buffer A
    Vector3f* d_vel_B;                  // Device array [max_objects] - velocity buffer B
    Vector3f* d_vel_ping;               // Points to A or B (current authoritative velocities)
    Vector3f* d_vel_pong;               // Points to B or A (working buffer for collision response)

    Vector3f* h_velocities;             // Host array for init (freed after sync)

    gkCollisionPair* d_collision_pairs; // Device array [max_pairs]

    Vector3f* h_positions;              // Pinned host array [max_objects] for render readback

    GPU_Buffer_Pool* buffer_pool;

    GridCell* d_grid;
    int* d_pair_counts;
    int* d_pair_offsets;
    int h_total_pairs;
};

// CUDA kernel: Physics update - integrates velocity and position, handles boundary collisions
__global__ void physics_update_kernel(GPU_PhysicsObject* objects,
                                      Vector3f* velocities,
                                      int num_objects,
                                      const GPU_PhysicsParams params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;

    // Load all needed fields into registers once — avoids repeated global loads
    float px = objects[i].position.x;
    float py = objects[i].position.y;
    float pz = objects[i].position.z;
    float radius = objects[i].radius;
    Vector3f vel = velocities[i];

    float boundary = params.boundarySize;
    float dt = params.deltaTime;

    // Update velocity with gravity
    vel.x += params.gravity.x * dt;
    vel.y += params.gravity.y * dt;
    vel.z += params.gravity.z * dt;

    // Update position
    px += vel.x * dt;
    py += vel.y * dt;
    pz += vel.z * dt;

    // Y-axis boundary
    if (py - radius < -boundary) {
        py = -boundary + radius;
        vel.y = -vel.y * params.dampingCoeff;
    }
    if (py + radius > boundary) {
        py = boundary - radius;
        vel.y = -vel.y * params.dampingCoeff;
    }

    // X-axis boundary
    if (fabsf(px) > boundary - radius) {
        vel.x = -vel.x * params.dampingCoeff;
        px = (px > 0.0f) ? boundary - radius : -boundary + radius;
    }

    // Z-axis boundary
    if (fabsf(pz) > boundary - radius) {
        vel.z = -vel.z * params.dampingCoeff;
        pz = (pz > 0.0f) ? boundary - radius : -boundary + radius;
    }

    // Write back only position (mass/radius unchanged) and updated velocity
    objects[i].position.x = px;
    objects[i].position.y = py;
    objects[i].position.z = pz;
    velocities[i] = vel;
}

// ============================================================================
// SPATIAL GRID KERNELS FOR BROAD-PHASE CULLING
// ============================================================================

__global__ void insert_objects_kernel(GPU_PhysicsObject* objects,
                                      int num_objects,
                                      GridCell* grid,
                                      float cell_size,
                                      float boundary)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    Vector3f pos = objects[obj_id].position;

    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);

    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    int cell_idx = cx + cy * SPATIAL_GRID_SIZE + cz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;

    int slot = atomicAdd(&grid[cell_idx].count, 1);
    if (slot < MAX_OBJECTS_PER_CELL) {
        grid[cell_idx].objects[slot] = obj_id;
    }
}

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

    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);

    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= SPATIAL_GRID_SIZE) continue;
                if (ny < 0 || ny >= SPATIAL_GRID_SIZE) continue;
                if (nz < 0 || nz >= SPATIAL_GRID_SIZE) continue;

                int cell_idx = nx + ny * SPATIAL_GRID_SIZE + nz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
                GridCell* cell = &grid[cell_idx];

                for (int i = 0; i < cell->count && i < MAX_OBJECTS_PER_CELL; i++) {
                    if (cell->objects[i] <= obj_id) continue;
                    count++;
                }
            }
        }
    }

    pair_counts[obj_id] = count;
}

__global__ void generate_pairs_kernel(GPU_PhysicsObject* objects,
                                      int num_objects,
                                      GridCell* grid,
                                      float cell_size,
                                      float boundary,
                                      int* pair_offsets,
                                      gkCollisionPair* pair_buffer,
                                      int max_pairs)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    Vector3f pos = objects[obj_id].position;
    int write_offset = pair_offsets[obj_id];

    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);

    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    int local_pair_idx = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= SPATIAL_GRID_SIZE) continue;
                if (ny < 0 || ny >= SPATIAL_GRID_SIZE) continue;
                if (nz < 0 || nz >= SPATIAL_GRID_SIZE) continue;

                int cell_idx = nx + ny * SPATIAL_GRID_SIZE + nz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
                GridCell* cell = &grid[cell_idx];

                for (int i = 0; i < cell->count && i < MAX_OBJECTS_PER_CELL; i++) {
                    int other_id = cell->objects[i];
                    if (other_id <= obj_id) continue;

                    int global_pair_idx = write_offset + local_pair_idx;
                    if (global_pair_idx >= max_pairs) return;

                    pair_buffer[global_pair_idx].idx1 = obj_id;
                    pair_buffer[global_pair_idx].idx2 = other_id;
                    local_pair_idx++;
                }
            }
        }
    }
}

// ============================================================================
// COLLISION DETECTION AND RESPONSE KERNELS
// ============================================================================

// Reads velocities from vel_ping (stable Jacobi snapshot).
// Writes impulse deltas to vel_pong via atomicAdd.
// vel_pong was seeded with a D2D copy of vel_ping before this kernel launched,
// so after the kernel vel_pong[i] = vel_ping[i] + sum(all impulse deltas for object i).
// The caller then swaps vel_ping and vel_pong pointers (free on host).
__global__ void check_and_respond_kernel(GPU_PhysicsObject* objects,
                                         Vector3f* vel_ping,
                                         Vector3f* vel_pong,
                                         gkCollisionPair* pairs,
                                         gkFloat* distances,
                                         float epsilon,
                                         int num_pairs,
                                         int num_objects)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;

    if (distances[p] > epsilon) return;

    int idA = pairs[p].idx1;
    int idB = pairs[p].idx2;
    if (idA < 0 || idA >= num_objects || idB < 0 || idB >= num_objects) return;

    // Load full structs into registers — one global read each, avoids re-fetching fields
    GPU_PhysicsObject objA = objects[idA];
    GPU_PhysicsObject objB = objects[idB];
    Vector3f velA = vel_ping[idA];
    Vector3f velB = vel_ping[idB];

    float dx = objB.position.x - objA.position.x;
    float dy = objB.position.y - objA.position.y;
    float dz = objB.position.z - objA.position.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    if (dist < 0.0001f) return;

    float inv_dist = 1.0f / dist;
    float nx = dx * inv_dist;
    float ny = dy * inv_dist;
    float nz = dz * inv_dist;

    float relVx = velB.x - velA.x;
    float relVy = velB.y - velA.y;
    float relVz = velB.z - velA.z;
    float velAlongNormal = relVx*nx + relVy*ny + relVz*nz;

    if (velAlongNormal > 0.0f) return;

    // Hoist reciprocals — used twice each
    float inv_massA = 1.0f / objA.mass;
    float inv_massB = 1.0f / objB.mass;
    float j = -(1.0f + RESTITUTION) * velAlongNormal / (inv_massA + inv_massB);

    float impulseX = j * nx;
    float impulseY = j * ny;
    float impulseZ = j * nz;

    atomicAdd(&vel_pong[idA].x, -impulseX * inv_massA);
    atomicAdd(&vel_pong[idA].y, -impulseY * inv_massA);
    atomicAdd(&vel_pong[idA].z, -impulseZ * inv_massA);

    atomicAdd(&vel_pong[idB].x,  impulseX * inv_massB);
    atomicAdd(&vel_pong[idB].y,  impulseY * inv_massB);
    atomicAdd(&vel_pong[idB].z,  impulseZ * inv_massB);
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

    ctx->h_objects = (GPU_PhysicsObject*)malloc(sizeof(GPU_PhysicsObject) * max_objects);
    ctx->h_velocities = (Vector3f*)malloc(sizeof(Vector3f) * max_objects);

    cudaMallocHost(&ctx->h_positions, sizeof(Vector3f) * max_objects);
    checkCUDAError("cudaMallocHost h_positions");

    cudaMalloc(&ctx->d_objects, sizeof(GPU_PhysicsObject) * max_objects);
    checkCUDAError("cudaMalloc d_objects");

    // Ping-pong velocity buffers
    cudaMalloc(&ctx->d_vel_A, sizeof(Vector3f) * max_objects);
    checkCUDAError("cudaMalloc d_vel_A");
    cudaMalloc(&ctx->d_vel_B, sizeof(Vector3f) * max_objects);
    checkCUDAError("cudaMalloc d_vel_B");
    ctx->d_vel_ping = ctx->d_vel_A;
    ctx->d_vel_pong = ctx->d_vel_B;

    cudaMalloc(&ctx->d_collision_pairs, sizeof(gkCollisionPair) * max_pairs);
    checkCUDAError("cudaMalloc d_collision_pairs");

    ctx->buffer_pool = (GPU_Buffer_Pool*)malloc(sizeof(GPU_Buffer_Pool));
    if (!ctx->buffer_pool) {
        fprintf(stderr, "Failed to allocate buffer pool\n");
        return false;
    }
    ctx->buffer_pool->max_objects = max_objects;
    ctx->buffer_pool->max_pairs = max_pairs;

    cudaMalloc(&ctx->buffer_pool->d_local_coords, sizeof(gkFloat) * max_objects * 12 * 3);
    checkCUDAError("cudaMalloc d_local_coords");
    cudaMalloc(&ctx->buffer_pool->d_all_coords, sizeof(gkFloat) * max_objects * 12 * 3);
    checkCUDAError("cudaMalloc d_all_coords");
    cudaMalloc(&ctx->buffer_pool->d_polytopes, sizeof(gkPolytope) * max_objects);
    checkCUDAError("cudaMalloc d_polytopes");
    cudaMalloc(&ctx->buffer_pool->d_simplices, sizeof(gkSimplex) * max_pairs);
    checkCUDAError("cudaMalloc d_simplices");
    cudaMalloc(&ctx->buffer_pool->d_distances, sizeof(gkFloat) * max_pairs);
    checkCUDAError("cudaMalloc d_distances");

    int total_cells = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    cudaMalloc(&ctx->d_grid, sizeof(GridCell) * total_cells);
    checkCUDAError("cudaMalloc d_grid");

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

    if (ctx->d_objects) cudaFree(ctx->d_objects);
    if (ctx->d_vel_A) cudaFree(ctx->d_vel_A);
    if (ctx->d_vel_B) cudaFree(ctx->d_vel_B);
    if (ctx->d_collision_pairs) cudaFree(ctx->d_collision_pairs);
    if (ctx->d_grid) cudaFree(ctx->d_grid);
    if (ctx->d_pair_counts) cudaFree(ctx->d_pair_counts);
    if (ctx->d_pair_offsets) cudaFree(ctx->d_pair_offsets);

    if (ctx->buffer_pool) {
        cudaFree(ctx->buffer_pool->d_local_coords);
        cudaFree(ctx->buffer_pool->d_all_coords);
        cudaFree(ctx->buffer_pool->d_polytopes);
        cudaFree(ctx->buffer_pool->d_simplices);
        cudaFree(ctx->buffer_pool->d_distances);
        free(ctx->buffer_pool);
    }

    if (ctx->h_positions) cudaFreeHost(ctx->h_positions);
    if (ctx->h_objects) { free(ctx->h_objects); ctx->h_objects = NULL; }
    if (ctx->h_velocities) { free(ctx->h_velocities); ctx->h_velocities = NULL; }

    free(ctx);
    *context = NULL;

    printf("GPU simulation context cleaned up\n");
}

bool gpu_gjk_register_object(GPU_GJK_Context* context, int object_id,
                             const GJK_Shape* shape,
                             Vector3f position, Vector3f velocity,
                             float mass, float radius)
{
    if (!context || !shape) return false;
    if (object_id < 0 || object_id >= context->max_objects) return false;

    context->h_objects[object_id].position = position;
    context->h_objects[object_id].mass = mass;
    context->h_objects[object_id].radius = radius;
    context->h_velocities[object_id] = velocity;

    int num_verts = shape->num_vertices;
    gkFloat* h_local_verts = (gkFloat*)malloc(sizeof(gkFloat) * num_verts * 3);
    for (int i = 0; i < num_verts; i++) {
        h_local_verts[i * 3 + 0] = shape->vertices[i].x;
        h_local_verts[i * 3 + 1] = shape->vertices[i].y;
        h_local_verts[i * 3 + 2] = shape->vertices[i].z;
    }

    int coord_offset = object_id * 12 * 3;
    cudaMemcpy(context->buffer_pool->d_local_coords + coord_offset,
               h_local_verts,
               sizeof(gkFloat) * num_verts * 3,
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy register object vertices", __LINE__);
    free(h_local_verts);

    if (object_id + 1 > context->num_objects)
        context->num_objects = object_id + 1;

    printf("Registered GPU object %d: pos=(%.2f, %.2f, %.2f), mass=%.2f, radius=%.2f\n",
           object_id, position.x, position.y, position.z, mass, radius);

    return true;
}

__global__ void init_unique_polytopes_kernel(gkFloat* all_coords,
                                             gkPolytope* polytopes,
                                             int num_objects,
                                             int num_vertices_per_shape)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;
    polytopes[obj_id].numpoints = num_vertices_per_shape;
    polytopes[obj_id].coord = all_coords + (obj_id * 12 * 3);
}

__global__ void transform_to_world_kernel(GPU_PhysicsObject* objects,
                                          gkFloat* local_coords,
                                          gkFloat* world_coords,
                                          int num_objects)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    // Load position once into registers — old per-vertex launch re-read this 12x per object
    float px = objects[obj_id].position.x;
    float py = objects[obj_id].position.y;
    float pz = objects[obj_id].position.z;

    int base = obj_id * 12 * 3;
    for (int v = 0; v < 12; v++) {
        int offset = base + v * 3;
        world_coords[offset + 0] = local_coords[offset + 0] + px;
        world_coords[offset + 1] = local_coords[offset + 1] + py;
        world_coords[offset + 2] = local_coords[offset + 2] + pz;
    }
}

bool gpu_gjk_step_simulation(GPU_GJK_Context* context, const GPU_PhysicsParams* params)
{
    if (!context || !context->initialized || !params) return false;

    int num_objs = context->num_objects;
    int num_pairs = context->num_pairs;
    int objBlocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Step 1: Physics update - reads/writes vel_ping in-place, updates positions
    physics_update_kernel<<<objBlocks, BLOCK_SIZE>>>(
        context->d_objects, context->d_vel_ping, num_objs, *params);
    checkCUDAError("physics_update_kernel", __LINE__);

    // Step 2: Transform local vertices to world space (one thread per object)
    int transformBlocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transform_to_world_kernel<<<transformBlocks, BLOCK_SIZE>>>(
        context->d_objects,
        context->buffer_pool->d_local_coords,
        context->buffer_pool->d_all_coords,
        num_objs);
    checkCUDAError("transform_to_world_kernel", __LINE__);

    // Step 3: Batched GJK collision detection
    const int threadsPerCollision = 16;
    const int collisionsPerBlock = 2;
    int gjkBlocks = (num_pairs + collisionsPerBlock - 1) / collisionsPerBlock;

    if (num_pairs > 0) {
        compute_minimum_distance_indexed_kernel<<<gjkBlocks, threadsPerCollision * collisionsPerBlock>>>(
            context->buffer_pool->d_polytopes,
            context->d_collision_pairs,
            context->buffer_pool->d_simplices,
            context->buffer_pool->d_distances,
            num_pairs);
        checkCUDAError("compute_minimum_distance_indexed_kernel (GJK)", __LINE__);

        // Step 4: Ping-pong collision response
        // Seed vel_pong with current velocities (D2D copy, fast)
        cudaMemcpy(context->d_vel_pong, context->d_vel_ping,
                   sizeof(Vector3f) * num_objs, cudaMemcpyDeviceToDevice);
        checkCUDAError("cudaMemcpy vel_pong <- vel_ping", __LINE__);

        // All threads read stable vel_ping; atomicAdd impulse deltas onto vel_pong
        int pairBlocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        check_and_respond_kernel<<<pairBlocks, BLOCK_SIZE>>>(
            context->d_objects,
            context->d_vel_ping,
            context->d_vel_pong,
            context->d_collision_pairs,
            context->buffer_pool->d_distances,
            params->collisionEpsilon,
            num_pairs,
            num_objs);
        checkCUDAError("check_and_respond_kernel", __LINE__);

        // Swap: pong (with impulses applied) becomes ping for next step
        Vector3f* tmp = context->d_vel_ping;
        context->d_vel_ping = context->d_vel_pong;
        context->d_vel_pong = tmp;
    }

    return true;
}

bool gpu_gjk_get_render_data(GPU_GJK_Context* context, GPU_RenderData* data)
{
    if (!context || !data) return false;

    int num_objs = context->num_objects;

    // position is still the first field of GPU_PhysicsObject, stride unchanged
    cudaMemcpy2D(context->h_positions,
                 sizeof(Vector3f),
                 context->d_objects,
                 sizeof(GPU_PhysicsObject),
                 sizeof(Vector3f),
                 num_objs,
                 cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy2D get_render_data", __LINE__);

    data->positions = context->h_positions;
    data->is_colliding = NULL;
    data->num_objects = num_objs;
    return true;
}

bool gpu_gjk_sync_objects_to_device(GPU_GJK_Context* context)
{
    if (!context || !context->initialized) return false;

    cudaMemcpy(context->d_objects, context->h_objects,
               sizeof(GPU_PhysicsObject) * context->num_objects,
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy sync_objects_to_device", __LINE__);

    // Both buffers initialized to the same starting velocities so the first
    // swap doesn't introduce a stale frame
    cudaMemcpy(context->d_vel_A, context->h_velocities,
               sizeof(Vector3f) * context->num_objects, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy vel_A init", __LINE__);
    cudaMemcpy(context->d_vel_B, context->h_velocities,
               sizeof(Vector3f) * context->num_objects, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy vel_B init", __LINE__);

    int objBlocks = (context->num_objects + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_unique_polytopes_kernel<<<objBlocks, BLOCK_SIZE>>>(
        context->buffer_pool->d_all_coords,
        context->buffer_pool->d_polytopes,
        context->num_objects,
        12);
    checkCUDAError("init_unique_polytopes_kernel", __LINE__);
    cudaDeviceSynchronize();

    free(context->h_objects);   context->h_objects = NULL;
    free(context->h_velocities); context->h_velocities = NULL;

    return true;
}

int gpu_gjk_update_collision_pairs_dynamic(GPU_GJK_Context* context, const GPU_PhysicsParams* params)
{
    if (!context || !context->initialized || !params) return 0;

    int num_objs = context->num_objects;
    float cell_size = COMPUTE_CELL_SIZE(params->boundarySize);

    int total_cells = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    cudaMemset(context->d_grid, 0, sizeof(GridCell) * total_cells);
    checkCUDAError("cudaMemset d_grid", __LINE__);

    int objBlocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    insert_objects_kernel<<<objBlocks, BLOCK_SIZE>>>(
        context->d_objects, num_objs, context->d_grid,
        cell_size, params->boundarySize);
    checkCUDAError("insert_objects_kernel", __LINE__);

    cudaMemset(context->d_pair_counts, 0, sizeof(int) * num_objs);
    checkCUDAError("cudaMemset d_pair_counts", __LINE__);

    count_pairs_kernel<<<objBlocks, BLOCK_SIZE>>>(
        context->d_objects, num_objs, context->d_grid,
        cell_size, params->boundarySize, context->d_pair_counts);
    checkCUDAError("count_pairs_kernel", __LINE__);

    int n_pow2 = next_power_of_2(num_objs);

    if (num_objs <= 1024) {
        int blockSize = n_pow2 / 2;
        if (blockSize > 512) blockSize = 512;
        int sharedMemSize = (n_pow2 + CONFLICT_FREE_OFFSET(n_pow2)) * sizeof(int);
        block_scan_kernel<<<1, blockSize, sharedMemSize>>>(
            num_objs, context->d_pair_offsets, context->d_pair_counts, nullptr);
        checkCUDAError("block_scan_kernel (prefix sum)", __LINE__);
    } else {
        if (n_pow2 > num_objs) {
            cudaMemset(context->d_pair_counts + num_objs, 0, (n_pow2 - num_objs) * sizeof(int));
            checkCUDAError("cudaMemset pad pair_counts", __LINE__);
        }
        recursive_scan(n_pow2, context->d_pair_offsets, context->d_pair_counts);
        checkCUDAError("recursive_scan (multi-block prefix sum)", __LINE__);
    }

    int h_last_offset, h_last_count;
    cudaMemcpy(&h_last_offset, context->d_pair_offsets + num_objs - 1, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy h_last_offset", __LINE__);
    cudaMemcpy(&h_last_count, context->d_pair_counts + num_objs - 1, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy h_last_count", __LINE__);
    context->h_total_pairs = h_last_offset + h_last_count;
    context->num_pairs = context->h_total_pairs;

    if (context->num_pairs > context->max_pairs) {
        fprintf(stderr, "Warning: Generated %d pairs, but max is %d. Clamping.\n",
                context->num_pairs, context->max_pairs);
        context->num_pairs = context->max_pairs;
    }

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
