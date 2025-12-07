// integrate_final_gjk.cu - GPU-owned physics simulation
#include "gpu_gjk_interface.h"
#include "sim_config.h"
#include "../GJK/gpu/warpParallelGJK.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU Physics Object (mirrored on device)
typedef struct {
    Vector3f position;
    Vector3f velocity;
    Vector3f initial_position;
    Vector3f initial_velocity;
    float mass;
    float radius;
} GPU_PhysicsObject;

// GPU Shape Data (local vertices on device)
typedef struct {
    gkFloat* d_local_vertices;  // Device pointer to local-space vertices
    int num_vertices;
} GPU_ShapeData;

// Persistent GPU buffer pool
struct GPU_Buffer_Pool {
    int max_pairs;

    // Coordinate buffers (world-space vertices for all shapes)
    gkFloat* d_all_coords;     // Flat array for all shape coordinates [max_objects * max_vertices_per_shape * 3]

    // Polytope, simplex, and distance buffers for parallel collision checks
    gkPolytope* d_polytopes1;  // Device array [max_pairs] - first polytope of each pair
    gkPolytope* d_polytopes2;  // Device array [max_pairs] - second polytope of each pair
    gkSimplex* d_simplices;    // Device array [max_pairs]
    gkFloat* d_distances;      // Device array [max_pairs]
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
    GPU_PhysicsObject* h_objects;     // Host copy for initialization/reset
    GPU_ShapeData* h_shapes;          // Host array of shape metadata
    GPU_ShapeData* d_shapes;          // Device array of shape metadata

    // Collision pair indices
    int* d_collision_pairs;           // Device array [num_pairs * 2]

    // Render data (pinned host memory for fast GPU->CPU transfer)
    Vector3f* h_positions;            // Pinned host array [num_objects]

    // Persistent buffer pool for collision detection
    GPU_Buffer_Pool* buffer_pool;
};

// CUDA kernel: Physics update for all objects
__global__ void physics_update_kernel(GPU_PhysicsObject* objects,
                                     int num_objects,
                                     const GPU_PhysicsParams params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;

    GPU_PhysicsObject* obj = &objects[i];

    // Update velocity with gravity
    obj->velocity.x += params.gravity.x * params.deltaTime;
    obj->velocity.y += params.gravity.y * params.deltaTime;
    obj->velocity.z += params.gravity.z * params.deltaTime;

    // Update position
    obj->position.x += obj->velocity.x * params.deltaTime;
    obj->position.y += obj->velocity.y * params.deltaTime;
    obj->position.z += obj->velocity.z * params.deltaTime;

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

// CUDA kernel: Fused collision check and response (eliminates intermediate bool array)
// Checks distance against epsilon and immediately applies collision response if colliding
__global__ void check_and_respond_kernel(GPU_PhysicsObject* objects,
                                         int* pairs,
                                         gkFloat* distances,
                                         float epsilon,
                                         int num_pairs)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;

    // Inline collision check - early exit if no collision
    if (fabsf(distances[p]) > epsilon) return;

    int idA = pairs[p * 2 + 0];
    int idB = pairs[p * 2 + 1];

    GPU_PhysicsObject* obj1 = &objects[idA];
    GPU_PhysicsObject* obj2 = &objects[idB];

    // Calculate collision normal (obj2 - obj1, same as CPU code)
    float dx = obj2->position.x - obj1->position.x;
    float dy = obj2->position.y - obj1->position.y;
    float dz = obj2->position.z - obj1->position.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    if (dist < 0.0001f) return;  // Avoid division by zero

    // Normalize collision normal
    float nx = dx / dist;
    float ny = dy / dist;
    float nz = dz / dist;

    // Separate objects to prevent sticking
    float overlap = (obj1->radius + obj2->radius) - dist;
    if (overlap > 0.0f) {
        obj1->position.x -= nx * overlap * 0.5f;
        obj1->position.y -= ny * overlap * 0.5f;
        obj1->position.z -= nz * overlap * 0.5f;

        obj2->position.x += nx * overlap * 0.5f;
        obj2->position.y += ny * overlap * 0.5f;
        obj2->position.z += nz * overlap * 0.5f;
    }

    // Relative velocity
    float relVx = obj2->velocity.x - obj1->velocity.x;
    float relVy = obj2->velocity.y - obj1->velocity.y;
    float relVz = obj2->velocity.z - obj1->velocity.z;

    // Velocity along normal
    float velAlongNormal = relVx * nx + relVy * ny + relVz * nz;

    // Don't resolve if objects are moving apart
    if (velAlongNormal > 0.0f) return;

    // Collision impulse (elastic collision)
    float j = -(1.0f + RESTITUTION) * velAlongNormal;
    j /= (1.0f / obj1->mass + 1.0f / obj2->mass);

    // Apply impulse
    float impulseX = j * nx;
    float impulseY = j * ny;
    float impulseZ = j * nz;

    obj1->velocity.x -= impulseX / obj1->mass;
    obj1->velocity.y -= impulseY / obj1->mass;
    obj1->velocity.z -= impulseZ / obj1->mass;

    obj2->velocity.x += impulseX / obj2->mass;
    obj2->velocity.y += impulseY / obj2->mass;
    obj2->velocity.z += impulseZ / obj2->mass;
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
    ctx->h_shapes = (GPU_ShapeData*)malloc(sizeof(GPU_ShapeData) * max_objects);
    memset(ctx->h_shapes, 0, sizeof(GPU_ShapeData) * max_objects);

    // Allocate pinned host memory for fast GPU->CPU transfer
    cudaMallocHost(&ctx->h_positions, sizeof(Vector3f) * max_objects);

    // Allocate GPU memory for physics objects
    cudaMalloc(&ctx->d_objects, sizeof(GPU_PhysicsObject) * max_objects);
    cudaMalloc(&ctx->d_shapes, sizeof(GPU_ShapeData) * max_objects);
    cudaMalloc(&ctx->d_collision_pairs, sizeof(int) * max_pairs * 2);

    // Allocate buffer pool
    ctx->buffer_pool = (GPU_Buffer_Pool*)malloc(sizeof(GPU_Buffer_Pool));
    ctx->buffer_pool->max_pairs = max_pairs;

    // Allocate persistent GPU buffers for collision detection
    cudaMalloc(&ctx->buffer_pool->d_all_coords, sizeof(gkFloat) * max_objects * 12 * 3);
    cudaMalloc(&ctx->buffer_pool->d_polytopes1, sizeof(gkPolytope) * max_pairs);
    cudaMalloc(&ctx->buffer_pool->d_polytopes2, sizeof(gkPolytope) * max_pairs);
    cudaMalloc(&ctx->buffer_pool->d_simplices, sizeof(gkSimplex) * max_pairs);
    cudaMalloc(&ctx->buffer_pool->d_distances, sizeof(gkFloat) * max_pairs);

    ctx->initialized = true;
    printf("GPU simulation context initialized (max_objects=%d, max_pairs=%d)\n",
           max_objects, max_pairs);

    return true;
}

void gpu_gjk_cleanup(GPU_GJK_Context** context)
{
    if (!context || !*context) return;

    GPU_GJK_Context* ctx = *context;

    // Free device memory
    if (ctx->d_objects) cudaFree(ctx->d_objects);
    if (ctx->d_shapes) cudaFree(ctx->d_shapes);
    if (ctx->d_collision_pairs) cudaFree(ctx->d_collision_pairs);

    // Free buffer pool
    if (ctx->buffer_pool) {
        cudaFree(ctx->buffer_pool->d_all_coords);
        cudaFree(ctx->buffer_pool->d_polytopes1);
        cudaFree(ctx->buffer_pool->d_polytopes2);
        cudaFree(ctx->buffer_pool->d_simplices);
        cudaFree(ctx->buffer_pool->d_distances);
        free(ctx->buffer_pool);
    }

    // Free shape local vertices
    if (ctx->h_shapes) {
        for (int i = 0; i < ctx->num_objects; i++) {
            if (ctx->h_shapes[i].d_local_vertices) {
                cudaFree(ctx->h_shapes[i].d_local_vertices);
            }
        }
        free(ctx->h_shapes);
    }

    // Free pinned host memory
    if (ctx->h_positions) cudaFreeHost(ctx->h_positions);

    // Free host memory
    if (ctx->h_objects) free(ctx->h_objects);

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

    // Store physics object on host
    context->h_objects[object_id].position = position;
    context->h_objects[object_id].velocity = velocity;
    context->h_objects[object_id].initial_position = position;
    context->h_objects[object_id].initial_velocity = velocity;
    context->h_objects[object_id].mass = mass;
    context->h_objects[object_id].radius = radius;

    // Store shape local vertices on GPU
    int num_verts = shape->num_vertices;
    gkFloat* d_local_verts;
    cudaMalloc(&d_local_verts, sizeof(gkFloat) * num_verts * 3);

    gkFloat* h_local_verts = (gkFloat*)malloc(sizeof(gkFloat) * num_verts * 3);
    for (int i = 0; i < num_verts; i++) {
        h_local_verts[i * 3 + 0] = shape->vertices[i].x;
        h_local_verts[i * 3 + 1] = shape->vertices[i].y;
        h_local_verts[i * 3 + 2] = shape->vertices[i].z;
    }

    cudaMemcpy(d_local_verts, h_local_verts,
               sizeof(gkFloat) * num_verts * 3,
               cudaMemcpyHostToDevice);
    free(h_local_verts);

    context->h_shapes[object_id].d_local_vertices = d_local_verts;
    context->h_shapes[object_id].num_vertices = num_verts;

    if (object_id + 1 > context->num_objects) {
        context->num_objects = object_id + 1;
    }

    printf("Registered GPU object %d: pos=(%.2f, %.2f, %.2f), mass=%.2f, radius=%.2f\n",
           object_id, position.x, position.y, position.z, mass, radius);

    return true;
}

// CUDA kernel: Initialize polytopes ONCE at startup (called from gpu_gjk_set_collision_pairs)
// GJK initializes support cache (s[], s_idx) itself - we only set numpoints and coord pointer
__global__ void init_polytopes_once_kernel(GPU_ShapeData* shapes,
                                           int* pairs,
                                           gkFloat* all_coords,
                                           gkPolytope* polytopes1,
                                           gkPolytope* polytopes2,
                                           int num_pairs)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;

    int idA = pairs[p * 2 + 0];
    int idB = pairs[p * 2 + 1];

    // Setup polytope A (first in pair) - only geometry, GJK handles support cache
    polytopes1[p].numpoints = shapes[idA].num_vertices;
    polytopes1[p].coord = all_coords + (idA * 12 * 3);

    // Setup polytope B (second in pair)
    polytopes2[p].numpoints = shapes[idB].num_vertices;
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

    // Copy initial object states to GPU
    cudaMemcpy(context->d_objects, context->h_objects,
               sizeof(GPU_PhysicsObject) * context->num_objects,
               cudaMemcpyHostToDevice);

    // Copy shape metadata to GPU
    cudaMemcpy(context->d_shapes, context->h_shapes,
               sizeof(GPU_ShapeData) * context->num_objects,
               cudaMemcpyHostToDevice);

    // Initialize polytopes ONCE (no longer done per-frame!)
    int pairBlocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_polytopes_once_kernel<<<pairBlocks, BLOCK_SIZE>>>(
        context->d_shapes, context->d_collision_pairs,
        context->buffer_pool->d_all_coords,
        context->buffer_pool->d_polytopes1,
        context->buffer_pool->d_polytopes2,
        num_pairs);
    cudaDeviceSynchronize();

    printf("Set %d collision pairs on GPU (polytopes initialized)\n", num_pairs);
    return true;
}

// CUDA kernel: Transform local vertices to world space for all objects
// Optimized: 1D grid processing all vertices across all objects
__global__ void transform_to_world_kernel(GPU_ShapeData* shapes,
                                         GPU_PhysicsObject* objects,
                                         gkFloat* world_coords,
                                         int num_objects)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each object has max 12 vertices, compute which object and vertex this thread handles
    int obj_id = tid / 12;
    int vert_id = tid % 12;

    if (obj_id >= num_objects) return;

    int num_verts = shapes[obj_id].num_vertices;
    if (vert_id >= num_verts) return;

    gkFloat* local = shapes[obj_id].d_local_vertices;
    gkFloat* world = world_coords + (obj_id * 12 * 3);

    // Transform: world = local + position
    world[vert_id * 3 + 0] = local[vert_id * 3 + 0] + objects[obj_id].position.x;
    world[vert_id * 3 + 1] = local[vert_id * 3 + 1] + objects[obj_id].position.y;
    world[vert_id * 3 + 2] = local[vert_id * 3 + 2] + objects[obj_id].position.z;
}

bool gpu_gjk_step_simulation(GPU_GJK_Context* context, const GPU_PhysicsParams* params)
{
    if (!context || !context->initialized || !params) return false;

    int num_objs = context->num_objects;
    int num_pairs = context->num_pairs;

    // Step 1: Physics update (parallel per object)
    int objBlocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    physics_update_kernel<<<objBlocks, BLOCK_SIZE>>>(context->d_objects, num_objs, *params);

    // Step 2: Transform local vertices to world space (optimized 1D grid)
    // Total work: num_objs * 12 vertices per object
    int total_vertices = num_objs * 12;
    int transformBlocks = (total_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transform_to_world_kernel<<<transformBlocks, BLOCK_SIZE>>>(
        context->d_shapes, context->d_objects,
        context->buffer_pool->d_all_coords, num_objs);

    // Step 3: Batched GJK collision detection (polytopes already initialized at startup!)
    // Polytopes point to d_all_coords which was just updated with new positions
    // Each collision uses 16 threads (half-warp), so use 32 threads per block for 2 pairs
    const int threadsPerCollision = 16;
    const int collisionsPerBlock = 2;
    const int threadsPerBlock = threadsPerCollision * collisionsPerBlock;  // 32
    int gjkBlocks = (num_pairs + collisionsPerBlock - 1) / collisionsPerBlock;

    compute_minimum_distance_warp_parallel<<<gjkBlocks, threadsPerBlock>>>(
        context->buffer_pool->d_polytopes1,  // First polytopes [num_pairs]
        context->buffer_pool->d_polytopes2,  // Second polytopes [num_pairs]
        context->buffer_pool->d_simplices,
        context->buffer_pool->d_distances,
        num_pairs);

    // Step 4: Fused collision check + response (eliminates separate check kernel and bool array)
    int pairBlocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_and_respond_kernel<<<pairBlocks, BLOCK_SIZE>>>(
        context->d_objects, context->d_collision_pairs,
        context->buffer_pool->d_distances,
        params->collisionEpsilon,
        num_pairs);

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

    data->positions = context->h_positions;
    data->is_colliding = NULL;  // No longer tracking per-object collision flags
    data->num_objects = num_objs;

    return true;
}

bool gpu_gjk_reset_simulation(GPU_GJK_Context* context)
{
    if (!context || !context->initialized) return false;

    // Reset objects to initial state on host
    for (int i = 0; i < context->num_objects; i++) {
        context->h_objects[i].position = context->h_objects[i].initial_position;
        context->h_objects[i].velocity = context->h_objects[i].initial_velocity;
    }

    // Copy to GPU
    cudaMemcpy(context->d_objects, context->h_objects,
               sizeof(GPU_PhysicsObject) * context->num_objects,
               cudaMemcpyHostToDevice);

    printf("GPU simulation reset to initial state\n");
    return true;
}

#ifdef __cplusplus
}
#endif
