// integrate_final_gjk.cu
// GL types must be defined before cuda_gl_interop.h
#ifdef _WIN32
#include <windows.h>
#include <GL/GL.h>
#endif
#include <cuda_gl_interop.h>

#include "sim_api.h"
#include "sim_config.h"
#include "../GJK/gpu/openGJK.h"
#include "utils/scan_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Error checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Global device arrays
// ============================================================================

static int g_num_objects  = 0;
static int g_num_pairs    = 0;
static int g_max_pairs    = 0;
static int g_total_verts  = 0;

// Physics state
static float4* d_positions;    // xyz = world pos, w = bounding_radius (constant)
static float4* d_vel_buf[2];   // xyz = velocity, w = mass (constant)
static float4* d_quats;        // rotation quaternion (identity for now)
static float3* d_scales;       // per-axis scale (constant)
static int     g_vel_ping = 0;
static int     g_vel_pong = 1;

// GJK vertex pool
static float3* d_verts_local;  // flat pool, local unit-scale
static float3* d_verts_world;  // flat pool, world-space (rebuilt each step)
static int*    d_vert_offsets; // per-object start in pool
static int*    d_vert_counts;  // per-object vertex count

// GJK working memory
static gkPolytope* d_polytopes;
static gkSimplex*  d_simplices;
static gkFloat*    d_distances;

// Broad-phase
static int*              d_grid_counts;
static int*              d_grid_objects;
static gkCollisionPair*  d_collision_pairs;
static int*              d_pair_counts;
static int*              d_pair_offsets;

// GL interop
static cudaGraphicsResource_t g_gl_pos_resource = nullptr;

// ============================================================================
// Device helpers
// ============================================================================

__device__ static inline float3 quat_rotate(float4 q, float3 v) {
    // q = (qx, qy, qz, qw)
    float3 u = make_float3(q.x, q.y, q.z);
    float  s = q.w;
    float  dot_uv = u.x*v.x + u.y*v.y + u.z*v.z;
    float3 cross_uv = make_float3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
    return make_float3(
        2.0f*dot_uv*u.x + (2.0f*s*s - 1.0f)*v.x + 2.0f*s*cross_uv.x,
        2.0f*dot_uv*u.y + (2.0f*s*s - 1.0f)*v.y + 2.0f*s*cross_uv.y,
        2.0f*dot_uv*u.z + (2.0f*s*s - 1.0f)*v.z + 2.0f*s*cross_uv.z
    );
}

// ============================================================================
// Kernels
// ============================================================================

__global__ void transform_to_world_kernel(
    const float4* positions,
    const float4* quats,
    const float3* scales,
    const float3* verts_local,
    float3*       verts_world,
    const int*    vert_offsets,
    const int*    vert_counts,
    int           num_objects)
{
    int obj = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj >= num_objects) return;

    float3 pos   = make_float3(positions[obj].x, positions[obj].y, positions[obj].z);
    float3 sc    = scales[obj];
    float4 quat  = quats[obj];

    int offset = vert_offsets[obj];
    int count  = vert_counts[obj];

    for (int v = 0; v < count; v++) {
        float3 lv = verts_local[offset + v];
        lv.x *= sc.x; lv.y *= sc.y; lv.z *= sc.z;
        float3 rv = quat_rotate(quat, lv);
        verts_world[offset + v] = make_float3(rv.x + pos.x, rv.y + pos.y, rv.z + pos.z);
    }
}

__global__ void physics_update_kernel(
    float4*      positions,
    float4*      velocities,
    const float3* scales,
    int           num_objects,
    float         dt,
    float         gravity_y,
    float         damping,
    float         boundary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;

    float4 pos = positions[i];   // w = bounding_radius
    float4 vel = velocities[i];  // w = mass

    float br = pos.w;  // world bounding radius (precomputed at init)

    vel.y += gravity_y * dt;
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    if (pos.y - br < -boundary) { pos.y = -boundary + br; vel.y = -vel.y * damping; }
    if (pos.y + br >  boundary) { pos.y =  boundary - br; vel.y = -vel.y * damping; }
    if (pos.x - br < -boundary || pos.x + br > boundary) {
        vel.x = -vel.x * damping;
        pos.x = (pos.x > 0.0f) ? boundary - br : -boundary + br;
    }
    if (pos.z - br < -boundary || pos.z + br > boundary) {
        vel.z = -vel.z * damping;
        pos.z = (pos.z > 0.0f) ? boundary - br : -boundary + br;
    }

    positions[i]  = pos;
    velocities[i] = vel;
}

__global__ void insert_objects_kernel(
    const float4* positions,
    int           num_objects,
    int*          grid_counts,
    int*          grid_objects,
    float         cell_size,
    float         boundary)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    float4 pos = positions[obj_id];
    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);
    cx = max(0, min(cx, SPATIAL_GRID_SIZE - 1));
    cy = max(0, min(cy, SPATIAL_GRID_SIZE - 1));
    cz = max(0, min(cz, SPATIAL_GRID_SIZE - 1));

    int cell_idx = cx + cy * SPATIAL_GRID_SIZE + cz * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    int slot = atomicAdd(&grid_counts[cell_idx], 1);
    if (slot < MAX_OBJECTS_PER_CELL)
        grid_objects[cell_idx * MAX_OBJECTS_PER_CELL + slot] = obj_id;
}

__global__ void count_pairs_kernel(
    int        num_objects,
    const int* grid_counts,
    const int* grid_objects,
    const float4* positions,
    float         cell_size,
    float         boundary,
    int*          pair_counts)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    float4 pos = positions[obj_id];
    int cx = max(0, min((int)floorf((pos.x + boundary) / cell_size), SPATIAL_GRID_SIZE - 1));
    int cy = max(0, min((int)floorf((pos.y + boundary) / cell_size), SPATIAL_GRID_SIZE - 1));
    int cz = max(0, min((int)floorf((pos.z + boundary) / cell_size), SPATIAL_GRID_SIZE - 1));

    int count = 0;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx+dx, ny = cy+dy, nz = cz+dz;
        if (nx<0||nx>=SPATIAL_GRID_SIZE||ny<0||ny>=SPATIAL_GRID_SIZE||nz<0||nz>=SPATIAL_GRID_SIZE) continue;
        int cell_idx = nx + ny*SPATIAL_GRID_SIZE + nz*SPATIAL_GRID_SIZE*SPATIAL_GRID_SIZE;
        int cc = min(grid_counts[cell_idx], MAX_OBJECTS_PER_CELL);
        for (int i = 0; i < cc; i++)
            if (grid_objects[cell_idx * MAX_OBJECTS_PER_CELL + i] > obj_id) count++;
    }
    pair_counts[obj_id] = count;
}

__global__ void generate_pairs_kernel(
    int            num_objects,
    const int*     grid_counts,
    const int*     grid_objects,
    const float4*  positions,
    float          cell_size,
    float          boundary,
    const int*     pair_offsets,
    gkCollisionPair* pair_buffer,
    int            max_pairs)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    float4 pos = positions[obj_id];
    int cx = max(0, min((int)floorf((pos.x + boundary) / cell_size), SPATIAL_GRID_SIZE - 1));
    int cy = max(0, min((int)floorf((pos.y + boundary) / cell_size), SPATIAL_GRID_SIZE - 1));
    int cz = max(0, min((int)floorf((pos.z + boundary) / cell_size), SPATIAL_GRID_SIZE - 1));

    int write_offset = pair_offsets[obj_id];
    int local_idx = 0;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx+dx, ny = cy+dy, nz = cz+dz;
        if (nx<0||nx>=SPATIAL_GRID_SIZE||ny<0||ny>=SPATIAL_GRID_SIZE||nz<0||nz>=SPATIAL_GRID_SIZE) continue;
        int cell_idx = nx + ny*SPATIAL_GRID_SIZE + nz*SPATIAL_GRID_SIZE*SPATIAL_GRID_SIZE;
        int cc = min(grid_counts[cell_idx], MAX_OBJECTS_PER_CELL);
        for (int i = 0; i < cc; i++) {
            int other = grid_objects[cell_idx * MAX_OBJECTS_PER_CELL + i];
            if (other <= obj_id) continue;
            int gidx = write_offset + local_idx;
            if (gidx >= max_pairs) return;
            pair_buffer[gidx].idx1 = obj_id;
            pair_buffer[gidx].idx2 = other;
            local_idx++;
        }
    }
}

__global__ void collision_response_kernel(
    float4*                positions,
    float4*                vel_ping,
    float4*                vel_pong,
    const gkCollisionPair* pairs,
    const gkFloat*         distances,
    float                  epsilon,
    int                    num_pairs,
    int                    num_objects)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;
    if (distances[p] > epsilon) return;

    int idA = pairs[p].idx1;
    int idB = pairs[p].idx2;
    if (idA < 0 || idA >= num_objects || idB < 0 || idB >= num_objects) return;

    float4 posA = positions[idA], posB = positions[idB];
    float4 velA = vel_ping[idA],  velB = vel_ping[idB];

    float dx = posB.x - posA.x, dy = posB.y - posA.y, dz = posB.z - posA.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    if (dist < 0.0001f) return;

    float inv_d = 1.0f / dist;
    float nx = dx*inv_d, ny = dy*inv_d, nz = dz*inv_d;

    float rvx = velB.x - velA.x, rvy = velB.y - velA.y, rvz = velB.z - velA.z;
    float vn = rvx*nx + rvy*ny + rvz*nz;
    if (vn > 0.0f) return;

    float inv_mA = 1.0f / velA.w;  // mass stored in .w
    float inv_mB = 1.0f / velB.w;
    float j = -(1.0f + RESTITUTION) * vn / (inv_mA + inv_mB);

    atomicAdd(&vel_pong[idA].x, -j * nx * inv_mA);
    atomicAdd(&vel_pong[idA].y, -j * ny * inv_mA);
    atomicAdd(&vel_pong[idA].z, -j * nz * inv_mA);
    atomicAdd(&vel_pong[idB].x,  j * nx * inv_mB);
    atomicAdd(&vel_pong[idB].y,  j * ny * inv_mB);
    atomicAdd(&vel_pong[idB].z,  j * nz * inv_mB);
}

__global__ void init_polytopes_kernel(
    gkPolytope*   polytopes,
    float3*       verts_world,
    const int*    vert_offsets,
    const int*    vert_counts,
    int           num_objects)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;
    polytopes[i].numpoints = vert_counts[i];
    polytopes[i].coord     = (gkFloat*)(verts_world + vert_offsets[i]);
    polytopes[i].s[0]      = 0; polytopes[i].s[1] = 0; polytopes[i].s[2] = 0;
    polytopes[i].s_idx     = 0;
}

__global__ void copy_positions_to_gl_kernel(float4* gl_buf, const float4* positions, int num_objects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;
    gl_buf[i] = positions[i];  // xyz = world pos, w = bounding_radius (unused by GL)
}

// ============================================================================
// Public API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

bool sim_init(const ObjectInitData* objects, int num_objects, unsigned int gl_pos_buffer) {
    g_num_objects = num_objects;
    g_max_pairs   = num_objects * 50;

    // Count total GJK verts
    g_total_verts = 0;
    for (int i = 0; i < num_objects; i++)
        g_total_verts += objects[i].num_gjk_verts;

    // Build CPU-side vert offsets
    int* h_vert_offsets = (int*)malloc(num_objects * sizeof(int));
    int* h_vert_counts  = (int*)malloc(num_objects * sizeof(int));
    float* h_verts_local = (float*)malloc(g_total_verts * 3 * sizeof(float));

    int cursor = 0;
    for (int i = 0; i < num_objects; i++) {
        h_vert_offsets[i] = cursor;
        h_vert_counts[i]  = objects[i].num_gjk_verts;
        memcpy(h_verts_local + cursor * 3, objects[i].gjk_verts,
               objects[i].num_gjk_verts * 3 * sizeof(float));
        cursor += objects[i].num_gjk_verts;
    }

    // Build CPU-side physics arrays
    float4* h_positions = (float4*)malloc(num_objects * sizeof(float4));
    float4* h_velocities = (float4*)malloc(num_objects * sizeof(float4));
    float4* h_quats      = (float4*)malloc(num_objects * sizeof(float4));
    float3* h_scales     = (float3*)malloc(num_objects * sizeof(float3));

    for (int i = 0; i < num_objects; i++) {
        h_positions[i]  = make_float4(objects[i].position[0], objects[i].position[1],
                                      objects[i].position[2], objects[i].bounding_radius);
        h_velocities[i] = make_float4(objects[i].velocity[0], objects[i].velocity[1],
                                      objects[i].velocity[2], objects[i].mass);
        h_quats[i]      = make_float4(0.0f, 0.0f, 0.0f, 1.0f);  // identity
        h_scales[i]     = make_float3(objects[i].scale[0], objects[i].scale[1], objects[i].scale[2]);
    }

    // Allocate device arrays
    CUDA_CHECK(cudaMalloc(&d_positions,    num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel_buf[0],   num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel_buf[1],   num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_quats,        num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_scales,       num_objects * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_verts_local,  g_total_verts * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_verts_world,  g_total_verts * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_vert_offsets, num_objects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vert_counts,  num_objects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_polytopes,    num_objects * sizeof(gkPolytope)));
    CUDA_CHECK(cudaMalloc(&d_simplices,    g_max_pairs * sizeof(gkSimplex)));
    CUDA_CHECK(cudaMalloc(&d_distances,    g_max_pairs * sizeof(gkFloat)));

    int total_cells = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    CUDA_CHECK(cudaMalloc(&d_grid_counts,    total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_objects,   total_cells * MAX_OBJECTS_PER_CELL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_pairs, g_max_pairs * sizeof(gkCollisionPair)));

    int scan_size = next_power_of_2(num_objects);
    CUDA_CHECK(cudaMalloc(&d_pair_counts,  scan_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_offsets, scan_size * sizeof(int)));

    // Upload
    CUDA_CHECK(cudaMemcpy(d_positions,    h_positions,   num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_buf[0],   h_velocities,  num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_buf[1],   h_velocities,  num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quats,        h_quats,       num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales,       h_scales,      num_objects * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vert_offsets, h_vert_offsets, num_objects * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vert_counts,  h_vert_counts,  num_objects * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verts_local,  h_verts_local,  g_total_verts * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Init polytope coord pointers (point into d_verts_world)
    int blocks = (num_objects + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_polytopes_kernel<<<blocks, BLOCK_SIZE>>>(
        d_polytopes, d_verts_world, d_vert_offsets, d_vert_counts, num_objects);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Register GL position buffer with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&g_gl_pos_resource,
                                            gl_pos_buffer,
                                            cudaGraphicsRegisterFlagsWriteDiscard));

    free(h_positions); free(h_velocities); free(h_quats); free(h_scales);
    free(h_vert_offsets); free(h_vert_counts); free(h_verts_local);

    g_vel_ping = 0; g_vel_pong = 1;
    return true;
}

void sim_cleanup(void) {
    if (g_gl_pos_resource) {
        cudaGraphicsUnregisterResource(g_gl_pos_resource);
        g_gl_pos_resource = nullptr;
    }
    cudaFree(d_positions);
    cudaFree(d_vel_buf[0]); cudaFree(d_vel_buf[1]);
    cudaFree(d_quats);
    cudaFree(d_scales);
    cudaFree(d_verts_local); cudaFree(d_verts_world);
    cudaFree(d_vert_offsets); cudaFree(d_vert_counts);
    cudaFree(d_polytopes); cudaFree(d_simplices); cudaFree(d_distances);
    cudaFree(d_grid_counts); cudaFree(d_grid_objects);
    cudaFree(d_collision_pairs);
    cudaFree(d_pair_counts); cudaFree(d_pair_offsets);
    g_num_objects = 0;
}

int sim_broad_phase(const PhysicsParams* params) {
    int num_objs = g_num_objects;
    float cell_size = COMPUTE_CELL_SIZE(params->boundary);
    int total_cells = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE;
    int blocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemset(d_grid_counts, 0, total_cells * sizeof(int)));

    insert_objects_kernel<<<blocks, BLOCK_SIZE>>>(
        d_positions, num_objs, d_grid_counts, d_grid_objects, cell_size, params->boundary);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemset(d_pair_counts, 0, num_objs * sizeof(int)));

    count_pairs_kernel<<<blocks, BLOCK_SIZE>>>(
        num_objs, d_grid_counts, d_grid_objects, d_positions, cell_size, params->boundary, d_pair_counts);
    CUDA_CHECK_LAST();

    int n_pow2 = next_power_of_2(num_objs);
    if (n_pow2 > num_objs)
        CUDA_CHECK(cudaMemset(d_pair_counts + num_objs, 0, (n_pow2 - num_objs) * sizeof(int)));

    if (num_objs <= 1024) {
        int bsz = n_pow2 / 2;
        if (bsz > 512) bsz = 512;
        int smem = (n_pow2 + CONFLICT_FREE_OFFSET(n_pow2)) * sizeof(int);
        block_scan_kernel<<<1, bsz, smem>>>(num_objs, d_pair_offsets, d_pair_counts, nullptr);
    } else {
        recursive_scan(n_pow2, d_pair_offsets, d_pair_counts);
    }
    CUDA_CHECK_LAST();

    int h_last_offset, h_last_count;
    CUDA_CHECK(cudaMemcpy(&h_last_offset, d_pair_offsets + num_objs - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_last_count,  d_pair_counts  + num_objs - 1, sizeof(int), cudaMemcpyDeviceToHost));
    g_num_pairs = h_last_offset + h_last_count;
    if (g_num_pairs > g_max_pairs) {
        fprintf(stderr, "Pair overflow: %d > %d, clamping\n", g_num_pairs, g_max_pairs);
        g_num_pairs = g_max_pairs;
    }

    if (g_num_pairs > 0) {
        generate_pairs_kernel<<<blocks, BLOCK_SIZE>>>(
            num_objs, d_grid_counts, d_grid_objects, d_positions,
            cell_size, params->boundary, d_pair_offsets, d_collision_pairs, g_max_pairs);
        CUDA_CHECK_LAST();
    }

    return g_num_pairs;
}

void sim_step(const PhysicsParams* params) {
    int num_objs  = g_num_objects;
    int num_pairs = g_num_pairs;
    int obj_blocks  = (num_objs  + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Integrate physics
    physics_update_kernel<<<obj_blocks, BLOCK_SIZE>>>(
        d_positions, d_vel_buf[g_vel_ping], d_scales,
        num_objs, params->delta_time, params->gravity[1], params->damping, params->boundary);
    CUDA_CHECK_LAST();

    // 2. Transform local verts to world space
    transform_to_world_kernel<<<obj_blocks, BLOCK_SIZE>>>(
        d_positions, d_quats, d_scales,
        d_verts_local, d_verts_world,
        d_vert_offsets, d_vert_counts, num_objs);
    CUDA_CHECK_LAST();

    if (num_pairs > 0) {
        // 3. GJK distance computation
        const int threads_per_gjk = 16;
        const int gjk_per_block   = 2;
        int gjk_blocks = (num_pairs + gjk_per_block - 1) / gjk_per_block;

        compute_minimum_distance_indexed_kernel<<<gjk_blocks, threads_per_gjk * gjk_per_block>>>(
            d_polytopes, d_collision_pairs, d_simplices, d_distances, num_pairs);
        CUDA_CHECK_LAST();

        // 4. Collision response (ping → pong)
        CUDA_CHECK(cudaMemcpy(d_vel_buf[g_vel_pong], d_vel_buf[g_vel_ping],
                              num_objs * sizeof(float4), cudaMemcpyDeviceToDevice));

        int pair_blocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        collision_response_kernel<<<pair_blocks, BLOCK_SIZE>>>(
            d_positions,
            d_vel_buf[g_vel_ping], d_vel_buf[g_vel_pong],
            d_collision_pairs, d_distances,
            params->collision_epsilon, num_pairs, num_objs);
        CUDA_CHECK_LAST();

        // 5. Swap ping/pong
        int tmp = g_vel_ping; g_vel_ping = g_vel_pong; g_vel_pong = tmp;
    }
}

void sim_copy_to_gl(void) {
    size_t size;
    float4* gl_ptr = nullptr;

    CUDA_CHECK(cudaGraphicsMapResources(1, &g_gl_pos_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&gl_ptr, &size, g_gl_pos_resource));

    int blocks = (g_num_objects + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copy_positions_to_gl_kernel<<<blocks, BLOCK_SIZE>>>(gl_ptr, d_positions, g_num_objects);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_gl_pos_resource, 0));
}

#ifdef __cplusplus
}
#endif
