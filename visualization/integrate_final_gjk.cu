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

static_assert(sizeof(gkFloat) == sizeof(float), "GJK float type mismatch: verts_world pool is float3 — recompile openGJK with gkFloat=float");

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
static float4* d_positions;       // xyz = world pos, w = bounding_radius (constant)
static float4* d_vel_buf[2];      // xyz = velocity, w = mass (constant)
static float4* d_quats;           // rotation quaternion, updated each step
static float4* d_ang_buf[2];      // xyz = angular velocity (omega), w = unused
static float3* d_inv_inertia;     // inverse principal moments in body frame (Ixx,Iyy,Izz)^-1
static float3* d_scales;          // per-axis scale (constant)
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
static int*              d_scan_aux;   // pre-allocated auxiliary buffer for recursive_scan
static int               g_grid_size;  // runtime grid divisions per axis
static float             g_cell_size;  // runtime cell size (>= 2 * max_bounding_radius)

// GL interop
static cudaGraphicsResource_t g_gl_pos_resource  = nullptr;
static cudaGraphicsResource_t g_gl_quat_resource = nullptr;

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

// Rotate by conjugate (R^T for unit quaternion)
__device__ static inline float3 quat_rotate_inv(float4 q, float3 v) {
    return quat_rotate(make_float4(-q.x, -q.y, -q.z, q.w), v);
}

// Apply friction impulse at a boundary contact.
// n        : wall normal (into simulation)
// r        : contact arm = contact_pt - center
// j_n_mag  : magnitude of the normal impulse already applied
// inv_mass : 1/m
// inv_I    : diagonal inverse inertia in body frame
// q        : current quaternion
__device__ static void apply_boundary_friction(
    float3& vel, float3& omega,
    float3 n, float3 r,
    float j_n_mag, float inv_mass,
    float3 inv_I, float4 q)
{
    // Velocity at contact point = linear + angular contribution
    float3 omg_r = make_float3(
        omega.y*r.z - omega.z*r.y,
        omega.z*r.x - omega.x*r.z,
        omega.x*r.y - omega.y*r.x);
    float3 v_contact = make_float3(vel.x + omg_r.x, vel.y + omg_r.y, vel.z + omg_r.z);

    // Remove normal component to get tangential velocity
    float v_n = v_contact.x*n.x + v_contact.y*n.y + v_contact.z*n.z;
    float3 v_tang = make_float3(v_contact.x - v_n*n.x,
                                v_contact.y - v_n*n.y,
                                v_contact.z - v_n*n.z);

    float v_t_sq = v_tang.x*v_tang.x + v_tang.y*v_tang.y + v_tang.z*v_tang.z;
    if (v_t_sq < 1e-6f) return;
    float v_t_mag = sqrtf(v_t_sq);

    // Tangent direction
    float inv_vt = 1.0f / v_t_mag;
    float3 t_dir = make_float3(v_tang.x * inv_vt, v_tang.y * inv_vt, v_tang.z * inv_vt);

    // Angular effective mass: (r × t) · (I^-1 * (r × t))
    float3 rxt = make_float3(r.y*t_dir.z - r.z*t_dir.y,
                              r.z*t_dir.x - r.x*t_dir.z,
                              r.x*t_dir.y - r.y*t_dir.x);
    float3 rxt_body = quat_rotate_inv(q, rxt);
    float3 Irxt     = make_float3(inv_I.x*rxt_body.x, inv_I.y*rxt_body.y, inv_I.z*rxt_body.z);
    float ang_eff   = rxt_body.x*Irxt.x + rxt_body.y*Irxt.y + rxt_body.z*Irxt.z;

    // Coulomb friction: clamp to not exceed impulse needed to stop sliding
    float j_f_max  = FRICTION_COEFF * j_n_mag;
    float j_f_stop = v_t_mag / (inv_mass + ang_eff);
    float j_f_mag  = fminf(j_f_max, j_f_stop);

    float3 j_f = make_float3(-j_f_mag * v_tang.x / v_t_mag,
                              -j_f_mag * v_tang.y / v_t_mag,
                              -j_f_mag * v_tang.z / v_t_mag);

    // Linear velocity change
    vel.x += j_f.x * inv_mass;
    vel.y += j_f.y * inv_mass;
    vel.z += j_f.z * inv_mass;

    // Angular velocity change: tau = r × j_f, rotate to body frame, apply inv_I, rotate back
    float3 tau_world = make_float3(r.y*j_f.z - r.z*j_f.y,
                                   r.z*j_f.x - r.x*j_f.z,
                                   r.x*j_f.y - r.y*j_f.x);
    float3 tau_body  = quat_rotate_inv(q, tau_world);
    float3 d_omg_body = make_float3(inv_I.x*tau_body.x, inv_I.y*tau_body.y, inv_I.z*tau_body.z);
    float3 d_omg     = quat_rotate(q, d_omg_body);
    omega.x += d_omg.x;
    omega.y += d_omg.y;
    omega.z += d_omg.z;
}

// Unified boundary contact: separating-contact guard, correct normal impulse
// (including angular effective mass), angular bounce impulse, then friction.
// n        : wall normal pointing INTO simulation
// r        : contact arm = contact_pt - center  (bounding sphere: r = -br * n)
__device__ static void apply_boundary_contact(
    float3& vel, float3& omega,
    float3 n, float3 r,
    float inv_mass, float3 inv_I, float4 q)
{
    // Full contact-point closing velocity (linear + angular contribution)
    float3 omg_r = make_float3(omega.y*r.z - omega.z*r.y,
                                omega.z*r.x - omega.x*r.z,
                                omega.x*r.y - omega.y*r.x);
    float v_n_close = -( (vel.x + omg_r.x)*n.x
                        +(vel.y + omg_r.y)*n.y
                        +(vel.z + omg_r.z)*n.z );
    if (v_n_close <= 0.0f) return;  // already separating — no impulse needed

    // Angular effective mass: (r × n) · I^-1 (r × n)
    // With bounding-sphere r = -br*n this is 0; kept correct for general r.
    float3 rxn       = make_float3(r.y*n.z - r.z*n.y, r.z*n.x - r.x*n.z, r.x*n.y - r.y*n.x);
    float3 rxn_body  = quat_rotate_inv(q, rxn);
    float3 Irxn      = make_float3(inv_I.x*rxn_body.x, inv_I.y*rxn_body.y, inv_I.z*rxn_body.z);
    float  ang_denom = rxn_body.x*Irxn.x + rxn_body.y*Irxn.y + rxn_body.z*Irxn.z;

    float j_n = (1.0f + RESTITUTION) * v_n_close / (inv_mass + ang_denom);

    // Linear impulse
    vel.x += j_n * inv_mass * n.x;
    vel.y += j_n * inv_mass * n.y;
    vel.z += j_n * inv_mass * n.z;

    // Angular impulse: dω = I^-1 * (r × j_n*n) = j_n * (I^-1 * rxn)
    float3 d_omg_body  = make_float3(j_n*Irxn.x, j_n*Irxn.y, j_n*Irxn.z);
    float3 d_omg_world = quat_rotate(q, d_omg_body);
    omega.x += d_omg_world.x;
    omega.y += d_omg_world.y;
    omega.z += d_omg_world.z;

    // Friction using the correctly-computed j_n as Coulomb clamp
    apply_boundary_friction(vel, omega, n, r, j_n, inv_mass, inv_I, q);
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
    float4*       positions,
    float4*       velocities,
    float4*       ang_vel,
    float4*       quats,
    const float3* inv_inertia,
    int           num_objects,
    float         dt,
    float         gravity_y,
    float         boundary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;

    float4 pos   = positions[i];   // w = bounding_radius
    float4 vel   = velocities[i];  // w = mass
    float4 q     = quats[i];
    float3 omega = make_float3(ang_vel[i].x, ang_vel[i].y, ang_vel[i].z);
    float3 inv_I = inv_inertia[i];

    float br      = pos.w;
    float inv_mass = 1.0f / vel.w;

    vel.y += gravity_y * dt;
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    float3 vel3 = make_float3(vel.x, vel.y, vel.z);

    // Apply angular damping to carry-over omega before boundary contacts
    omega.x *= ANGULAR_DAMPING;
    omega.y *= ANGULAR_DAMPING;
    omega.z *= ANGULAR_DAMPING;

    // Floor (y = -boundary), n = (0,1,0), r = (0,-br,0)
    if (pos.y - br < -boundary) {
        pos.y = -boundary + br;
        apply_boundary_contact(vel3, omega,
            make_float3(0.0f,  1.0f, 0.0f),
            make_float3(0.0f, -br,   0.0f),
            inv_mass, inv_I, q);
    }
    // Ceiling (y = +boundary), n = (0,-1,0), r = (0,+br,0)
    if (pos.y + br > boundary) {
        pos.y = boundary - br;
        apply_boundary_contact(vel3, omega,
            make_float3(0.0f, -1.0f, 0.0f),
            make_float3(0.0f, +br,   0.0f),
            inv_mass, inv_I, q);
    }
    // -X wall, n = (1,0,0), r = (-br,0,0)
    if (pos.x - br < -boundary) {
        pos.x = -boundary + br;
        apply_boundary_contact(vel3, omega,
            make_float3( 1.0f, 0.0f, 0.0f),
            make_float3(-br,   0.0f, 0.0f),
            inv_mass, inv_I, q);
    }
    // +X wall, n = (-1,0,0), r = (+br,0,0)
    if (pos.x + br > boundary) {
        pos.x = boundary - br;
        apply_boundary_contact(vel3, omega,
            make_float3(-1.0f, 0.0f, 0.0f),
            make_float3(+br,   0.0f, 0.0f),
            inv_mass, inv_I, q);
    }
    // -Z wall, n = (0,0,1), r = (0,0,-br)
    if (pos.z - br < -boundary) {
        pos.z = -boundary + br;
        apply_boundary_contact(vel3, omega,
            make_float3(0.0f, 0.0f,  1.0f),
            make_float3(0.0f, 0.0f, -br),
            inv_mass, inv_I, q);
    }
    // +Z wall, n = (0,0,-1), r = (0,0,+br)
    if (pos.z + br > boundary) {
        pos.z = boundary - br;
        apply_boundary_contact(vel3, omega,
            make_float3(0.0f, 0.0f, -1.0f),
            make_float3(0.0f, 0.0f, +br),
            inv_mass, inv_I, q);
    }

    vel.x = vel3.x; vel.y = vel3.y; vel.z = vel3.z;
    positions[i]  = pos;
    velocities[i] = vel;

    // Quaternion integration from angular velocity
    float4 dq;
    dq.x =  0.5f*( omega.x*q.w + omega.y*q.z - omega.z*q.y);
    dq.y =  0.5f*( omega.y*q.w + omega.z*q.x - omega.x*q.z);
    dq.z =  0.5f*( omega.z*q.w + omega.x*q.y - omega.y*q.x);
    dq.w =  0.5f*(-omega.x*q.x - omega.y*q.y - omega.z*q.z);

    q.x += dq.x * dt; q.y += dq.y * dt; q.z += dq.z * dt; q.w += dq.w * dt;
    float inv_len = rsqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    quats[i] = make_float4(q.x*inv_len, q.y*inv_len, q.z*inv_len, q.w*inv_len);

    ang_vel[i] = make_float4(omega.x, omega.y, omega.z, 0.0f);
}

__global__ void insert_objects_kernel(
    const float4* positions,
    int           num_objects,
    int*          grid_counts,
    int*          grid_objects,
    float         cell_size,
    float         boundary,
    int           grid_size)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    float4 pos = positions[obj_id];
    int cx = (int)floorf((pos.x + boundary) / cell_size);
    int cy = (int)floorf((pos.y + boundary) / cell_size);
    int cz = (int)floorf((pos.z + boundary) / cell_size);
    cx = max(0, min(cx, grid_size - 1));
    cy = max(0, min(cy, grid_size - 1));
    cz = max(0, min(cz, grid_size - 1));

    int cell_idx = cx + cy * grid_size + cz * grid_size * grid_size;
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
    int*          pair_counts,
    int           grid_size)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    float4 pos = positions[obj_id];
    int cx = max(0, min((int)floorf((pos.x + boundary) / cell_size), grid_size - 1));
    int cy = max(0, min((int)floorf((pos.y + boundary) / cell_size), grid_size - 1));
    int cz = max(0, min((int)floorf((pos.z + boundary) / cell_size), grid_size - 1));

    int count = 0;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx+dx, ny = cy+dy, nz = cz+dz;
        if (nx<0||nx>=grid_size||ny<0||ny>=grid_size||nz<0||nz>=grid_size) continue;
        int cell_idx = nx + ny*grid_size + nz*grid_size*grid_size;
        int cc = min(grid_counts[cell_idx], MAX_OBJECTS_PER_CELL);
        for (int i = 0; i < cc; i++) {
            int other = grid_objects[cell_idx * MAX_OBJECTS_PER_CELL + i];
            if (other <= obj_id) continue;
            float4 opos = positions[other];
            float fx = pos.x - opos.x, fy = pos.y - opos.y, fz = pos.z - opos.z;
            float r_sum = pos.w + opos.w;  // w = bounding_radius
            if (fx*fx + fy*fy + fz*fz < r_sum*r_sum) count++;
        }
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
    int            max_pairs,
    int            grid_size)
{
    int obj_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (obj_id >= num_objects) return;

    float4 pos = positions[obj_id];
    int cx = max(0, min((int)floorf((pos.x + boundary) / cell_size), grid_size - 1));
    int cy = max(0, min((int)floorf((pos.y + boundary) / cell_size), grid_size - 1));
    int cz = max(0, min((int)floorf((pos.z + boundary) / cell_size), grid_size - 1));

    int write_offset = pair_offsets[obj_id];
    int local_idx = 0;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int nx = cx+dx, ny = cy+dy, nz = cz+dz;
        if (nx<0||nx>=grid_size||ny<0||ny>=grid_size||nz<0||nz>=grid_size) continue;
        int cell_idx = nx + ny*grid_size + nz*grid_size*grid_size;
        int cc = min(grid_counts[cell_idx], MAX_OBJECTS_PER_CELL);
        for (int i = 0; i < cc; i++) {
            int other = grid_objects[cell_idx * MAX_OBJECTS_PER_CELL + i];
            if (other <= obj_id) continue;
            float4 opos = positions[other];
            float fx = pos.x - opos.x, fy = pos.y - opos.y, fz = pos.z - opos.z;
            float r_sum = pos.w + opos.w;
            if (fx*fx + fy*fy + fz*fz >= r_sum*r_sum) continue;
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
    float4*                ang_ping,
    float4*                ang_pong,
    const float4*          quats,
    const float3*          inv_inertia,
    const gkSimplex*       simplices,
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
    float4 qA   = quats[idA],     qB   = quats[idB];
    float3 omgA = make_float3(ang_ping[idA].x, ang_ping[idA].y, ang_ping[idA].z);
    float3 omgB = make_float3(ang_ping[idB].x, ang_ping[idB].y, ang_ping[idB].z);
    float3 iA   = inv_inertia[idA];
    float3 iB   = inv_inertia[idB];

    // Center-to-center direction (fallback normal)
    float dx = posB.x - posA.x, dy = posB.y - posA.y, dz = posB.z - posA.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    if (dist < 0.0001f) return;
    float inv_d = 1.0f / dist;

    // Contact normal and point from GJK witness points (world space)
    float wAx = (float)simplices[p].witnesses[0][0];
    float wAy = (float)simplices[p].witnesses[0][1];
    float wAz = (float)simplices[p].witnesses[0][2];
    float wBx = (float)simplices[p].witnesses[1][0];
    float wBy = (float)simplices[p].witnesses[1][1];
    float wBz = (float)simplices[p].witnesses[1][2];

    float wdx = wBx - wAx, wdy = wBy - wAy, wdz = wBz - wAz;
    float wdist = sqrtf(wdx*wdx + wdy*wdy + wdz*wdz);

    float nx, ny, nz;
    float3 rA, rB;
    if (wdist > 0.001f) {
        float inv_wd = 1.0f / wdist;
        nx = wdx * inv_wd; ny = wdy * inv_wd; nz = wdz * inv_wd;
        // Use per-shape witness points directly — exact surface-to-center lever arms
        rA = make_float3(wAx - posA.x, wAy - posA.y, wAz - posA.z);
        rB = make_float3(wBx - posB.x, wBy - posB.y, wBz - posB.z);
    } else {
        // Witnesses converged (deep penetration) — fall back to center-to-center
        nx = dx * inv_d; ny = dy * inv_d; nz = dz * inv_d;
        rA = make_float3(nx * posA.w, ny * posA.w, nz * posA.w);
        rB = make_float3(-nx * posB.w, -ny * posB.w, -nz * posB.w);
    }

    // Linear + angular velocity at contact point
    float3 vA_ang = make_float3(omgA.y*rA.z - omgA.z*rA.y,
                                 omgA.z*rA.x - omgA.x*rA.z,
                                 omgA.x*rA.y - omgA.y*rA.x);
    float3 vB_ang = make_float3(omgB.y*rB.z - omgB.z*rB.y,
                                 omgB.z*rB.x - omgB.x*rB.z,
                                 omgB.x*rB.y - omgB.y*rB.x);

    float rvx = (velB.x + vB_ang.x) - (velA.x + vA_ang.x);
    float rvy = (velB.y + vB_ang.y) - (velA.y + vA_ang.y);
    float rvz = (velB.z + vB_ang.z) - (velA.z + vA_ang.z);
    float vn = rvx*nx + rvy*ny + rvz*nz;
    if (vn > 0.0f) return;

    // Angular contributions to impulse denominator
    // tA_world = rA × n
    float3 tA_world = make_float3(rA.y*nz - rA.z*ny, rA.z*nx - rA.x*nz, rA.x*ny - rA.y*nx);
    // tB_world = rB × n
    float3 tB_world = make_float3(rB.y*nz - rB.z*ny, rB.z*nx - rB.x*nz, rB.x*ny - rB.y*nx);

    // Rotate to body frame, apply diagonal inverse inertia
    float3 tA_body = quat_rotate_inv(qA, tA_world);
    float3 IA_tA   = make_float3(iA.x*tA_body.x, iA.y*tA_body.y, iA.z*tA_body.z);
    float3 tB_body = quat_rotate_inv(qB, tB_world);
    float3 IB_tB   = make_float3(iB.x*tB_body.x, iB.y*tB_body.y, iB.z*tB_body.z);

    // Dot product is invariant under rotation — compute in body frame
    float ang_denom_A = tA_body.x*IA_tA.x + tA_body.y*IA_tA.y + tA_body.z*IA_tA.z;
    float ang_denom_B = tB_body.x*IB_tB.x + tB_body.y*IB_tB.y + tB_body.z*IB_tB.z;

    float inv_mA = 1.0f / velA.w;
    float inv_mB = 1.0f / velB.w;
    float j = -(1.0f + RESTITUTION) * vn / (inv_mA + inv_mB + ang_denom_A + ang_denom_B);

    // Linear impulse
    atomicAdd(&vel_pong[idA].x, -j * nx * inv_mA);
    atomicAdd(&vel_pong[idA].y, -j * ny * inv_mA);
    atomicAdd(&vel_pong[idA].z, -j * nz * inv_mA);
    atomicAdd(&vel_pong[idB].x,  j * nx * inv_mB);
    atomicAdd(&vel_pong[idB].y,  j * ny * inv_mB);
    atomicAdd(&vel_pong[idB].z,  j * nz * inv_mB);

    // Angular impulse: delta_omega = R * (inv_I_body * (R^T * (r × j*n)))
    // For A: impulse is -j*n (A pushes back), so cross = rA × (-j*n) = -j * tA_world
    float3 dOmgA_body  = make_float3(-j*IA_tA.x, -j*IA_tA.y, -j*IA_tA.z);
    float3 dOmgA_world = quat_rotate(qA, dOmgA_body);
    atomicAdd(&ang_pong[idA].x, dOmgA_world.x);
    atomicAdd(&ang_pong[idA].y, dOmgA_world.y);
    atomicAdd(&ang_pong[idA].z, dOmgA_world.z);

    // For B: impulse is +j*n, so cross = rB × (j*n) = j * tB_world
    float3 dOmgB_body  = make_float3(j*IB_tB.x, j*IB_tB.y, j*IB_tB.z);
    float3 dOmgB_world = quat_rotate(qB, dOmgB_body);
    atomicAdd(&ang_pong[idB].x, dOmgB_world.x);
    atomicAdd(&ang_pong[idB].y, dOmgB_world.y);
    atomicAdd(&ang_pong[idB].z, dOmgB_world.z);
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
    gl_buf[i] = positions[i];
}

__global__ void copy_quats_to_gl_kernel(float4* gl_buf, const float4* quats, int num_objects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_objects) return;
    gl_buf[i] = quats[i];
}

// ============================================================================
// Public API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

bool sim_init(const ObjectInitData* objects, int num_objects,
              unsigned int gl_pos_buffer, unsigned int gl_quat_buffer) {
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
    float4* h_positions    = (float4*)malloc(num_objects * sizeof(float4));
    float4* h_velocities   = (float4*)malloc(num_objects * sizeof(float4));
    float4* h_quats        = (float4*)malloc(num_objects * sizeof(float4));
    float4* h_ang_vel      = (float4*)malloc(num_objects * sizeof(float4));
    float3* h_inv_inertia  = (float3*)malloc(num_objects * sizeof(float3));
    float3* h_scales       = (float3*)malloc(num_objects * sizeof(float3));

    for (int i = 0; i < num_objects; i++) {
        h_positions[i]  = make_float4(objects[i].position[0], objects[i].position[1],
                                      objects[i].position[2], objects[i].bounding_radius);
        h_velocities[i] = make_float4(objects[i].velocity[0], objects[i].velocity[1],
                                      objects[i].velocity[2], objects[i].mass);
        h_quats[i]      = make_float4(0.0f, 0.0f, 0.0f, 1.0f);  // identity
        h_ang_vel[i]    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        h_inv_inertia[i] = make_float3(objects[i].inv_inertia[0],
                                       objects[i].inv_inertia[1],
                                       objects[i].inv_inertia[2]);
        h_scales[i]     = make_float3(objects[i].scale[0], objects[i].scale[1], objects[i].scale[2]);
    }

    // Allocate device arrays
    CUDA_CHECK(cudaMalloc(&d_positions,    num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel_buf[0],   num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel_buf[1],   num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_quats,        num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_ang_buf[0],   num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_ang_buf[1],   num_objects * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_inv_inertia,  num_objects * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_scales,       num_objects * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_verts_local,  g_total_verts * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_verts_world,  g_total_verts * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_vert_offsets, num_objects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vert_counts,  num_objects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_polytopes,    num_objects * sizeof(gkPolytope)));
    CUDA_CHECK(cudaMalloc(&d_simplices,    g_max_pairs * sizeof(gkSimplex)));
    CUDA_CHECK(cudaMalloc(&d_distances,    g_max_pairs * sizeof(gkFloat)));

    // Compute runtime grid size from max bounding radius
    float max_br = 0.0f;
    for (int i = 0; i < num_objects; i++)
        if (objects[i].bounding_radius > max_br) max_br = objects[i].bounding_radius;
    float boundary = COMPUTE_BOUNDARY(num_objects);
    float min_cell = 2.0f * max_br;  // cell must be >= diameter of largest object
    g_cell_size = fmaxf(min_cell, (2.0f * boundary) / MAX_SPATIAL_GRID_SIZE);
    g_grid_size = max(1, min((int)(2.0f * boundary / g_cell_size), MAX_SPATIAL_GRID_SIZE));

    int total_cells = g_grid_size * g_grid_size * g_grid_size;
    CUDA_CHECK(cudaMalloc(&d_grid_counts,    total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_objects,   total_cells * MAX_OBJECTS_PER_CELL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_pairs, g_max_pairs * sizeof(gkCollisionPair)));

    int scan_size = next_power_of_2(num_objects);
    CUDA_CHECK(cudaMalloc(&d_pair_counts,  scan_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_offsets, scan_size * sizeof(int)));

    int aux_size = scan_aux_size(scan_size);
    if (aux_size > 0)
        CUDA_CHECK(cudaMalloc(&d_scan_aux, aux_size * sizeof(int)));
    else
        d_scan_aux = nullptr;

    // Upload
    CUDA_CHECK(cudaMemcpy(d_positions,    h_positions,   num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_buf[0],   h_velocities,  num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_buf[1],   h_velocities,  num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quats,        h_quats,       num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ang_buf[0],   h_ang_vel,     num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ang_buf[1],   h_ang_vel,     num_objects * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_inertia,  h_inv_inertia, num_objects * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales,       h_scales,      num_objects * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vert_offsets, h_vert_offsets, num_objects * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vert_counts,  h_vert_counts,  num_objects * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verts_local,  h_verts_local,  g_total_verts * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Init polytope coord pointers
    int blocks = (num_objects + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_polytopes_kernel<<<blocks, BLOCK_SIZE>>>(
        d_polytopes, d_verts_world, d_vert_offsets, d_vert_counts, num_objects);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Register GL buffers with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&g_gl_pos_resource,
                                            gl_pos_buffer,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&g_gl_quat_resource,
                                            gl_quat_buffer,
                                            cudaGraphicsRegisterFlagsWriteDiscard));

    free(h_positions); free(h_velocities); free(h_quats);
    free(h_ang_vel); free(h_inv_inertia); free(h_scales);
    free(h_vert_offsets); free(h_vert_counts); free(h_verts_local);

    g_vel_ping = 0; g_vel_pong = 1;
    return true;
}

void sim_cleanup(void) {
    if (g_gl_pos_resource) {
        cudaGraphicsUnregisterResource(g_gl_pos_resource);
        g_gl_pos_resource = nullptr;
    }
    if (g_gl_quat_resource) {
        cudaGraphicsUnregisterResource(g_gl_quat_resource);
        g_gl_quat_resource = nullptr;
    }
    cudaFree(d_positions);
    cudaFree(d_vel_buf[0]); cudaFree(d_vel_buf[1]);
    cudaFree(d_quats);
    cudaFree(d_ang_buf[0]); cudaFree(d_ang_buf[1]);
    cudaFree(d_inv_inertia);
    cudaFree(d_scales);
    cudaFree(d_verts_local); cudaFree(d_verts_world);
    cudaFree(d_vert_offsets); cudaFree(d_vert_counts);
    cudaFree(d_polytopes); cudaFree(d_simplices); cudaFree(d_distances);
    cudaFree(d_grid_counts); cudaFree(d_grid_objects);
    cudaFree(d_collision_pairs);
    cudaFree(d_pair_counts); cudaFree(d_pair_offsets);
    if (d_scan_aux) cudaFree(d_scan_aux);
    g_num_objects = 0;
}

int sim_broad_phase(const PhysicsParams* params) {
    int num_objs = g_num_objects;
    int total_cells = g_grid_size * g_grid_size * g_grid_size;
    int blocks = (num_objs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemset(d_grid_counts, 0, total_cells * sizeof(int)));

    insert_objects_kernel<<<blocks, BLOCK_SIZE>>>(
        d_positions, num_objs, d_grid_counts, d_grid_objects, g_cell_size, params->boundary, g_grid_size);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemset(d_pair_counts, 0, num_objs * sizeof(int)));

    count_pairs_kernel<<<blocks, BLOCK_SIZE>>>(
        num_objs, d_grid_counts, d_grid_objects, d_positions, g_cell_size, params->boundary, d_pair_counts, g_grid_size);
    CUDA_CHECK_LAST();

    int n_pow2 = next_power_of_2(num_objs);
    if (n_pow2 > num_objs)
        CUDA_CHECK(cudaMemset(d_pair_counts + num_objs, 0, (n_pow2 - num_objs) * sizeof(int)));

    if (n_pow2 <= SCAN_B) {
        int bsz = n_pow2 / 2;
        int smem = (n_pow2 + CONFLICT_FREE_OFFSET(n_pow2)) * sizeof(int);
        block_scan_kernel<<<1, bsz, smem>>>(n_pow2, d_pair_offsets, d_pair_counts, nullptr);
    } else {
        recursive_scan(n_pow2, d_pair_offsets, d_pair_counts, d_scan_aux);
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
            g_cell_size, params->boundary, d_pair_offsets, d_collision_pairs, g_max_pairs, g_grid_size);
        CUDA_CHECK_LAST();
    }

    return g_num_pairs;
}

void sim_step(const PhysicsParams* params) {
    int num_objs  = g_num_objects;
    int num_pairs = g_num_pairs;
    int obj_blocks  = (num_objs  + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Integrate physics (position, angular velocity → quaternion, boundary friction)
    physics_update_kernel<<<obj_blocks, BLOCK_SIZE>>>(
        d_positions, d_vel_buf[g_vel_ping], d_ang_buf[g_vel_ping], d_quats, d_inv_inertia,
        num_objs, params->delta_time, params->gravity_y, params->boundary);
    CUDA_CHECK_LAST();

    // 2. Transform local verts to world space
    transform_to_world_kernel<<<obj_blocks, BLOCK_SIZE>>>(
        d_positions, d_quats, d_scales,
        d_verts_local, d_verts_world,
        d_vert_offsets, d_vert_counts, num_objs);
    CUDA_CHECK_LAST();

    if (num_pairs > 0) {
        // 3. GJK distance computation
        compute_minimum_distance_indexed_device(
            num_pairs, d_polytopes, d_collision_pairs, d_simplices, d_distances);
        CUDA_CHECK_LAST();

        // 4. Collision response (ping → pong, linear + angular)
        CUDA_CHECK(cudaMemcpy(d_vel_buf[g_vel_pong], d_vel_buf[g_vel_ping],
                              num_objs * sizeof(float4), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_ang_buf[g_vel_pong], d_ang_buf[g_vel_ping],
                              num_objs * sizeof(float4), cudaMemcpyDeviceToDevice));

        int pair_blocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        collision_response_kernel<<<pair_blocks, BLOCK_SIZE>>>(
            d_positions,
            d_vel_buf[g_vel_ping], d_vel_buf[g_vel_pong],
            d_ang_buf[g_vel_ping], d_ang_buf[g_vel_pong],
            d_quats, d_inv_inertia, d_simplices,
            d_collision_pairs, d_distances,
            params->collision_epsilon, num_pairs, num_objs);
        CUDA_CHECK_LAST();

        // 5. Swap ping/pong
        int tmp = g_vel_ping; g_vel_ping = g_vel_pong; g_vel_pong = tmp;
    }
}

void sim_copy_to_gl(void) {
    size_t size;
    int blocks = (g_num_objects + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Map both resources in a single driver call
    cudaGraphicsResource_t resources[2] = { g_gl_pos_resource, g_gl_quat_resource };
    CUDA_CHECK(cudaGraphicsMapResources(2, resources, 0));

    float4* pos_ptr = nullptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&pos_ptr, &size, g_gl_pos_resource));
    copy_positions_to_gl_kernel<<<blocks, BLOCK_SIZE>>>(pos_ptr, d_positions, g_num_objects);
    CUDA_CHECK_LAST();

    float4* quat_ptr = nullptr;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&quat_ptr, &size, g_gl_quat_resource));
    copy_quats_to_gl_kernel<<<blocks, BLOCK_SIZE>>>(quat_ptr, d_quats, g_num_objects);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaGraphicsUnmapResources(2, resources, 0));
}

#ifdef __cplusplus
}
#endif
