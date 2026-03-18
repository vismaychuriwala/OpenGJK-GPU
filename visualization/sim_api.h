#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float  position[3];
    float  velocity[3];
    float  scale[3];
    float  color[4];
    float  mass;
    float  bounding_radius;      // world-space bounding radius enclosing all sub-meshes
    float  inv_inertia[3];       // inverse principal moments (Ixx,Iyy,Izz) in body frame
    int    num_submeshes;        // number of convex sub-meshes
    float** submesh_verts;       // [num_submeshes] flat float3 arrays, local unit-scale
    int*    submesh_vert_counts; // [num_submeshes] vertex counts per sub-mesh
    int    mesh_id;
} ObjectInitData;

typedef struct {
    float gravity_y;
    float delta_time;
    float boundary;
    float collision_epsilon;
} PhysicsParams;

// gl_pos_buffer:  caller-created GL buffer (GL_DYNAMIC_DRAW), size = num_objects * 16 bytes (float4)
// gl_quat_buffer: caller-created GL buffer (GL_DYNAMIC_DRAW), size = num_objects * 16 bytes (float4)
bool sim_init(const ObjectInitData* objects, int num_objects,
              unsigned int gl_pos_buffer, unsigned int gl_quat_buffer);
void sim_cleanup(void);
int  sim_broad_phase(const PhysicsParams* params);
void sim_step(const PhysicsParams* params);
void sim_copy_to_gl(void);

#ifdef __cplusplus
}
#endif
