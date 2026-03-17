#pragma once
#include <stdint.h>

typedef struct {
    float pos[3];
    float normal[3];
} AtlasVertex;

typedef struct {
    int base_vertex;
    int first_index;
    int index_count;
    int vertex_count;
} MeshEntry;

typedef struct {
    AtlasVertex* vertices;
    int          num_vertices;
    int          vertex_cap;
    uint32_t*    indices;
    int          num_indices;
    int          index_cap;
    MeshEntry*   meshes;
    int          num_meshes;
    int          mesh_cap;
} MeshAtlas;

#ifdef __cplusplus
extern "C" {
#endif

void atlas_init(MeshAtlas* atlas);
void atlas_free(MeshAtlas* atlas);

// Append a mesh to the atlas, returns mesh_id
int atlas_add_mesh(MeshAtlas* atlas,
                   const AtlasVertex* verts, int nv,
                   const uint32_t* indices, int ni);

// Built-in convex shape generators — return mesh_id
int atlas_add_icosahedron(MeshAtlas* atlas);
int atlas_add_box(MeshAtlas* atlas);
int atlas_add_tetrahedron(MeshAtlas* atlas);
int atlas_add_octahedron(MeshAtlas* atlas);

// GJK vertex cloud generators — unit scale, local space
// Returns malloc'd float[*out_count * 3]. Caller frees.
float* gen_gjk_icosahedron(int* out_count);
float* gen_gjk_box(int* out_count);
float* gen_gjk_tetrahedron(int* out_count);
float* gen_gjk_octahedron(int* out_count);

// World-space bounding radius from unit-scale verts + per-axis scale
float compute_bounding_radius(const float* verts_f3, int num_verts, const float scale[3]);

#ifdef __cplusplus
}
#endif
