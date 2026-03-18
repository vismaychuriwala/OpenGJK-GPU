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

// Random convex hull: n random points on unit sphere, seeded by `seed`.
// n must be >= 4 and <= MAX_HULL_VERTS. Returns malloc'd float[n * 3]. Caller frees.
float* gen_gjk_random_hull(int n, unsigned int seed, int* out_count);

// Build a flat-shaded convex hull mesh from a GJK vertex cloud and add it to the atlas.
// Uses the same points as gen_gjk_random_hull. Returns mesh_id, or -1 on failure.
int atlas_add_convex_hull(MeshAtlas* atlas, const float* gjk_verts, int n);

// World-space bounding radius from unit-scale verts + per-axis scale
float compute_bounding_radius(const float* verts_f3, int num_verts, const float scale[3]);

// Shifts all n vertices in-place so their volumetric centroid is at the origin.
// Must be called before atlas_add_convex_hull / compute_hull_inertia_k so that
// positions[i] in the simulation correctly represents the centre of mass.
void center_hull_verts(float* pts_flat, int n);

// Per-axis unit inertia factors for a convex hull (uniform density, unit-scale).
// Assumes vertices are already COM-centred (call center_hull_verts first).
// With scale (sx,sy,sz) and mass m:  Ixx = m*(ky*sy²+kz*sz²), etc.
void compute_hull_inertia_k(const float* gjk_verts, int n, float* kx, float* ky, float* kz);

// Load an OBJ file and build:
// Loads an OBJ, adds the original triangulated mesh to the atlas for rendering,
// and runs V-HACD convex decomposition for physics.
// Returns 1 on success, 0 on failure.
// out_gjk_verts and out_gjk_counts are malloc'd arrays of length *out_num_hulls.
// Caller frees: each out_gjk_verts[h], then out_gjk_verts and out_gjk_counts.
int load_obj_shape(MeshAtlas* atlas, const char* path,
                   int* out_render_mesh_id,
                   float*** out_gjk_verts,
                   int** out_gjk_counts, int* out_num_hulls);

#ifdef __cplusplus
}
#endif
