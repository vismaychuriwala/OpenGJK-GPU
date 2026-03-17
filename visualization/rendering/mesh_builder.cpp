#include "mesh_builder.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void atlas_grow_verts(MeshAtlas* a, int needed) {
    if (a->num_vertices + needed <= a->vertex_cap) return;
    while (a->vertex_cap < a->num_vertices + needed) a->vertex_cap = a->vertex_cap ? a->vertex_cap * 2 : 256;
    a->vertices = (AtlasVertex*)realloc(a->vertices, a->vertex_cap * sizeof(AtlasVertex));
}

static void atlas_grow_indices(MeshAtlas* a, int needed) {
    if (a->num_indices + needed <= a->index_cap) return;
    while (a->index_cap < a->num_indices + needed) a->index_cap = a->index_cap ? a->index_cap * 2 : 512;
    a->indices = (uint32_t*)realloc(a->indices, a->index_cap * sizeof(uint32_t));
}

static void atlas_grow_meshes(MeshAtlas* a) {
    if (a->num_meshes < a->mesh_cap) return;
    a->mesh_cap = a->mesh_cap ? a->mesh_cap * 2 : 16;
    a->meshes = (MeshEntry*)realloc(a->meshes, a->mesh_cap * sizeof(MeshEntry));
}

void atlas_init(MeshAtlas* atlas) {
    memset(atlas, 0, sizeof(MeshAtlas));
}

void atlas_free(MeshAtlas* atlas) {
    free(atlas->vertices);
    free(atlas->indices);
    free(atlas->meshes);
    memset(atlas, 0, sizeof(MeshAtlas));
}

int atlas_add_mesh(MeshAtlas* a, const AtlasVertex* verts, int nv,
                   const uint32_t* indices, int ni) {
    atlas_grow_verts(a, nv);
    atlas_grow_indices(a, ni);
    atlas_grow_meshes(a);

    int mesh_id = a->num_meshes++;
    MeshEntry* m = &a->meshes[mesh_id];
    m->base_vertex  = a->num_vertices;
    m->first_index  = a->num_indices;
    m->vertex_count = nv;
    m->index_count  = ni;

    memcpy(a->vertices + a->num_vertices, verts, nv * sizeof(AtlasVertex));
    memcpy(a->indices  + a->num_indices,  indices, ni * sizeof(uint32_t));

    a->num_vertices += nv;
    a->num_indices  += ni;
    return mesh_id;
}

// ============================================================================
// Shape geometry
// ============================================================================

static void normalize3(float* v) {
    float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-6f) { v[0] /= len; v[1] /= len; v[2] /= len; }
}

static void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void face_normal(const float* v0, const float* v1, const float* v2, float* out) {
    float e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    float e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
    cross3(e1, e2, out);
    normalize3(out);
}

// ---- Icosahedron ----

#define PHI_MESH 1.6180339887498948482f

static const float ico_verts_raw[12][3] = {
    {-1, PHI_MESH, 0}, {1, PHI_MESH, 0}, {-1, -PHI_MESH, 0}, {1, -PHI_MESH, 0},
    {0, -1, PHI_MESH}, {0,  1, PHI_MESH}, {0, -1, -PHI_MESH}, {0,  1, -PHI_MESH},
    {PHI_MESH, 0, -1}, {PHI_MESH, 0,  1}, {-PHI_MESH, 0, -1}, {-PHI_MESH, 0,  1},
};

static const uint32_t ico_tris[20][3] = {
    {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
    {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
    {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
    {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
};

int atlas_add_icosahedron(MeshAtlas* atlas) {
    AtlasVertex verts[12];
    for (int i = 0; i < 12; i++) {
        float v[3] = { ico_verts_raw[i][0], ico_verts_raw[i][1], ico_verts_raw[i][2] };
        normalize3(v);
        memcpy(verts[i].pos,    v, 12);
        memcpy(verts[i].normal, v, 12);  // for unit sphere: normal == position
    }
    uint32_t idx[60];
    for (int i = 0; i < 20; i++) {
        idx[i*3+0] = ico_tris[i][0];
        idx[i*3+1] = ico_tris[i][1];
        idx[i*3+2] = ico_tris[i][2];
    }
    return atlas_add_mesh(atlas, verts, 12, idx, 60);
}

// ---- Box (24 verts, per-face normals) ----

int atlas_add_box(MeshAtlas* atlas) {
    // 6 faces, 4 verts each, 2 tris each
    static const float face_normals[6][3] = {
        { 1,0,0},{-1,0,0},{0, 1,0},{0,-1,0},{0,0, 1},{0,0,-1}
    };
    // for each face, 4 corner positions in CCW order when looking along +normal
    static const float face_corners[6][4][3] = {
        {{ .5f,-.5f,-.5f},{ .5f, .5f,-.5f},{ .5f, .5f, .5f},{ .5f,-.5f, .5f}}, // +X
        {{-.5f,-.5f, .5f},{-.5f, .5f, .5f},{-.5f, .5f,-.5f},{-.5f,-.5f,-.5f}}, // -X
        {{-.5f, .5f,-.5f},{-.5f, .5f, .5f},{ .5f, .5f, .5f},{ .5f, .5f,-.5f}}, // +Y
        {{-.5f,-.5f, .5f},{-.5f,-.5f,-.5f},{ .5f,-.5f,-.5f},{ .5f,-.5f, .5f}}, // -Y
        {{-.5f,-.5f, .5f},{ .5f,-.5f, .5f},{ .5f, .5f, .5f},{-.5f, .5f, .5f}}, // +Z
        {{ .5f,-.5f,-.5f},{-.5f,-.5f,-.5f},{-.5f, .5f,-.5f},{ .5f, .5f,-.5f}}, // -Z
    };

    AtlasVertex verts[24];
    uint32_t    idx[36];
    for (int f = 0; f < 6; f++) {
        for (int c = 0; c < 4; c++) {
            memcpy(verts[f*4+c].pos,    face_corners[f][c],  12);
            memcpy(verts[f*4+c].normal, face_normals[f],     12);
        }
        // two tris: 0,1,2 and 0,2,3
        idx[f*6+0] = f*4+0; idx[f*6+1] = f*4+1; idx[f*6+2] = f*4+2;
        idx[f*6+3] = f*4+0; idx[f*6+4] = f*4+2; idx[f*6+5] = f*4+3;
    }
    return atlas_add_mesh(atlas, verts, 24, idx, 36);
}

// ---- Tetrahedron (12 verts: 3 per face, flat normals) ----

int atlas_add_tetrahedron(MeshAtlas* atlas) {
    // Vertices of a regular tetrahedron inscribed in unit sphere
    float s = 1.0f / sqrtf(3.0f);
    float tv[4][3] = {
        { s,  s,  s},
        {-s, -s,  s},
        {-s,  s, -s},
        { s, -s, -s},
    };
    // 4 faces
    static const int tet_faces[4][3] = {{0,2,1},{0,1,3},{0,3,2},{1,2,3}};

    AtlasVertex verts[12];
    uint32_t    idx[12];
    for (int f = 0; f < 4; f++) {
        float n[3];
        face_normal(tv[tet_faces[f][0]], tv[tet_faces[f][1]], tv[tet_faces[f][2]], n);
        for (int v = 0; v < 3; v++) {
            memcpy(verts[f*3+v].pos,    tv[tet_faces[f][v]], 12);
            memcpy(verts[f*3+v].normal, n, 12);
        }
        idx[f*3+0] = f*3+0;
        idx[f*3+1] = f*3+1;
        idx[f*3+2] = f*3+2;
    }
    return atlas_add_mesh(atlas, verts, 12, idx, 12);
}

// ---- Octahedron (smooth normals = vertex positions) ----

int atlas_add_octahedron(MeshAtlas* atlas) {
    float ov[6][3] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
    };
    // 8 faces
    static const int oct_faces[8][3] = {
        {0,2,4},{0,4,3},{0,3,5},{0,5,2},
        {1,4,2},{1,3,4},{1,5,3},{1,2,5},
    };
    AtlasVertex verts[6];
    for (int i = 0; i < 6; i++) {
        memcpy(verts[i].pos,    ov[i], 12);
        memcpy(verts[i].normal, ov[i], 12);
    }
    uint32_t idx[24];
    for (int f = 0; f < 8; f++) {
        idx[f*3+0] = oct_faces[f][0];
        idx[f*3+1] = oct_faces[f][1];
        idx[f*3+2] = oct_faces[f][2];
    }
    return atlas_add_mesh(atlas, verts, 6, idx, 24);
}

// ============================================================================
// GJK vertex cloud generators
// ============================================================================

float* gen_gjk_icosahedron(int* out_count) {
    *out_count = 12;
    float* v = (float*)malloc(12 * 3 * sizeof(float));
    for (int i = 0; i < 12; i++) {
        float x = ico_verts_raw[i][0], y = ico_verts_raw[i][1], z = ico_verts_raw[i][2];
        float len = sqrtf(x*x + y*y + z*z);
        v[i*3+0] = x/len; v[i*3+1] = y/len; v[i*3+2] = z/len;
    }
    return v;
}

float* gen_gjk_box(int* out_count) {
    *out_count = 8;
    float* v = (float*)malloc(8 * 3 * sizeof(float));
    float c[8][3] = {
        {-.5f,-.5f,-.5f},{.5f,-.5f,-.5f},{.5f,.5f,-.5f},{-.5f,.5f,-.5f},
        {-.5f,-.5f, .5f},{.5f,-.5f, .5f},{.5f,.5f, .5f},{-.5f,.5f, .5f},
    };
    for (int i = 0; i < 8; i++) { v[i*3+0]=c[i][0]; v[i*3+1]=c[i][1]; v[i*3+2]=c[i][2]; }
    return v;
}

float* gen_gjk_tetrahedron(int* out_count) {
    *out_count = 4;
    float* v = (float*)malloc(4 * 3 * sizeof(float));
    float s = 1.0f / sqrtf(3.0f);
    float c[4][3] = {{ s, s, s},{-s,-s, s},{-s, s,-s},{ s,-s,-s}};
    for (int i = 0; i < 4; i++) { v[i*3+0]=c[i][0]; v[i*3+1]=c[i][1]; v[i*3+2]=c[i][2]; }
    return v;
}

float* gen_gjk_octahedron(int* out_count) {
    *out_count = 6;
    float* v = (float*)malloc(6 * 3 * sizeof(float));
    float c[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int i = 0; i < 6; i++) { v[i*3+0]=c[i][0]; v[i*3+1]=c[i][1]; v[i*3+2]=c[i][2]; }
    return v;
}

// ============================================================================
// Bounding radius
// ============================================================================

float compute_bounding_radius(const float* verts_f3, int num_verts, const float scale[3]) {
    float max_r2 = 0.0f;
    for (int i = 0; i < num_verts; i++) {
        float x = verts_f3[i*3+0] * scale[0];
        float y = verts_f3[i*3+1] * scale[1];
        float z = verts_f3[i*3+2] * scale[2];
        float r2 = x*x + y*y + z*z;
        if (r2 > max_r2) max_r2 = r2;
    }
    return sqrtf(max_r2);
}
