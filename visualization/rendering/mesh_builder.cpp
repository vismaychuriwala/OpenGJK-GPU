#include "mesh_builder.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <unordered_set>

#include <tinyobj/tiny_obj_loader.h>

#define ENABLE_VHACD_IMPLEMENTATION 1
#include <v-hacd/VHACD.h>

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

static void set_spherical_uv(AtlasVertex* v) {
    float x = v->pos[0], y = v->pos[1], z = v->pos[2];
    float len = sqrtf(x*x + y*y + z*z);
    if (len > 1e-6f) { x /= len; y /= len; z /= len; }
    if (y >  1.0f) y =  1.0f;
    if (y < -1.0f) y = -1.0f;
    v->uv[0] = atan2f(z, x) * 0.15915f + 0.5f;
    v->uv[1] = asinf(y) * 0.31831f + 0.5f;
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
        set_spherical_uv(&verts[i]);
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
            set_spherical_uv(&verts[f*4+c]);
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
            set_spherical_uv(&verts[f*3+v]);
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
        set_spherical_uv(&verts[i]);
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
// 3D incremental convex hull
// ============================================================================

typedef struct { int v[3]; } HullFace;

// Returns 6x signed volume of tetrahedron (a,b,c,p). Positive = p above face (a,b,c).
static float hull_signed_vol6(const float (*pts)[3], HullFace f, int p) {
    const float *a=pts[f.v[0]], *b=pts[f.v[1]], *c=pts[f.v[2]], *d=pts[p];
    float bx=b[0]-a[0], by=b[1]-a[1], bz=b[2]-a[2];
    float cx=c[0]-a[0], cy=c[1]-a[1], cz=c[2]-a[2];
    float dx=d[0]-a[0], dy=d[1]-a[1], dz=d[2]-a[2];
    float nx=by*cz-bz*cy, ny=bz*cx-bx*cz, nz=bx*cy-by*cx;
    return nx*dx + ny*dy + nz*dz;
}

// Builds the convex hull of pts[0..n-1] (each 3 floats). Returns face list.
static std::vector<HullFace> build_hull(const float* pts_flat, int n) {
    std::vector<HullFace> hull;
    const float (*pts)[3] = (const float (*)[3])pts_flat;
    if (n < 4) return hull;

    // --- Find initial tetrahedron ---
    int i0=0, i1=1, i2=-1, i3=-1;
    // O(n): find the most-separated axis-aligned pair as i0/i1
    { int xmn=0,xmx=0,ymn=0,ymx=0,zmn=0,zmx=0;
      for (int i=1;i<n;i++) {
          if (pts[i][0]<pts[xmn][0]) xmn=i; if (pts[i][0]>pts[xmx][0]) xmx=i;
          if (pts[i][1]<pts[ymn][1]) ymn=i; if (pts[i][1]>pts[ymx][1]) ymx=i;
          if (pts[i][2]<pts[zmn][2]) zmn=i; if (pts[i][2]>pts[zmx][2]) zmx=i;
      }
      int cands[6]={xmn,xmx,ymn,ymx,zmn,zmx};
      float bd=0;
      for (int a=0;a<6;a++) for (int b=a+1;b<6;b++) {
          float dx=pts[cands[b]][0]-pts[cands[a]][0],
                dy=pts[cands[b]][1]-pts[cands[a]][1],
                dz=pts[cands[b]][2]-pts[cands[a]][2];
          float d=dx*dx+dy*dy+dz*dz;
          if (d>bd) { bd=d; i0=cands[a]; i1=cands[b]; }
      } }
    { float e[3]={pts[i1][0]-pts[i0][0],pts[i1][1]-pts[i0][1],pts[i1][2]-pts[i0][2]};
      float bd=0;
      for (int i=0;i<n;i++) {
          if (i==i0||i==i1) continue;
          float vx=pts[i][0]-pts[i0][0], vy=pts[i][1]-pts[i0][1], vz=pts[i][2]-pts[i0][2];
          float cx=vy*e[2]-vz*e[1], cy=vz*e[0]-vx*e[2], cz=vx*e[1]-vy*e[0];
          float d=cx*cx+cy*cy+cz*cz;
          if (d>bd) { bd=d; i2=i; }
      } }
    if (i2<0) return hull;
    { float bd=0;
      for (int i=0;i<n;i++) {
          if (i==i0||i==i1||i==i2) continue;
          HullFace f; f.v[0]=i0; f.v[1]=i1; f.v[2]=i2;
          float d=hull_signed_vol6(pts,f,i); if (d<0) d=-d;
          if (d>bd) { bd=d; i3=i; }
      } }
    if (i3<0) return hull;

    // Orient base face so i3 is on the negative side
    { HullFace f; f.v[0]=i0; f.v[1]=i1; f.v[2]=i2;
      if (hull_signed_vol6(pts,f,i3)>0) { int t=i1; i1=i2; i2=t; } }

    hull.push_back({i0,i1,i2});
    hull.push_back({i0,i2,i3});
    hull.push_back({i0,i3,i1});
    hull.push_back({i1,i3,i2});

    // --- Incremental expansion ---
    std::vector<int> vis, ha, hb, alive;  // ha/hb = horizon edge endpoints

    for (int pi=0; pi<n; pi++) {
        if (pi==i0||pi==i1||pi==i2||pi==i3) continue;

        vis.clear();
        int nf = (int)hull.size();
        for (int fi=0;fi<nf;fi++)
            if (hull_signed_vol6(pts,hull[fi],pi) > 1e-7f) vis.push_back(fi);
        if (vis.empty()) continue;

        // Horizon edges — O(nv) via edge hash set
        // A directed edge (a→b) is on the horizon iff its reverse (b→a) is not in the visible set.
        static std::unordered_set<long long> vis_edges;
        vis_edges.clear();
        for (int vi : vis)
            for (int ei=0;ei<3;ei++) {
                int ea=hull[vi].v[ei], eb=hull[vi].v[(ei+1)%3];
                vis_edges.insert(((long long)ea << 32) | (unsigned int)eb);
            }
        ha.clear(); hb.clear();
        for (int vi : vis)
            for (int ei=0;ei<3;ei++) {
                int ea=hull[vi].v[ei], eb=hull[vi].v[(ei+1)%3];
                if (!vis_edges.count(((long long)eb << 32) | (unsigned int)ea))
                    { ha.push_back(ea); hb.push_back(eb); }
            }

        // Remove visible faces (mark-and-compact)
        alive.assign(nf, 1);
        for (int vi : vis) alive[vi]=0;
        int w=0;
        for (int fi=0;fi<nf;fi++) if (alive[fi]) hull[w++]=hull[fi];
        hull.resize(w);

        // Stitch horizon to new point
        for (int hi=0;hi<(int)ha.size();hi++) {
            HullFace f; f.v[0]=ha[hi]; f.v[1]=hb[hi]; f.v[2]=pi;
            hull.push_back(f);
        }
    }
    return hull;
}

// ============================================================================
// Random convex hull generators
// ============================================================================

float* gen_gjk_random_hull(int n, unsigned int seed, int* out_count) {
    if (n < 4) n = 4;
    *out_count = n;
    float* v = (float*)malloc(n * 3 * sizeof(float));
    // Seeded LCG for reproducible per-variant shapes
    unsigned int s = seed ^ 0xDEADBEEFu;
    for (int i = 0; i < n; ) {
        s = s * 1664525u + 1013904223u; float x = (int)s * (1.0f/2147483648.0f);
        s = s * 1664525u + 1013904223u; float y = (int)s * (1.0f/2147483648.0f);
        s = s * 1664525u + 1013904223u; float z = (int)s * (1.0f/2147483648.0f);
        float r2 = x*x + y*y + z*z;
        if (r2 < 1e-6f || r2 > 1.0f) continue;   // rejection sample unit sphere
        float inv_r = 1.0f / sqrtf(r2);
        v[i*3+0] = x * inv_r;
        v[i*3+1] = y * inv_r;
        v[i*3+2] = z * inv_r;
        i++;
    }
    return v;
}

int atlas_add_convex_hull(MeshAtlas* atlas, const float* pts_flat, int n) {
    std::vector<HullFace> hull = build_hull(pts_flat, n);
    int nf = (int)hull.size();
    if (nf <= 0) return -1;

    const float (*pts)[3] = (const float (*)[3])pts_flat;
    int nv = nf * 3;
    AtlasVertex* verts = (AtlasVertex*)malloc(nv * sizeof(AtlasVertex));
    uint32_t*    idx   = (uint32_t*)   malloc(nv * sizeof(uint32_t));

    for (int fi = 0; fi < nf; fi++) {
        float fn[3];
        face_normal(pts[hull[fi].v[0]], pts[hull[fi].v[1]], pts[hull[fi].v[2]], fn);
        for (int k = 0; k < 3; k++) {
            memcpy(verts[fi*3+k].pos,    pts[hull[fi].v[k]], 12);
            memcpy(verts[fi*3+k].normal, fn,                 12);
            set_spherical_uv(&verts[fi*3+k]);
        }
        idx[fi*3+0] = fi*3+0; idx[fi*3+1] = fi*3+1; idx[fi*3+2] = fi*3+2;
    }

    int mesh_id = atlas_add_mesh(atlas, verts, nv, idx, nv);
    free(verts); free(idx);
    return mesh_id;
}

// ============================================================================
// Polyhedral inertia (per-axis unit factors)
// ============================================================================

// Shared helper: builds hull and accumulates signed-tet volume + first/second moments.
static void hull_volume_moments(const float* pts_flat, int n,
                                 double* out_vol,
                                 double* out_cx,  double* out_cy,  double* out_cz,
                                 double* out_ix2, double* out_iy2, double* out_iz2) {
    std::vector<HullFace> hull = build_hull(pts_flat, n);
    int nf = (int)hull.size();
    const float (*pts)[3] = (const float (*)[3])pts_flat;

    double vol=0, cx=0, cy=0, cz=0, ix2=0, iy2=0, iz2=0;
    for (int fi = 0; fi < nf; fi++) {
        const float *a=pts[hull[fi].v[0]], *b=pts[hull[fi].v[1]], *c=pts[hull[fi].v[2]];
        double dv = ((double)a[0]*((double)b[1]*(double)c[2]-(double)b[2]*(double)c[1])
                    -(double)a[1]*((double)b[0]*(double)c[2]-(double)b[2]*(double)c[0])
                    +(double)a[2]*((double)b[0]*(double)c[1]-(double)b[1]*(double)c[0])) / 6.0;
        vol += dv;
        cx  += dv * (a[0]+b[0]+c[0]);
        cy  += dv * (a[1]+b[1]+c[1]);
        cz  += dv * (a[2]+b[2]+c[2]);
#define TET_I2(ak,bk,ck) (dv/10.0*((double)(ak)*(ak)+(double)(ak)*(bk)+(double)(bk)*(bk)+(double)(ak)*(ck)+(double)(bk)*(ck)+(double)(ck)*(ck)))
        ix2 += TET_I2(a[0],b[0],c[0]);
        iy2 += TET_I2(a[1],b[1],c[1]);
        iz2 += TET_I2(a[2],b[2],c[2]);
#undef TET_I2
    }
    *out_vol=vol; *out_cx=cx; *out_cy=cy; *out_cz=cz;
    *out_ix2=ix2; *out_iy2=iy2; *out_iz2=iz2;
}

void center_hull_verts(float* pts_flat, int n) {
    double vol, cx, cy, cz, ix2, iy2, iz2;
    hull_volume_moments(pts_flat, n, &vol, &cx, &cy, &cz, &ix2, &iy2, &iz2);
    if (vol < 1e-10) return;
    float com[3] = { (float)(cx/(4.0*vol)), (float)(cy/(4.0*vol)), (float)(cz/(4.0*vol)) };
    for (int i = 0; i < n; i++) {
        pts_flat[i*3+0] -= com[0];
        pts_flat[i*3+1] -= com[1];
        pts_flat[i*3+2] -= com[2];
    }
}

void compute_hull_inertia_k(const float* pts_flat, int n, float* kx, float* ky, float* kz) {
    double vol, cx, cy, cz, ix2, iy2, iz2;
    hull_volume_moments(pts_flat, n, &vol, &cx, &cy, &cz, &ix2, &iy2, &iz2);
    if (vol < 1e-10) { *kx = *ky = *kz = 1.0f/10.0f; return; }
    // If center_hull_verts was called, com≈(0,0,0) and the parallel axis terms vanish.
    // Kept here for correctness if called on un-centred geometry.
    double com[3] = { cx/(4.0*vol), cy/(4.0*vol), cz/(4.0*vol) };
    *kx = (float)((ix2 - vol*com[0]*com[0]) / vol);
    *ky = (float)((iy2 - vol*com[1]*com[1]) / vol);
    *kz = (float)((iz2 - vol*com[2]*com[2]) / vol);
}

// ============================================================================
// Bounding radius
// ============================================================================

// ============================================================================
// OBJ loading
// ============================================================================

int load_obj_shape(MeshAtlas* atlas, const char* path,
                   int* out_render_mesh_id,
                   float*** out_gjk_verts,
                   int** out_gjk_counts, int* out_num_hulls)
{
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = tinyobj::LoadObj(shapes, materials, path);
    if (!err.empty() || shapes.empty()) return 0;

    // Collect positions, UVs, and triangle indices across all shapes
    std::vector<float>    all_pos;
    std::vector<float>    all_uv;
    std::vector<uint32_t> all_idx;
    for (auto& s : shapes) {
        uint32_t base = (uint32_t)(all_pos.size() / 3);
        all_pos.insert(all_pos.end(), s.mesh.positions.begin(), s.mesh.positions.end());
        all_uv.insert(all_uv.end(), s.mesh.texcoords.begin(), s.mesh.texcoords.end());
        for (auto idx : s.mesh.indices)
            all_idx.push_back(base + (uint32_t)idx);
    }
    bool has_uv = (all_uv.size() / 2 >= all_pos.size() / 3);
    int total_verts = (int)(all_pos.size() / 3);
    int total_tris  = (int)(all_idx.size()  / 3);
    if (total_verts < 4 || total_tris < 1) return 0;

    // Normalise so the furthest vertex sits on the unit sphere
    float max_r2 = 0.0f;
    for (int i = 0; i < total_verts; i++) {
        float x = all_pos[i*3], y = all_pos[i*3+1], z = all_pos[i*3+2];
        float r2 = x*x + y*y + z*z;
        if (r2 > max_r2) max_r2 = r2;
    }
    float inv_r = (max_r2 > 1e-8f) ? 1.0f / sqrtf(max_r2) : 1.0f;
    for (auto& v : all_pos) v *= inv_r;

    // Centre vertices at their centroid so body position = centre of mass
    {
        float cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < total_verts; i++) {
            cx += all_pos[i*3]; cy += all_pos[i*3+1]; cz += all_pos[i*3+2];
        }
        cx /= total_verts; cy /= total_verts; cz /= total_verts;
        for (int i = 0; i < total_verts; i++) {
            all_pos[i*3+0] -= cx; all_pos[i*3+1] -= cy; all_pos[i*3+2] -= cz;
        }
    }

    // Build flat-shaded render mesh from original OBJ triangles
    {
        std::vector<AtlasVertex> rverts;
        std::vector<uint32_t>    ridx;
        rverts.reserve(total_tris * 3);
        ridx.reserve(total_tris * 3);
        for (int t = 0; t < total_tris; t++) {
            uint32_t i0 = all_idx[t*3], i1 = all_idx[t*3+1], i2 = all_idx[t*3+2];
            float ax = all_pos[i0*3], ay = all_pos[i0*3+1], az = all_pos[i0*3+2];
            float bx = all_pos[i1*3], by = all_pos[i1*3+1], bz = all_pos[i1*3+2];
            float cx = all_pos[i2*3], cy = all_pos[i2*3+1], cz = all_pos[i2*3+2];
            float ex = bx-ax, ey = by-ay, ez = bz-az;
            float fx = cx-ax, fy = cy-ay, fz = cz-az;
            float nx = ey*fz - ez*fy, ny = ez*fx - ex*fz, nz = ex*fy - ey*fx;
            float nl = sqrtf(nx*nx + ny*ny + nz*nz);
            if (nl > 1e-8f) { float inv = 1.0f/nl; nx*=inv; ny*=inv; nz*=inv; }
            AtlasVertex va = {{ax,ay,az},{nx,ny,nz}};
            AtlasVertex vb = {{bx,by,bz},{nx,ny,nz}};
            AtlasVertex vc = {{cx,cy,cz},{nx,ny,nz}};
            if (has_uv) {
                va.uv[0] = all_uv[i0*2]; va.uv[1] = all_uv[i0*2+1];
                vb.uv[0] = all_uv[i1*2]; vb.uv[1] = all_uv[i1*2+1];
                vc.uv[0] = all_uv[i2*2]; vc.uv[1] = all_uv[i2*2+1];
            }
            uint32_t base = (uint32_t)rverts.size();
            rverts.push_back(va); rverts.push_back(vb); rverts.push_back(vc);
            ridx.push_back(base); ridx.push_back(base+1); ridx.push_back(base+2);
        }
        *out_render_mesh_id = atlas_add_mesh(atlas, rverts.data(), (int)rverts.size(),
                                              ridx.data(), (int)ridx.size());
    }

    // V-HACD convex decomposition for physics
    printf("[OBJ] %s — %d verts, %d tris, running V-HACD...\n", path, total_verts, total_tris);
    VHACD::IVHACD* vhacd = VHACD::CreateVHACD();
    VHACD::IVHACD::Parameters params;
    params.m_maxConvexHulls = 8;
    params.m_resolution     = 100000;
    bool ok = vhacd->Compute(all_pos.data(), (uint32_t)total_verts,
                              all_idx.data(), (uint32_t)total_tris, params);
    if (!ok) { vhacd->Release(); return 0; }

    uint32_t num_hulls = vhacd->GetNConvexHulls();
    printf("[OBJ] V-HACD produced %u hulls\n", num_hulls);

    float** gjk_arr = (float**)malloc(num_hulls * sizeof(float*));
    int*    cnt_arr = (int*)   malloc(num_hulls * sizeof(int));

    for (uint32_t h = 0; h < num_hulls; h++) {
        VHACD::IVHACD::ConvexHull hull;
        vhacd->GetConvexHull(h, hull);
        int nv = (int)hull.m_points.size();
        float* gjk = (float*)malloc(nv * 3 * sizeof(float));
        for (int v = 0; v < nv; v++) {
            gjk[v*3+0] = (float)hull.m_points[v].mX;
            gjk[v*3+1] = (float)hull.m_points[v].mY;
            gjk[v*3+2] = (float)hull.m_points[v].mZ;
        }
        // No per-hull centering: all hulls share the same body-COM frame
        // (all_pos was already centred above, V-HACD preserves that frame)
        gjk_arr[h] = gjk;
        cnt_arr[h] = nv;
    }

    vhacd->Clean();
    vhacd->Release();

    *out_gjk_verts  = gjk_arr;
    *out_gjk_counts = cnt_arr;
    *out_num_hulls  = (int)num_hulls;
    return 1;
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
