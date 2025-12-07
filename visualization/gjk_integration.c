#include "gjk_integration.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <string.h>

// OpenGJK CPU implementation
#include "../GJK/cpu/openGJK.h"

// =========================
// Basic vector utilities
// =========================

static float dot(Vector3f a, Vector3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vector3f subtract(Vector3f a, Vector3f b) {
    return (Vector3f){a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vector3f negate(Vector3f v) {
    return (Vector3f){-v.x, -v.y, -v.z};
}

static Vector3f cross(Vector3f a, Vector3f b) {
    return (Vector3f){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static float lengthSquared(Vector3f v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

static Vector3f tripleProduct(Vector3f a, Vector3f b, Vector3f c) {
    return cross(a, cross(b, c));
}

// =========================
// Simplex type (kept for possible debug uses, but not used by OpenGJK path)
// =========================

typedef struct {
    Vector3f points[4];
    int count;
} Simplex;

// =========================
// OpenGJK bridge helpers
// =========================

// Convert our GJK_Shape into OpenGJK's gkPolytope in world space
static int build_polytope_from_shape(const GJK_Shape* shape, gkPolytope* poly) {
    if (!shape || shape->num_vertices <= 0 || !shape->vertices) {
        return 0;
    }

    memset(poly, 0, sizeof(gkPolytope));

    poly->numpoints = shape->num_vertices;
    poly->coord = (gkFloat*)malloc(sizeof(gkFloat) * poly->numpoints * 3);
    if (!poly->coord) {
        return 0;
    }

    for (int i = 0; i < poly->numpoints; ++i) {
        gkFloat x = (gkFloat)(shape->vertices[i].x + shape->position.x);
        gkFloat y = (gkFloat)(shape->vertices[i].y + shape->position.y);
        gkFloat z = (gkFloat)(shape->vertices[i].z + shape->position.z);

        poly->coord[3 * i + 0] = x;
        poly->coord[3 * i + 1] = y;
        poly->coord[3 * i + 2] = z;
    }

    // Initialize support point to first vertex
    poly->s[0] = poly->coord[0];
    poly->s[1] = poly->coord[1];
    poly->s[2] = poly->coord[2];
    poly->s_idx = 0;

    return 1;
}

// =========================
// CPU OpenGJK wrappers
// =========================

// This is what main.c was trying to call
bool openGJK_collision_cpu(const GJK_Shape* shapeA, const GJK_Shape* shapeB) {
    if (!shapeA || !shapeB) return false;

    gkPolytope pa, pb;
    if (!build_polytope_from_shape(shapeA, &pa)) return false;
    if (!build_polytope_from_shape(shapeB, &pb)) {
        free(pa.coord);
        return false;
    }

    gkSimplex simplex;
    memset(&simplex, 0, sizeof(gkSimplex));

    gkFloat dist = compute_minimum_distance(pa, pb, &simplex);

    free(pa.coord);
    free(pb.coord);

    const gkFloat eps = (gkFloat)1e-6;
    return dist <= eps;
}

bool openGJK_distance_cpu(const GJK_Shape* shapeA, const GJK_Shape* shapeB, float* distance_out) {
    if (!shapeA || !shapeB) return false;

    gkPolytope pa, pb;
    if (!build_polytope_from_shape(shapeA, &pa)) return false;
    if (!build_polytope_from_shape(shapeB, &pb)) {
        free(pa.coord);
        return false;
    }

    gkSimplex simplex;
    memset(&simplex, 0, sizeof(gkSimplex));

    gkFloat dist = compute_minimum_distance(pa, pb, &simplex);

    free(pa.coord);
    free(pb.coord);

    if (distance_out) {
        *distance_out = (float)dist;
    }

    const gkFloat eps = (gkFloat)1e-6;
    return dist <= eps;
}

// Keep old names as thin wrappers so any existing usage still works
bool gjk_collision_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB) {
    return openGJK_collision_cpu(shapeA, shapeB);
}

bool gjk_distance_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB, float* distance) {
    return openGJK_distance_cpu(shapeA, shapeB, distance);
}

// =========================
// Shape helpers 
// =========================

GJK_Shape create_cube_shape(Vector3f position, float size) {
    GJK_Shape shape;
    shape.position = position;
    shape.num_vertices = 8;
    shape.vertices = (Vector3f*)malloc(sizeof(Vector3f) * 8);

    float half = size / 2.0f;

    shape.vertices[0] = (Vector3f){ -half, -half, -half };
    shape.vertices[1] = (Vector3f){  half, -half, -half };
    shape.vertices[2] = (Vector3f){  half,  half, -half };
    shape.vertices[3] = (Vector3f){ -half,  half, -half };
    shape.vertices[4] = (Vector3f){ -half, -half,  half };
    shape.vertices[5] = (Vector3f){  half, -half,  half };
    shape.vertices[6] = (Vector3f){  half,  half,  half };
    shape.vertices[7] = (Vector3f){ -half,  half,  half };

    return shape;
}

GJK_Shape create_sphere_shape(Vector3f position, float radius) {
    GJK_Shape shape;
    shape.position = position;
    shape.num_vertices = 12;  // Icosahedron (12 vertices)
    shape.vertices = (Vector3f*)malloc(sizeof(Vector3f) * 12);

    // Golden ratio for icosahedron
    const float phi = 1.618033988749895f;  // (1 + sqrt(5)) / 2
    const float a = 1.0f;
    const float b = phi;

    // Normalize to sphere radius
    const float norm = radius / sqrtf(a*a + b*b);
    const float na = a * norm;
    const float nb = b * norm;

    // 12 vertices of icosahedron on unit sphere scaled to radius
    shape.vertices[0]  = (Vector3f){  0,  na,  nb };
    shape.vertices[1]  = (Vector3f){  0,  na, -nb };
    shape.vertices[2]  = (Vector3f){  0, -na,  nb };
    shape.vertices[3]  = (Vector3f){  0, -na, -nb };
    shape.vertices[4]  = (Vector3f){  na,  nb,  0 };
    shape.vertices[5]  = (Vector3f){  na, -nb,  0 };
    shape.vertices[6]  = (Vector3f){ -na,  nb,  0 };
    shape.vertices[7]  = (Vector3f){ -na, -nb,  0 };
    shape.vertices[8]  = (Vector3f){  nb,  0,  na };
    shape.vertices[9]  = (Vector3f){ -nb,  0,  na };
    shape.vertices[10] = (Vector3f){  nb,  0, -na };
    shape.vertices[11] = (Vector3f){ -nb,  0, -na };

    return shape;
}

void free_shape(GJK_Shape* shape) {
    if (shape->vertices) {
        free(shape->vertices);
        shape->vertices = NULL;
    }
}

void draw_gjk_debug_info(const GJK_Shape* shapeA, const GJK_Shape* shapeB, bool collision) {
    // Placeholder for future debug visualization
    (void)shapeA;
    (void)shapeB;
    (void)collision;
}
