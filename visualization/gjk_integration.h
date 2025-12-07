#ifndef GJK_INTEGRATION_H
#define GJK_INTEGRATION_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simple 3D vector used by visualizer & GJK integration
typedef struct {
    float x;
    float y;
    float z;
} Vector3f;

// Shape definition used by both CPU and GPU integration code
typedef struct {
    Vector3f* vertices;   // local-space vertices
    int       num_vertices;
    Vector3f  position;   // world-space translation
} GJK_Shape;

// CPU-side APIs

// Old names (now backed by OpenGJK)
bool gjk_collision_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB);
bool gjk_distance_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB, float* distance);

// New explicit OpenGJK wrappers (what main.c is calling)
bool openGJK_collision_cpu(const GJK_Shape* shapeA, const GJK_Shape* shapeB);
bool openGJK_distance_cpu(const GJK_Shape* shapeA, const GJK_Shape* shapeB, float* distance);

// Shape helpers
GJK_Shape create_cube_shape(Vector3f position, float size);
GJK_Shape create_sphere_shape(Vector3f position, float radius);  // Icosahedron approximation
void      free_shape(GJK_Shape* shape);

// Optional debug draw hook
void draw_gjk_debug_info(const GJK_Shape* shapeA, const GJK_Shape* shapeB, bool collision);

#ifdef __cplusplus
}
#endif

#endif // GJK_INTEGRATION_H
