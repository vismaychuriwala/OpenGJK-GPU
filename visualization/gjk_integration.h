#ifndef GJK_INTEGRATION_H
#define GJK_INTEGRATION_H

#include <stdbool.h>

// Structure matching your GJK implementation
typedef struct {
    float x, y, z;
} Vector3f;

typedef struct {
    Vector3f* vertices;
    int num_vertices;
    Vector3f position;
    char name[32];
} GJK_Shape;

// GJK function declarations
bool gjk_collision_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB);
bool gjk_distance_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB, float* distance);

// Helper functions
GJK_Shape create_cube_shape(Vector3f position, float size);
GJK_Shape create_tetrahedron_shape(Vector3f position, float size);
void free_shape(GJK_Shape* shape);

// Visualization helpers
void draw_gjk_debug_info(const GJK_Shape* shapeA, const GJK_Shape* shapeB, bool collision);

#endif