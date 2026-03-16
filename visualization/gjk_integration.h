#ifndef GJK_INTEGRATION_H
#define GJK_INTEGRATION_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float z;
} Vector3f;

typedef struct {
    Vector3f* vertices;   // local-space vertices
    int       num_vertices;
    Vector3f  position;   // world-space translation
} GJK_Shape;

// Shape helpers
GJK_Shape create_cube_shape(Vector3f position, float size);
GJK_Shape create_sphere_shape(Vector3f position, float radius);
void      free_shape(GJK_Shape* shape);

#ifdef __cplusplus
}
#endif

#endif // GJK_INTEGRATION_H
