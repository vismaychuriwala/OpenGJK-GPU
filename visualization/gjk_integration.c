#include "gjk_integration.h"
#include <stdlib.h>
#include <math.h>

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
    shape.num_vertices = 12;
    shape.vertices = (Vector3f*)malloc(sizeof(Vector3f) * 12);

    const float phi = 1.618033988749895f;
    const float a = 1.0f;
    const float b = phi;
    const float norm = radius / sqrtf(a*a + b*b);
    const float na = a * norm;
    const float nb = b * norm;

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
