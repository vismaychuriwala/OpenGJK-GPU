#include "openGJK.h"
#include "gjk_integration.h"
#include <stdlib.h>
#include <math.h>

gkPolytope convert_to_poly(const GJK_Shape* shape)
{
    gkPolytope p;
    p.numpoints = shape->num_vertices;
    p.coord = (gkFloat*)malloc(sizeof(gkFloat) * p.numpoints * 3);

    for (int i = 0; i < p.numpoints; i++)
    {
        p.coord[i*3 + 0] = shape->vertices[i].x + shape->position.x;
        p.coord[i*3 + 1] = shape->vertices[i].y + shape->position.y;
        p.coord[i*3 + 2] = shape->vertices[i].z + shape->position.z;
    }

    return p;
}

bool openGJK_collision_cpu(const GJK_Shape* shapeA, const GJK_Shape* shapeB)
{
    gkSimplex simplex;
    gkPolytope A = convert_to_poly(shapeA);
    gkPolytope B = convert_to_poly(shapeB);

    gkFloat dist = compute_minimum_distance(A, B, &simplex);

    free(A.coord);
    free(B.coord);

    return dist <= 1e-4;   // treat very small distances as collision
}
