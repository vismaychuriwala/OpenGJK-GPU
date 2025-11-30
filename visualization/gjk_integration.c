#include "gjk_integration.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

// Improved GJK implementation
typedef struct {
    Vector3f points[4];
    int count;
} Simplex;

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

// Support function for Minkowski difference
static Vector3f support(const GJK_Shape* shapeA, const GJK_Shape* shapeB, Vector3f direction) {
    // Find furthest point in direction for shapeA
    float maxDotA = -FLT_MAX;
    Vector3f bestA = {0};
    
    for (int i = 0; i < shapeA->num_vertices; i++) {
        Vector3f world_vertex = {
            shapeA->vertices[i].x + shapeA->position.x,
            shapeA->vertices[i].y + shapeA->position.y,
            shapeA->vertices[i].z + shapeA->position.z
        };
        float dotProduct = dot(world_vertex, direction);
        if (dotProduct > maxDotA) {
            maxDotA = dotProduct;
            bestA = world_vertex;
        }
    }
    
    // Find furthest point in opposite direction for shapeB
    float maxDotB = -FLT_MAX;
    Vector3f bestB = {0};
    Vector3f negDirection = negate(direction);
    
    for (int i = 0; i < shapeB->num_vertices; i++) {
        Vector3f world_vertex = {
            shapeB->vertices[i].x + shapeB->position.x,
            shapeB->vertices[i].y + shapeB->position.y,
            shapeB->vertices[i].z + shapeB->position.z
        };
        float dotProduct = dot(world_vertex, negDirection);
        if (dotProduct > maxDotB) {
            maxDotB = dotProduct;
            bestB = world_vertex;
        }
    }
    
    // Return Minkowski difference
    return subtract(bestA, bestB);
}

// Check if simplex contains origin and update direction
static int simplexContainsOrigin(Simplex* s, Vector3f* direction) {
    Vector3f a = s->points[s->count - 1];
    Vector3f ao = negate(a);
    
    if (s->count == 3) {
        // Triangle case
        Vector3f b = s->points[2];
        Vector3f c = s->points[1];
        Vector3f ab = subtract(b, a);
        Vector3f ac = subtract(c, a);
        
        Vector3f abPerp = tripleProduct(ac, ab, ab);
        Vector3f acPerp = tripleProduct(ab, ac, ac);
        
        if (dot(abPerp, ao) > 0) {
            // Origin is outside AB, remove C
            s->points[1] = s->points[2];
            s->count = 2;
            *direction = abPerp;
        } else if (dot(acPerp, ao) > 0) {
            // Origin is outside AC, remove B
            s->points[2] = s->points[1];
            s->points[1] = s->points[0];
            s->count = 2;
            *direction = acPerp;
        } else {
            // Origin is inside triangle, we have collision
            return 1;
        }
    } else if (s->count == 2) {
        // Line case
        Vector3f b = s->points[0];
        Vector3f ab = subtract(b, a);
        
        if (dot(ab, ao) > 0) {
            // Origin is outside AB in AB direction
            *direction = tripleProduct(ab, ao, ab);
        } else {
            // Origin is outside AB in A direction
            s->points[0] = s->points[1];
            s->count = 1;
            *direction = ao;
        }
    }
    
    return 0;
}

// Real GJK collision detection
bool gjk_collision_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB) {
    Simplex simplex;
    simplex.count = 0;
    
    // Initial direction
    Vector3f direction = {1, 0, 0};
    
    // First support point
    simplex.points[simplex.count++] = support(shapeA, shapeB, direction);
    
    // Negate direction for next search
    direction = negate(simplex.points[0]);
    
    int maxIterations = 32;
    for (int i = 0; i < maxIterations; i++) {
        // Get new support point in current direction
        Vector3f newPoint = support(shapeA, shapeB, direction);
        
        // If we didn't move past origin, no collision
        if (dot(newPoint, direction) <= 0) {
            return false;
        }
        
        // Add new point to simplex
        simplex.points[simplex.count++] = newPoint;
        
        // Check if simplex contains origin
        if (simplexContainsOrigin(&simplex, &direction)) {
            return true;
        }
        
        // Prevent infinite loop
        if (simplex.count > 3) {
            break;
        }
    }
    
    return false;
}

bool gjk_distance_check(const GJK_Shape* shapeA, const GJK_Shape* shapeB, float* distance) {
    bool collision = gjk_collision_check(shapeA, shapeB);
    
    // Calculate center-to-center distance for display
    float dx = shapeA->position.x - shapeB->position.x;
    float dy = shapeA->position.y - shapeB->position.y;
    float dz = shapeA->position.z - shapeB->position.z;
    *distance = sqrt(dx * dx + dy * dy + dz * dz);
    
    return collision;
}

// Shape creation functions (keep these the same)
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

void free_shape(GJK_Shape* shape) {
    if (shape->vertices) {
        free(shape->vertices);
        shape->vertices = NULL;
    }
}

void draw_gjk_debug_info(const GJK_Shape* shapeA, const GJK_Shape* shapeB, bool collision) {
    // Placeholder for future debug visualization
}