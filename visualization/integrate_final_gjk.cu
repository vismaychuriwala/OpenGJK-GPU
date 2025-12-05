// integrate_final_gjk.cu
#include "gpu_gjk_interface.h"
#include "../GJK/gpu/warpParallelGJK.h"   // For gkPolytope, gkSimplex and kernel
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simple GPU GJK context: now stores full shapes
struct GPU_GJK_Context {
    bool     initialized;
    int      device_id;
    int      max_objects;
    int      num_objects;
    // We just keep a copy of the shapes (pointers to their vertex arrays)
    GJK_Shape shapes[64];   // max_objects <= 64 is plenty for the visualizer
};

bool gpu_gjk_init(GPU_GJK_Context** context, int max_objects)
{
    printf("Initializing GPU WarpParallelGJK bridge...\n");

    *context = (GPU_GJK_Context*)malloc(sizeof(GPU_GJK_Context));
    if (!*context) {
        fprintf(stderr, "gpu_gjk_init: failed to allocate context\n");
        return false;
    }

    (*context)->initialized = true;
    (*context)->device_id   = 0;
    (*context)->max_objects = (max_objects < 64) ? max_objects : 64;
    (*context)->num_objects = 0;

    printf("GPU GJK context ready (max_objects = %d)\n", (*context)->max_objects);
    return true;
}

void gpu_gjk_cleanup(GPU_GJK_Context** context)
{
    if (context && *context) {
        free(*context);
        *context = NULL;
    }
}

bool gpu_gjk_register_shape(GPU_GJK_Context* context,
                            const GJK_Shape* shape,
                            int object_id)
{
    if (!context || !shape) return false;
    if (object_id < 0 || object_id >= context->max_objects) return false;

    // Copy the struct; this copies the vertex pointer, not the vertex data.
    context->shapes[object_id] = *shape;

    if (object_id + 1 > context->num_objects)
        context->num_objects = object_id + 1;

    printf("Registered shape %d: num_vertices = %d, pos = (%.2f, %.2f, %.2f)\n",
           object_id,
           shape->num_vertices,
           shape->position.x, shape->position.y, shape->position.z);
    return true;
}

bool gpu_gjk_update_position(GPU_GJK_Context* context,
                             int object_id,
                             Vector3f new_position)
{
    if (!context) return false;
    if (object_id < 0 || object_id >= context->num_objects) return false;

    context->shapes[object_id].position = new_position;
    return true;
}

// Helper: build a gkPolytope from a GJK_Shape (cube) on the GPU
static bool build_polytope_from_shape(const GJK_Shape* shape,
                                      gkPolytope* poly,
                                      gkFloat** d_coord_out)
{
    if (!shape || !poly || !d_coord_out) return false;

    const int num = shape->num_vertices;
    if (num <= 0) return false;

    // Host coordinate buffer: [x0, y0, z0, x1, y1, z1, ...]
    const int coordCount = num * 3;
    gkFloat* h_coord = (gkFloat*)malloc(sizeof(gkFloat) * coordCount);
    if (!h_coord) return false;

    for (int i = 0; i < num; ++i) {
        float wx = shape->vertices[i].x + shape->position.x;
        float wy = shape->vertices[i].y + shape->position.y;
        float wz = shape->vertices[i].z + shape->position.z;

        h_coord[i * 3 + 0] = (gkFloat)wx;
        h_coord[i * 3 + 1] = (gkFloat)wy;
        h_coord[i * 3 + 2] = (gkFloat)wz;
    }

    // Device coordinates
    gkFloat* d_coord = NULL;
    cudaMalloc((void**)&d_coord, sizeof(gkFloat) * coordCount);
    cudaMemcpy(d_coord, h_coord,
               sizeof(gkFloat) * coordCount,
               cudaMemcpyHostToDevice);

    free(h_coord);

    poly->numpoints = num;
    poly->coord     = d_coord;
    poly->s[0] = poly->s[1] = poly->s[2] = 0.0f;
    poly->s_idx = 0;

    *d_coord_out = d_coord;
    return true;
}

// Main function: for each pair, run WarpParallelGJK kernel once
bool gpu_gjk_batch_check(GPU_GJK_Context* context,
                         int* object_pairs,
                         int num_pairs,
                         bool* results)
{
    if (!context || !context->initialized) return false;
    if (!object_pairs || !results || num_pairs <= 0) return false;

    const float collisionEpsilon = 1e-4f;  // distance threshold ~0 => collision
    const int threadsPerBlock = 16;        // must match THREADS_PER_COMPUTATION in warpParallelGJK.cu

    for (int p = 0; p < num_pairs; ++p) {
        int idA = object_pairs[2 * p + 0];
        int idB = object_pairs[2 * p + 1];

        if (idA < 0 || idB < 0 ||
            idA >= context->num_objects ||
            idB >= context->num_objects) {
            results[p] = false;
            continue;
        }

        const GJK_Shape* shapeA = &context->shapes[idA];
        const GJK_Shape* shapeB = &context->shapes[idB];

        // Build polytopes on GPU for this pair
        gkPolytope h_polyA, h_polyB;
        gkSimplex  h_simplex;
        memset(&h_simplex, 0, sizeof(gkSimplex));

        gkFloat* d_coordA = NULL;
        gkFloat* d_coordB = NULL;

        if (!build_polytope_from_shape(shapeA, &h_polyA, &d_coordA) ||
            !build_polytope_from_shape(shapeB, &h_polyB, &d_coordB)) {
            fprintf(stderr, "Failed to build polytopes for pair %d\n", p);
            results[p] = false;
            if (d_coordA) cudaFree(d_coordA);
            if (d_coordB) cudaFree(d_coordB);
            continue;
        }

        // Device-side containers
        gkPolytope* d_polyA = NULL;
        gkPolytope* d_polyB = NULL;
        gkSimplex*  d_simplex = NULL;
        gkFloat*    d_distance = NULL;

        cudaMalloc((void**)&d_polyA, sizeof(gkPolytope));
        cudaMalloc((void**)&d_polyB, sizeof(gkPolytope));
        cudaMalloc((void**)&d_simplex, sizeof(gkSimplex));
        cudaMalloc((void**)&d_distance, sizeof(gkFloat));

        cudaMemcpy(d_polyA, &h_polyA, sizeof(gkPolytope), cudaMemcpyHostToDevice);
        cudaMemcpy(d_polyB, &h_polyB, sizeof(gkPolytope), cudaMemcpyHostToDevice);
        cudaMemcpy(d_simplex, &h_simplex, sizeof(gkSimplex), cudaMemcpyHostToDevice);

        // One GJK computation (n = 1), one block of 16 threads
        compute_minimum_distance_warp_parallel<<<1, threadsPerBlock>>>(
            d_polyA, d_polyB, d_simplex, d_distance, 1);

        cudaDeviceSynchronize();

        gkFloat h_distance = 0.0f;
        cudaMemcpy(&h_distance, d_distance, sizeof(gkFloat), cudaMemcpyDeviceToHost);

        // Distance ~ 0 => collision
        results[p] = (fabsf((float)h_distance) <= collisionEpsilon);

        // Cleanup for this pair
        cudaFree(d_polyA);
        cudaFree(d_polyB);
        cudaFree(d_simplex);
        cudaFree(d_distance);
        cudaFree(d_coordA);
        cudaFree(d_coordB);
    }

    printf("GPU WarpParallelGJK - processed %d pairs\n", num_pairs);
    return true;
}

// Convenience wrapper used by main.c
bool gpu_gjk_collision_check(GPU_GJK_Context* context,
                             const GJK_Shape* shapeA,
                             const GJK_Shape* shapeB)
{
    (void)shapeA;
    (void)shapeB;

    int pairs[2] = {0, 1};
    bool results[1] = {false};

    if (!gpu_gjk_batch_check(context, pairs, 1, results))
        return false;

    return results[0];
}

#ifdef __cplusplus
}
#endif
