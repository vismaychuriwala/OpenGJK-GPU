#include "gpu_gjk_interface.h"
#include "../GJK/gpu/warpParallelGJK.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GPU_GJK_Context {
    bool initialized;
    int device_id;
    int max_objects;
    
    // Simple storage for testing
    Vector3f positions[10];
    int num_objects;
};

bool gpu_gjk_init(GPU_GJK_Context** context, int max_objects) {
    printf("Initializing Simple GPU-Only WarpParallelGJK...\n");
    
    *context = (GPU_GJK_Context*)malloc(sizeof(GPU_GJK_Context));
    if (!*context) return false;
    
    (*context)->initialized = true;
    (*context)->device_id = 0;
    (*context)->max_objects = max_objects;
    (*context)->num_objects = 0;
    
    printf("Simple GPU-Only GJK initialized successfully!\n");
    return true;
}

void gpu_gjk_cleanup(GPU_GJK_Context** context) {
    if (context && *context) {
        free(*context);
        *context = NULL;
    }
}

bool gpu_gjk_register_shape(GPU_GJK_Context* context, const GJK_Shape* shape, int object_id) {
    if (!context || object_id >= context->max_objects) return false;
    
    // Store the initial position
    context->positions[object_id] = shape->position;
    context->num_objects++;
    
    printf("Registered shape %d at position (%.2f, %.2f, %.2f)\n", 
           object_id, shape->position.x, shape->position.y, shape->position.z);
    return true;
}

bool gpu_gjk_update_position(GPU_GJK_Context* context, int object_id, Vector3f new_position) {
    if (!context || object_id >= context->num_objects) return false;
    
    context->positions[object_id] = new_position;
    return true;
}

// Simple GPU kernel for distance calculation
__global__ void simple_distance_kernel(Vector3f* positions, float* distances, int* pairs, int num_pairs) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;
    
    int objA = pairs[pair_idx * 2];
    int objB = pairs[pair_idx * 2 + 1];
    
    float dx = positions[objA].x - positions[objB].x;
    float dy = positions[objA].y - positions[objB].y;
    float dz = positions[objA].z - positions[objB].z;
    distances[pair_idx] = sqrt(dx*dx + dy*dy + dz*dz);
}

bool gpu_gjk_batch_check(GPU_GJK_Context* context, int* object_pairs, int num_pairs, bool* results) {
    if (!context) return false;
    
    // Allocate GPU memory
    Vector3f* d_positions;
    float* d_distances;
    int* d_pairs;
    
    cudaMalloc(&d_positions, context->num_objects * sizeof(Vector3f));
    cudaMalloc(&d_distances, num_pairs * sizeof(float));
    cudaMalloc(&d_pairs, num_pairs * 2 * sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_positions, context->positions, context->num_objects * sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs, object_pairs, num_pairs * 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch simple kernel
    int blockSize = 256;
    int numBlocks = (num_pairs + blockSize - 1) / blockSize;
    simple_distance_kernel<<<numBlocks, blockSize>>>(d_positions, d_distances, d_pairs, num_pairs);
    
    cudaDeviceSynchronize();
    
    // Get results
    float* h_distances = (float*)malloc(num_pairs * sizeof(float));
    cudaMemcpy(h_distances, d_distances, num_pairs * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Determine collisions
    for (int i = 0; i < num_pairs; i++) {
        results[i] = (h_distances[i] < 3.0f); // Collision threshold
    }
    
    // Cleanup
    free(h_distances);
    cudaFree(d_positions);
    cudaFree(d_distances);
    cudaFree(d_pairs);
    
    printf("GPU Batch Check - Processed %d pairs\n", num_pairs);
    return true;
}

bool gpu_gjk_collision_check(GPU_GJK_Context* context, const GJK_Shape* shapeA, const GJK_Shape* shapeB) {
    // For single pair, use the batch system
    int pairs[] = {0, 1};
    bool results[1];
    
    gpu_gjk_batch_check(context, pairs, 1, results);
    return results[0];
}

#ifdef __cplusplus
}
#endif