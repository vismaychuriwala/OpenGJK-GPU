// gpu_gjk_interface.h
#pragma once
#include "gjk_integration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration only - no struct definition
typedef struct GPU_GJK_Context GPU_GJK_Context;

// Function declarations only
bool gpu_gjk_init(GPU_GJK_Context** context, int max_objects);
void gpu_gjk_cleanup(GPU_GJK_Context** context);
bool gpu_gjk_register_shape(GPU_GJK_Context* context, const GJK_Shape* shape, int object_id);
bool gpu_gjk_update_position(GPU_GJK_Context* context, int object_id, Vector3f new_position);
bool gpu_gjk_batch_check(GPU_GJK_Context* context, int* object_pairs, int num_pairs, bool* results);
bool gpu_gjk_collision_check(GPU_GJK_Context* context, const GJK_Shape* shapeA, const GJK_Shape* shapeB);
bool openGJK_collision_cpu(const GJK_Shape* shapeA, const GJK_Shape* shapeB);

#ifdef __cplusplus
}
#endif