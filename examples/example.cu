#include "GJK/openGJK.h"
#include <cuda_runtime.h>

// Wrapper function to launch the CUDA kernel
void launch_gjk_kernel(gkPolytope* d_bd1, gkPolytope* d_bd2,
                       gkSimplex* d_s, gkFloat* d_distance, int n) {
    compute_minimum_distance<<<1, 1>>>(d_bd1, d_bd2, d_s, d_distance, n);
    cudaDeviceSynchronize();
}
