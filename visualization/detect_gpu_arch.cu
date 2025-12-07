// Simple CUDA program to detect GPU compute capability
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Print in format: sm_XY
    printf("sm_%d%d", prop.major, prop.minor);

    return 0;
}
