# OpenGJK-GPU

CUDA implementation of [openGJK](https://github.com/MattiaMontanari/openGJK).

## Port Summary

### Key Changes from Original openGJK

1. **Memory Layout**: Changed from double-pointer `gkFloat** coord` to flattened array `gkFloat* coord` for GPU compatibility
   - Added `getCoord(body, index, component)` macro for array access: `body->coord[(index) * 3 + (component)]`

2. **Function Decorators**: Added `__device__` to all 17 helper functions (determinant, crossProduct, S1D, S2D, S3D, support, subalgorithm, W0D-W3D, compute_witnesses)

3. **Kernel Signature**: Changed from CPU function to `__global__` kernel:
   ```cuda
   __global__ void compute_minimum_distance(gkPolytope* polytypes1, gkPolytope* polytypes2,
                                             gkSimplex* simplices, gkFloat* distances, int n);
   ```

4. **Performance Optimization**: Changed to copy-by-value for polytopes and simplices to keep data in registers/local memory instead of global memory

5. **Precision Macros**: Added `gkSqrt` macro for float/double precision handling

## Test Results

Built and ran the basic example with userP.dat and userQ.dat (9 vertices each):

```
Distance between bodies 3.653650
Witnesses: (1.025173, 1.490318, 0.255463) and (-1.025173, -1.490318, -0.255463)
```

The CUDA implementation produces correct results matching the original CPU version.
