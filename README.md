# OpenGJK-GPU

CUDA implementation of [openGJK](https://github.com/MattiaMontanari/openGJK).

## Port Summary

### Key Changes from Original openGJK

1. **Memory Layout**: Changed from double-pointer `gkFloat** coord` to flattened array `gkFloat* coord` for GPU compatibility
   - Added `getCoord(body, index, component)` macro for array access: `body->coord[(index) * 3 + (component)]`

2. **Function Decorators**: Added `__device__` to all 17 helper functions (determinant, crossProduct, S1D, S2D, S3D, support, subalgorithm, W0D-W3D, compute_witnesses)

3. **Kernel Signature**: Changed from CPU function to `__global__` kernel:
   ```cuda
   __global__ void compute_minimum_distance(gkPolytope* polytopes1, gkPolytope* polytopes2,
                                             gkSimplex* simplices, gkFloat* distances, int n);
   ```

4. **Performance Optimization**: Changed to copy-by-value for polytopes and simplices to keep data in registers/local memory instead of global memory

5. **Precision Macros**: Added `gkSqrt` macro for float/double precision handling

6. **Code Structure**: GPU wrapper in `GJK::GPU` namespace with built-in CUDA timing support

## Test Results

Example with 100,000 polytope pairs (9 vertices each):

```
Testing 100000 polytope pairs
Distance between bodies (first pair): 3.653650
GPU time: 1.0719 ms
Witnesses (first pair): (1.025173, 1.490318, 0.255463) and (-1.025173, -1.490318, -0.255463)
```

The CUDA implementation produces correct results matching the original CPU version.

* **Tested On:**  
  * OS: Windows 11
  * CPU: AMD Ryzen 7 5800H with Radeon Graphics (8C/16T, 3.2GHz base)
  * RAM: 32GB DDR4
  * GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB GDDR6)
  * CUDA Toolkit: 13.0
  * Driver Version: 581.15