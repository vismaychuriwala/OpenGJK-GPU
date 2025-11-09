#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "GJK/openGJK.h"

// C++ callable wrapper for CUDA kernel
void launch_gjk_kernel(gkPolytope* d_bd1, gkPolytope* d_bd2,
                       gkSimplex* d_s, gkFloat* d_distance, int n);

#endif // EXAMPLE_H
