# OpenGJK-GPU

CUDA implementation of [openGJK](https://github.com/MattiaMontanari/openGJK) and the EPA algorithm for high-performance collision detection on NVIDIA GPUs.

**Contributors:** [Marcus Hedlund](https://github.com/mhedlund7), [Vismay Churiwala](https://vismaychuriwala.com/), [Cindy Wei](https://www.linkedin.com/in/cindy-wei-7ba778227/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)

https://github.com/user-attachments/assets/62ae9925-616c-447a-969d-25487e2ee11c

*1000 Polytopes Collision Simulation at 60 FPS*
## Overview

OpenGJK-GPU provides efficient GPU-accelerated implementations of the **GJK (Gilbert-Johnson-Keerthi)** and **EPA (Expanding Polytope Algorithm)** algorithms for collision detection between convex polytopes.

**GJK** is a fast iterative algorithm for computing the minimum distance between two convex polytopes in 3D space. It works by iteratively building a simplex within the Minkowski difference of the two polytopes, converging toward the closest point to the origin.

**EPA** is a complementary algorithm that computes penetration depth and witness points when two polytopes are overlapping. While GJK can detect collisions, EPA determines how deeply objects penetrate each other and where the contact points are—critical for collision response in physics engines.

Our GPU implementations leverage two levels of parallelism: processing multiple collision pairs simultaneously and parallelizing computation within each collision across a warp. This approach achieves speedups of up to 37x over CPU implementations when handling many polytope pairs.

## Basic API

The API provides two main functions for collision detection:

### GJK Distance Computation

```cpp
void GJK::GPU::computeDistances(const int n,
                                const gkPolytope* bd1,
                                const gkPolytope* bd2,
                                gkSimplex* simplices,
                                gkFloat* distances);
```

Computes minimum distance between `n` pairs of polytopes. Returns distances in the `distances` array (0.0 indicates collision).

### EPA Collision Information

```cpp
void GJK::GPU::computeCollisionInformation(const int n,
                                          const gkPolytope* bd1,
                                          const gkPolytope* bd2,
                                          gkSimplex* simplices,
                                          gkFloat* distances,
                                          gkFloat* witness1,
                                          gkFloat* witness2,
                                          gkFloat* contact_normals = nullptr);
```

Computes penetration depth, witness points, and contact normals for colliding polytopes. Takes pre-computed simplices and distances from GJK as input.

**Data Format:**
- Polytopes use flattened coordinate arrays: `[x0, y0, z0, x1, y1, z1, ...]`
- All memory allocation and GPU transfers are handled internally

## Getting Started

### Prerequisites

**Required:**
- Git
- C/C++ compiler (GCC, Clang, or MSVC)
- CMake (version 3.18 or higher)
- **CUDA Toolkit** (version 11.0 or higher)
- **NVIDIA GPU** with CUDA support

Clone the repository:

```bash
git clone https://github.com/vismaychuriwala/OpenGJK-GPU.git
cd OpenGJK-GPU
```

Then use these commands to build and run our examples:

### Building

**On Windows:**
```cmd
cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build build --config Release
cd build\bin\Release
```

**On Linux/Mac:**
```bash
cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build build
cd build/bin
```

### Running Examples

The repository includes two usage examples in `examples/usage/`:

#### GJK Example (`GJKUsage`)

Demonstrates basic GJK distance computation:

**Windows:**
```cmd
GJKUsage.exe
```

**Linux/Mac:**
```bash
./GJKUsage
```

**Expected output:**
```
Distance between bodies 3.653650
Witnesses: (1.025173, 1.490318, 0.255463) and (-1.025173, -1.490318, -0.255463)
```

#### EPA Example (`EPAUsage`)

Demonstrates sequential GJK and EPA usage for collision information:

**Windows:**
```cmd
EPAUsage.exe
```

**Linux/Mac:**
```bash
./EPAUsage
```

**Expected output:**
```
Penetration depth: 1.500000
Witness point on cube 1: (1.000000, 0.500000, 0.707107)
Witness point on cube 2: (-0.500000, 0.500000, 0.707107)
Contact normal (from cube 1 to cube 2): (1.000000, -0.000000, 0.000000)
```

### Precision Configuration

To switch between 32-bit (float) and 64-bit (double) precision, edit `GJK/common.h`:

- **32-bit**: `#define USE_32BITS`
- **64-bit**: `//#define USE_32BITS`

## Implementation Details

### Key Features

#### 1. Warp-Parallel Execution

Our GPU kernels leverage CUDA warp-level parallelism to achieve significant performance improvements. Each GJK collision pair is processed by a dedicated half warp (16 threads) while EPA uses a full warp (32 threads), with threads collaborating to parallelize computationally expensive operations:

- **GJK Algorithm**: Uses 16 threads per collision to parallelize expensive support function calls and sub-distance algorithms. Threads within a warp use `__shfl_sync()` operations to share data and perform parallel reductions, eliminating the need for shared memory synchronization.

- **EPA Algorithm**: Uses all 32 threads of a warp to parallelize face normal computation and closest-face searches. Face normals and distances are computed in parallel across threads, with warp shuffles used for efficient reduction to find the global minimum.

This two-level parallelism (across collisions and within each collision) maximizes GPU utilization and minimizes memory access overhead.

|![GPU GJK Block Diagram](images/GPUGJKImprovedBlockDiagram.png)|
|:--:|
|*GPU GJK Implementation Block Diagram - showing warp-parallel execution structure*|

#### 2. Separate GJK/EPA API

The API design separates GJK and EPA into independent functions, providing flexibility for different use cases:

- **`computeDistances()`**: Performs GJK distance computation only. Fast and efficient for collision detection when penetration depth is not needed.

- **`computeCollisionInformation()`**: Performs EPA using pre-computed GJK results. Can be called conditionally only when collisions are detected, avoiding unnecessary computation for separated objects.

This separation allows users to optimize their collision detection pipeline by running EPA only when needed, reducing computational overhead for non-colliding objects.

|![GJK EPA Interface](images/GJKEPAInterface.png)|
|:--:|
|*GJK/EPA Interface Diagram - showing the sequential workflow*|

#### 3. Automatic Memory Management

All GPU memory operations are handled internally by the wrapper functions, providing a clean and simple API:

- Automatic allocation and deallocation of device memory for polytopes, simplices, and results
- Efficient host-to-device and device-to-host memory transfers
- Proper cleanup on function exit, preventing memory leaks
- No manual CUDA memory management required from the user

This abstraction allows users to focus on their application logic without worrying about low-level GPU memory management details.

### Performance

|![GJK Runtime GPU VS CPU Varying Number of Vertices](images/chart1_vertices_runtime.png)|![GJK Runtime GPU VS CPU Varying Number of Polytopes](images/chart2_polytopes_runtime.png)|
|:--:|:--:|
|*GJK Runtime GPU VS CPU Varying Number of Vertices*|*GJK Runtime GPU VS CPU Varying Number of Polytopes*|

Performance benchmarks demonstrate significant GPU acceleration across varying problem sizes. When testing 1000 polytope pairs with increasing vertex complexity, the GPU achieves speedups ranging from **7x** at 50 vertices to a peak of **37x** at 1000 vertices per polytope, before decreasing to **16x** at 5000 vertices. This indicates optimal GPU efficiency at moderate vertex counts where the workload is substantial enough to leverage parallelism without becoming memory-bound. When varying the number of polytope pairs while keeping vertex count fixed at 500, the GPU advantage scales consistently with problem size—starting close to **2x speedup** at 50 polytopes and reaching **27x speedup** at 50,000 polytopes. This scaling behavior demonstrates the GPU's strength in handling massively parallel workloads where each polytope pair collision check can be processed independently. Overall, GPU-accelerated GJK is particularly beneficial for large-scale collision detection scenarios typical in physics simulations and game engines, where many polytope pairs must be processed simultaneously.


## Visualization

https://github.com/user-attachments/assets/70eb00b7-2d5a-4ff5-84ec-ee096b90c1f6

*20,000 Polytopes Collision Simulation at 60 FPS - demonstrating the algorithm's scalability*

The repository includes a real-time physics simulation visualizer demonstrating collision detection with thousands of polytopes at 60 FPS. The simulation employs **spatial grid subdivision** for broad-phase culling, dynamically generating collision pairs each frame to minimize unnecessary GJK computations. Two visualization options are available: **OpenGL** (recommended, faster) and **Raylib** (legacy).

### OpenGL Version (Recommended)

#### Requirements

**Compiler:**
- **Visual Studio 2022+** with C++ build tools
- **CUDA Toolkit 13.0+** (default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`)
- Must use **x64 Native Tools Command Prompt**

**Libraries:**
- **GLFW 3.4** - Download from https://www.glfw.org/download.html, extract to `C:\glfw-3.4.bin.WIN64`
- **GLM** - Download from https://github.com/g-truc/glm/releases, extract to `C:\glm`
- **GLAD** - Already included in `glad/` folder (generated from https://glad.dav1d.de/)

#### Setup Steps

1. **Install Visual Studio 2022+** with C++ build tools
2. **Install CUDA Toolkit 13.0+** (ensure it's in the default location or update paths in build scripts)
3. **Download and extract GLFW 3.4:**
   - Download from https://www.glfw.org/download.html
   - Extract to `C:\glfw-3.4.bin.WIN64`
4. **Download and extract GLM:**
   - Download from https://github.com/g-truc/glm/releases
   - Extract to `C:\glm`
5. **Open x64 Native Tools Command Prompt** (from Visual Studio)

#### Build & Run

```cmd
cd visualization
build_opengl.bat
gjk_visualizer_opengl.exe
```

#### Controls

- **WASD** - Move camera
- **Q/E** - Up/Down
- **Mouse** - Rotate view (hold left button)
- **Scroll** - Zoom
- **F** - Toggle wireframe
- **R** - Reset camera
- **ESC** - Exit

**Configuration**: Edit `visualization/sim_config.h` to adjust simulation parameters (object count, grid size, physics constants).

### Raylib Version (Legacy)

For the legacy Raylib version, see `visualization/README.md` for setup instructions.

## References

* [CPU OpenGJK](https://github.com/MattiaMontanari/openGJK)
* [Improving the GJK algorithm for faster and more reliable distance queries between convex objects](https://web.archive.org/web/20200320045859id_/https://ora.ox.ac.uk/objects/uuid:69c743d9-73de-4aff-8e6f-b4dd7c010907/download_file?safe_filename=GJK.PDF&file_format=application%2Fpdf&type_of_work=Journal+article)
* [A Strange But Elegant Approach to a Surprisingly Hard Problem](https://www.youtube.com/watch?v=ajv46BSqcK4)
* [The Gilbert–Johnson–Keerthi algorithm explained as simply as possible](https://computerwebsite.net/writing/gjk)
* [Winterdev:](https://winter.dev/) [GJK,](https://www.youtube.com/watch?v=MDusDn8oTSE) [EPA](https://www.youtube.com/watch?v=0XQ2FSz3EK8)
* [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
