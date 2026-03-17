# OpenGJK Visualization

GPU-accelerated rigid body physics simulation using GJK collision detection, rendered with OpenGL.

## Requirements

- **CMake 3.18+**
- **Visual Studio 2022** with C++ build tools
- **CUDA Toolkit 13.0+**
- **GLFW 3.4** — extract to `C:\glfw-3.4.bin.WIN64`
- **GLAD / GLM** — vendored in `include/`, no install needed

## Build

### From `visualization/` (standalone)

**CMake GUI:**
1. Source: `visualization/`
2. Build: `visualization/build`
3. Configure → set generator to `Visual Studio 17 2022`, platform `x64`
4. Adjust `GLFW_DIR` cache variable if your GLFW path differs
5. Generate → Open Project → build `gjk_visualizer`

**Command line:**
```cmd
cd visualization
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target gjk_visualizer
```

### From repo root

```cmd
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target gjk_visualizer
```

The executable is placed in `build/Release/` (or `build/bin/` when built from root). Shaders are copied automatically post-build.

## Run

```cmd
build\Release\gjk_visualizer.exe
```

## Controls

| Key / Input | Action |
|---|---|
| WASD | Move camera |
| Q / E | Up / Down |
| Mouse (left drag) | Rotate view |
| Scroll | Zoom |

## Configuration

Edit `sim_config.h` to change simulation parameters:

| Constant | Default | Description |
|---|---|---|
| `NUM_OBJECTS` | 1000 | Number of rigid bodies |
| `RESTITUTION` | 0.9 | Bounce coefficient |
| `ANGULAR_DAMPING` | 0.995 | Spin decay per frame |
| `MAX_SPATIAL_GRID_SIZE` | 30 | Broad-phase grid resolution |
