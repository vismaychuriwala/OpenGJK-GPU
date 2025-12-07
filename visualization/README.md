# OpenGJK Visualization

Two visualization options: **OpenGL** (recommended, faster) and **Raylib** (legacy).

## Requirements

### Compiler
- **Visual Studio 2022/2026** with C++ build tools
- **CUDA Toolkit 13.0+** (default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`)
- Must use **x64 Native Tools Command Prompt**

### Libraries

**OpenGL version:**
- **GLFW 3.4** - Download from https://www.glfw.org/download.html, extract to `C:\glfw-3.4.bin.WIN64`
- **GLM** - Download from https://github.com/g-truc/glm/releases, extract to `C:\glm`
- **GLAD** - Already included in `glad/` folder (generated from https://glad.dav1d.de/)

**Raylib version (legacy):**
- **Raylib** - Download from https://github.com/raysan5/raylib/releases, extract to `C:\raylib` (needs `include/` and `lib/raylib.lib`)

## Build & Run

### OpenGL Version (Recommended)
```cmd
cd visualization
build_opengl.bat
gjk_visualizer_opengl.exe
```

### Raylib Version (Legacy)
```cmd
cd visualization\old_raylib
build_final.bat
gjk_visualizer_gpu_only.exe
```

## Controls
- **WASD** - Move camera
- **Q/E** - Up/Down
- **Mouse** - Rotate view (hold left button)
- **Scroll** - Zoom
- **SPACE** - Reset simulation
- **W key** - Toggle wireframe (OpenGL only)
