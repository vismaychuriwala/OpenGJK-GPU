

## 1. Install Dependencies

### **Raylib**
- Download Raylib for Windows from: https://github.com/raysan5/raylib/releases
- Extract to `C:\raylib\`
- Should have: `C:\raylib\include\` and `C:\raylib\lib\raylib.lib`

### **CUDA Toolkit**
- Download from: https://developer.nvidia.com/cuda-downloads
- Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` (or later)
- If using a different path, update `CUDA_PATH` in `build_final.bat`
- **Note**: Using Visual Studio 2026 requires the `-allow-unsupported-compiler` flag (already added to build script)

### **Visual Studio**
- Requires Visual Studio 2022 or 2026 with C++ build tools
- Must use "x64 Native Tools Command Prompt" (not regular Command Prompt)
- The build script expects `cl.exe` (MSVC compiler) and `nvcc` (CUDA compiler) in PATH

## 2. Build & Run

```cmd
# Open "x64 Native Tools Command Prompt for VS 2022"
# (Finds compiler automatically)

cd OpenGJK-GPU\visualization
build_final.bat
gjk_visualizer_gpu_only.exe
```

