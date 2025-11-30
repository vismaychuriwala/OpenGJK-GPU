@echo off
echo Building Simple GPU-Only WarpParallelGJK Version...
echo.

set RAYLIB_INCLUDE=/IC:\raylib\include
set RAYLIB_LIB=C:\raylib\lib\raylib.lib
set WIN_LIBS=opengl32.lib gdi32.lib winmm.lib user32.lib kernel32.lib advapi32.lib shell32.lib
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

echo Step 1: Checking for CUDA files...
dir *.cu*

echo Step 2: Compiling GPU-only integration bridge...
nvcc -arch=sm_89 -c integrate_final_gjk.cu -o gpu_gjk_bridge.obj -I"%CUDA_PATH%\include" -I"..\GJK\gpu" -I"..\GJK" -I"."

if %errorlevel% neq 0 (
    echo.
    echo Trying alternative CUDA file...
    nvcc -arch=sm_89 -c gpu_gjk_bridge.cu -o gpu_gjk_bridge.obj -I"%CUDA_PATH%\include" -I"..\GJK\gpu" -I"..\GJK" -I"."
)

echo Step 3: Building final executable...
cl /MD /DUSE_CUDA %RAYLIB_INCLUDE% -I"%CUDA_PATH%\include" /Fe:gjk_visualizer_gpu_only.exe main.c gjk_integration.c gpu_gjk_bridge.obj %RAYLIB_LIB% %WIN_LIBS% "%CUDA_PATH%\lib\x64\cudart.lib"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo GPU-ONLY BUILD SUCCESSFUL! 🎉
    echo ========================================
    echo Run with: gjk_visualizer_gpu_only.exe
    echo.
) else (
    echo.
    echo ========================================
    echo BUILD FAILED! ❌
    echo ========================================
    echo Check for errors above
    echo.
)

pause