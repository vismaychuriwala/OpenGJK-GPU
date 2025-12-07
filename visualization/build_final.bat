@echo off
echo Building Simple GPU-Only WarpParallelGJK Version...
echo.

set RAYLIB_INCLUDE=/IC:\raylib\include
set RAYLIB_LIB=C:\raylib\lib\raylib.lib
set WIN_LIBS=opengl32.lib gdi32.lib winmm.lib user32.lib kernel32.lib advapi32.lib shell32.lib
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

echo Step 0: Auto-detecting GPU architecture...
nvcc detect_gpu_arch.cu -o detect_gpu_arch.exe >nul 2>nul
if %errorlevel% equ 0 (
    detect_gpu_arch.exe > gpu_arch.tmp
    set /p GPU_ARCH=<gpu_arch.tmp
    del detect_gpu_arch.exe gpu_arch.tmp
    echo Detected GPU architecture: %GPU_ARCH%
) else (
    echo WARNING: Could not auto-detect GPU, defaulting to sm_86
    set GPU_ARCH=sm_86
)
echo.

echo Step 1: Checking for CUDA integration file...
dir integrate_final_gjk.cu

echo.
echo Step 2: Compiling GPU integration bridge (integrate_final_gjk.cu)...
nvcc -arch=%GPU_ARCH% -c integrate_final_gjk.cu -o gpu_gjk_bridge.obj ^
    -I"%CUDA_PATH%\include" -I"..\GJK\gpu" -I"..\GJK" -I"."

if %errorlevel% neq 0 (
    echo.
    echo Fallback: trying gpu_gjk_bridge.cu instead...
    nvcc -arch=%GPU_ARCH% -c gpu_gjk_bridge.cu -o gpu_gjk_bridge.obj ^
        -I"%CUDA_PATH%\include" -I"..\GJK\gpu" -I"..\GJK" -I"."
)

echo.
echo Step 2b: Compiling WarpParallelGJK kernel...
nvcc -arch=%GPU_ARCH% -c ..\GJK\gpu\warpParallelGJK.cu -o warpParallelGJK.obj ^
    -I"%CUDA_PATH%\include" -I"..\GJK\gpu" -I"..\GJK" -I"."

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Failed to compile warpParallelGJK.cu ❌
    echo Make sure it exists at ..\GJK\gpu\warpParallelGJK.cu
    echo ========================================
    echo.
    pause
    exit /b 1
)

echo.
echo Step 3: Building final executable...
cl /MD /DUSE_CUDA %RAYLIB_INCLUDE% -I"%CUDA_PATH%\include" -I"..\GJK\cpu" ^
    /Fe:gjk_visualizer_gpu_only.exe ^
    main.c gjk_integration.c ..\GJK\cpu\openGJK.c gpu_gjk_bridge.obj warpParallelGJK.obj ^
    %RAYLIB_LIB% %WIN_LIBS% "%CUDA_PATH%\lib\x64\cudart.lib"

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
