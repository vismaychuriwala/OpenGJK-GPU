@echo off
echo ================================================
echo Building OpenGJK Physics Simulation - OpenGL Version
echo ================================================

REM Set paths to dependencies
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set GLFW_PATH=C:\glfw-3.4.bin.WIN64
set GLM_PATH=C:\glm
REM Determine actual include dir for GLM (accepts C:\glm\include, C:\glm\glm, or C:\glm)
if exist "%GLM_PATH%\include\glm\glm.hpp" (
    set GLM_INCL=%GLM_PATH%\include
) else if exist "%GLM_PATH%\glm\glm.hpp" (
    set GLM_INCL=%GLM_PATH%
) else if exist "%GLM_PATH%\glm.hpp" (
    set GLM_INCL=%GLM_PATH%
) else (
    echo Error: glm header not found in %GLM_PATH%
    echo Expected one of:
    echo   %GLM_PATH%\include\glm\glm.hpp
    echo   %GLM_PATH%\glm\glm.hpp
    echo   %GLM_PATH%\glm.hpp
    exit /b 1
)
set GLAD_PATH=glad

REM Auto-detect GPU architecture (same as build_final.bat)
echo Step 0: Auto-detecting GPU architecture...
nvcc -allow-unsupported-compiler detect_gpu_arch.cu -o detect_gpu_arch.exe >nul 2>nul
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
echo Step 1: Compiling GLAD OpenGL loader...
cl /c /MD ^
    /I"%GLAD_PATH%\include" ^
    "%GLAD_PATH%\src\glad.c" ^
    /Fo:glad.obj

if errorlevel 1 (
    echo Error: Failed to compile GLAD
    exit /b 1
)

echo.
echo Step 2: Compiling input handling module...
cl /c /MD /EHsc /std:c++17 ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLAD_PATH%\include" ^
    input.cpp ^
    /Fo:input.obj

if errorlevel 1 (
    echo Error: Failed to compile input.cpp
    exit /b 1
)

echo.
echo Step 3: Compiling camera module...
cl /c /MD /EHsc /std:c++17 ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLM_INCL%" ^
    /I"%GLAD_PATH%\include" ^
    camera.cpp ^
    /Fo:camera.obj

if errorlevel 1 (
    echo Error: Failed to compile camera.cpp
    exit /b 1
)

echo.
echo Step 4: Compiling OpenGL renderer...
cl /c /MD /EHsc /std:c++17 ^
    /I"%GLAD_PATH%\include" ^
    /I"%GLM_INCL%" ^
    opengl_renderer.cpp ^
    /Fo:opengl_renderer.obj

if errorlevel 1 (
    echo Error: Failed to compile opengl_renderer.cpp
    exit /b 1
)

echo.
echo Step 5: Compiling CUDA physics kernel (GPU GJK)...
nvcc -allow-unsupported-compiler -arch=%GPU_ARCH% -c ^
    -Xcompiler "/MD" ^
    ..\GJK\gpu\openGJK.cu ^
    -I"%CUDA_PATH%\include" ^
    -I"..\GJK\gpu" ^
    -I"..\GJK" ^
    -I"." ^
    -o openGJK_gpu.obj

if errorlevel 1 (
    echo Error: Failed to compile openGJK.cu
    exit /b 1
)

echo.
echo Step 6: Compiling CUDA physics integration...
nvcc -allow-unsupported-compiler -arch=%GPU_ARCH% -c ^
    -Xcompiler "/MD" ^
    integrate_final_gjk.cu ^
    -I"%CUDA_PATH%\include" ^
    -I"..\GJK\gpu" ^
    -I"..\GJK" ^
    -I"." ^
    -o gpu_gjk_bridge.obj

if errorlevel 1 (
    echo Error: Failed to compile integrate_final_gjk.cu
    exit /b 1
)

echo.
echo Step 7: Compiling CPU GJK (fallback)...
cl /c /MD ^
    /I"..\GJK\cpu" ^
    ..\GJK\cpu\openGJK.c ^
    /Fo:openGJK.obj

if errorlevel 1 (
    echo Error: Failed to compile openGJK.c
    exit /b 1
)

echo.
echo Step 8: Compiling GJK integration layer...
cl /c /MD /DUSE_CUDA ^
    /I"%CUDA_PATH%\include" ^
    /I"..\GJK\cpu" ^
    gjk_integration.c ^
    /Fo:gjk_integration.obj

if errorlevel 1 (
    echo Error: Failed to compile gjk_integration.c
    exit /b 1
)

echo.
echo Step 9: Compiling main OpenGL application...
cl /c /MD /EHsc /std:c++17 /DUSE_CUDA ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLAD_PATH%\include" ^
    /I"%GLM_INCL%" ^
    /I"%CUDA_PATH%\include" ^
    /I"..\GJK\cpu" ^
    /I"." ^
    main_opengl.cpp ^
    /Fo:main_opengl.obj

if errorlevel 1 (
    echo Error: Failed to compile main_opengl.cpp
    exit /b 1
)

echo.
echo Step 10: Linking final executable...
link /OUT:gjk_visualizer_opengl.exe ^
    main_opengl.obj ^
    glad.obj ^
    input.obj ^
    camera.obj ^
    opengl_renderer.obj ^
    gjk_integration.obj ^
    openGJK.obj ^
    gpu_gjk_bridge.obj ^
    openGJK_gpu.obj ^
    "%GLFW_PATH%\lib-vc2022\glfw3.lib" ^
    opengl32.lib ^
    gdi32.lib ^
    user32.lib ^
    shell32.lib ^
    "%CUDA_PATH%\lib\x64\cudart.lib"

if errorlevel 1 (
    echo Error: Linking failed
    exit /b 1
)

echo.
echo ================================================
echo Build successful!
echo Executable: gjk_visualizer_opengl.exe
echo ================================================
echo.
echo To run the simulation:
echo     gjk_visualizer_opengl.exe
echo.
