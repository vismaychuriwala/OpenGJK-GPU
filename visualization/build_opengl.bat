@echo off
echo ================================================
echo Building OpenGJK Physics Simulation - OpenGL Version
echo ================================================

REM Parse command line arguments
set BUILD_TYPE=Release
if "%1"=="--debug" (
    set BUILD_TYPE=Debug
    echo Build Configuration: DEBUG
) else (
    echo Build Configuration: RELEASE
)

REM Set compiler flags based on build type
if "%BUILD_TYPE%"=="Debug" (
    set CL_FLAGS=/MDd /Od /Zi
    set NVCC_FLAGS=/MDd /Od /Zi
    set LINK_FLAGS=/DEBUG
) else (
    set CL_FLAGS=/MD /O2
    set NVCC_FLAGS=/MD /O2
    set LINK_FLAGS=
)

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

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Auto-detect GPU architecture (same as build_final.bat)
echo Step 0: Auto-detecting GPU architecture...
nvcc -allow-unsupported-compiler utils\detect_gpu_arch.cu -o build\detect_gpu_arch.exe >nul 2>nul
if %errorlevel% equ 0 (
    build\detect_gpu_arch.exe > build\gpu_arch.tmp
    set /p GPU_ARCH=<build\gpu_arch.tmp
    del build\detect_gpu_arch.exe build\gpu_arch.tmp
    echo Detected GPU architecture: %GPU_ARCH%
) else (
    echo WARNING: Could not auto-detect GPU, defaulting to sm_86
    set GPU_ARCH=sm_86
)
echo.
echo Step 1: Compiling GLAD OpenGL loader...
cl /c %CL_FLAGS% ^
    /I"%GLAD_PATH%\include" ^
    "%GLAD_PATH%\src\glad.c" ^
    /Fo:build\glad.obj

if errorlevel 1 (
    echo Error: Failed to compile GLAD
    exit /b 1
)

echo.
echo Step 2: Compiling input handling module...
cl /c %CL_FLAGS% /EHsc /std:c++17 ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLAD_PATH%\include" ^
    rendering\input.cpp ^
    /Fo:build\input.obj

if errorlevel 1 (
    echo Error: Failed to compile input.cpp
    exit /b 1
)

echo.
echo Step 3: Compiling camera module...
cl /c %CL_FLAGS% /EHsc /std:c++17 ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLM_INCL%" ^
    /I"%GLAD_PATH%\include" ^
    rendering\camera.cpp ^
    /Fo:build\camera.obj

if errorlevel 1 (
    echo Error: Failed to compile camera.cpp
    exit /b 1
)

echo.
echo Step 4: Compiling OpenGL renderer...
cl /c %CL_FLAGS% /EHsc /std:c++17 ^
    /I"%GLAD_PATH%\include" ^
    /I"%GLM_INCL%" ^
    rendering\opengl_renderer.cpp ^
    /Fo:build\opengl_renderer.obj

if errorlevel 1 (
    echo Error: Failed to compile opengl_renderer.cpp
    exit /b 1
)

echo.
echo Step 5: Compiling CUDA physics kernel (GPU GJK)...
nvcc -allow-unsupported-compiler -arch=%GPU_ARCH% -c ^
    -Xcompiler "%NVCC_FLAGS%" ^
    ..\GJK\gpu\openGJK.cu ^
    -I"%CUDA_PATH%\include" ^
    -I"..\GJK\gpu" ^
    -I"..\GJK" ^
    -I"." ^
    -o build\openGJK_gpu.obj

if errorlevel 1 (
    echo Error: Failed to compile openGJK.cu
    exit /b 1
)

echo.
echo Step 6: Compiling CUDA physics integration...
nvcc -allow-unsupported-compiler -arch=%GPU_ARCH% -c ^
    -Xcompiler "%NVCC_FLAGS%" ^
    integrate_final_gjk.cu ^
    -I"%CUDA_PATH%\include" ^
    -I"..\GJK\gpu" ^
    -I"..\GJK" ^
    -I"." ^
    -o build\gpu_gjk_bridge.obj

if errorlevel 1 (
    echo Error: Failed to compile integrate_final_gjk.cu
    exit /b 1
)

echo.
echo Step 7: Compiling CPU GJK (fallback)...
cl /c %CL_FLAGS% ^
    /I"..\GJK\cpu" ^
    ..\GJK\cpu\openGJK.c ^
    /Fo:build\openGJK.obj

if errorlevel 1 (
    echo Error: Failed to compile openGJK.c
    exit /b 1
)

echo.
echo Step 8: Compiling GJK integration layer...
cl /c %CL_FLAGS% /DUSE_CUDA ^
    /I"%CUDA_PATH%\include" ^
    /I"..\GJK\cpu" ^
    gjk_integration.c ^
    /Fo:build\gjk_integration.obj

if errorlevel 1 (
    echo Error: Failed to compile gjk_integration.c
    exit /b 1
)

echo.
echo Step 9: Compiling main OpenGL application...
cl /c %CL_FLAGS% /EHsc /std:c++17 /DUSE_CUDA ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLAD_PATH%\include" ^
    /I"%GLM_INCL%" ^
    /I"%CUDA_PATH%\include" ^
    /I"..\GJK\cpu" ^
    /I"." ^
    main_opengl.cpp ^
    /Fo:build\main_opengl.obj

if errorlevel 1 (
    echo Error: Failed to compile main_opengl.cpp
    exit /b 1
)

echo.
echo Step 10: Linking final executable...
link %LINK_FLAGS% /OUT:gjk_visualizer_opengl.exe ^
    build\main_opengl.obj ^
    build\glad.obj ^
    build\input.obj ^
    build\camera.obj ^
    build\opengl_renderer.obj ^
    build\gjk_integration.obj ^
    build\openGJK.obj ^
    build\gpu_gjk_bridge.obj ^
    build\openGJK_gpu.obj ^
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
echo Build successful! [%BUILD_TYPE%]
echo Executable: gjk_visualizer_opengl.exe
echo ================================================
echo.
echo To run the simulation:
echo     gjk_visualizer_opengl.exe
echo.
echo Usage:
echo     build_opengl.bat        - Build optimized release version (default)
echo     build_opengl.bat --debug - Build debug version with symbols
echo.
