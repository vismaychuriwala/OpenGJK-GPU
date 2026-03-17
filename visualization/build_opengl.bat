@echo off
echo ================================================
echo Building OpenGJK Physics Simulation - OpenGL
echo ================================================

set BUILD_TYPE=Release
if "%1"=="--debug" (
    set BUILD_TYPE=Debug
    echo Build Configuration: DEBUG
) else (
    echo Build Configuration: RELEASE
)

if "%BUILD_TYPE%"=="Debug" (
    set CL_FLAGS=/MDd /Od /Zi
    set NVCC_FLAGS=/MDd /Od /Zi
    set LINK_FLAGS=/DEBUG
) else (
    set CL_FLAGS=/MD /O2
    set NVCC_FLAGS=/MD /O2
    set LINK_FLAGS=
)

for /f "delims=" %%i in ('where nvcc 2^>nul') do (
    set NVCC_EXE=%%i
    goto :found_nvcc
)
echo Error: nvcc not found in PATH.
exit /b 1
:found_nvcc
set CUDA_PATH=%NVCC_EXE:\bin\nvcc.exe=%
echo Detected CUDA path: %CUDA_PATH%

set GLFW_PATH=C:\glfw-3.4.bin.WIN64
set GLM_PATH=C:\glm
if exist "%GLM_PATH%\include\glm\glm.hpp" (
    set GLM_INCL=%GLM_PATH%\include
) else if exist "%GLM_PATH%\glm\glm.hpp" (
    set GLM_INCL=%GLM_PATH%
) else if exist "%GLM_PATH%\glm.hpp" (
    set GLM_INCL=%GLM_PATH%
) else (
    echo Error: glm header not found in %GLM_PATH%
    exit /b 1
)
set GLAD_PATH=glad

if not exist build mkdir build

echo.
echo Step 1: Auto-detecting GPU architecture...
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
echo Step 2: Compiling GLAD...
cl /c %CL_FLAGS% /I"%GLAD_PATH%\include" "%GLAD_PATH%\src\glad.c" /Fo:build\glad.obj
if errorlevel 1 ( echo Error: GLAD & exit /b 1 )

echo.
echo Step 3: Compiling input...
cl /c %CL_FLAGS% /EHsc /std:c++17 /I"%GLFW_PATH%\include" /I"%GLAD_PATH%\include" ^
    rendering\input.cpp /Fo:build\input.obj
if errorlevel 1 ( echo Error: input.cpp & exit /b 1 )

echo.
echo Step 4: Compiling camera...
cl /c %CL_FLAGS% /EHsc /std:c++17 /I"%GLFW_PATH%\include" /I"%GLM_INCL%" /I"%GLAD_PATH%\include" ^
    rendering\camera.cpp /Fo:build\camera.obj
if errorlevel 1 ( echo Error: camera.cpp & exit /b 1 )

echo.
echo Step 5: Compiling mesh builder...
cl /c %CL_FLAGS% /EHsc /std:c++17 ^
    rendering\mesh_builder.cpp /Fo:build\mesh_builder.obj
if errorlevel 1 ( echo Error: mesh_builder.cpp & exit /b 1 )

echo.
echo Step 6: Compiling OpenGL renderer...
cl /c %CL_FLAGS% /EHsc /std:c++17 /I"%GLAD_PATH%\include" /I"%GLM_INCL%" /I"." ^
    rendering\opengl_renderer.cpp /Fo:build\opengl_renderer.obj
if errorlevel 1 ( echo Error: opengl_renderer.cpp & exit /b 1 )

echo.
echo Step 7: Compiling GPU GJK kernel...
nvcc -allow-unsupported-compiler -arch=%GPU_ARCH% -c ^
    -Xcompiler "%NVCC_FLAGS%" ^
    ..\GJK\gpu\openGJK.cu ^
    -I"%CUDA_PATH%\include" ^
    -I"..\GJK\gpu" ^
    -I"..\GJK" ^
    -I"." ^
    -o build\openGJK_gpu.obj
if errorlevel 1 ( echo Error: openGJK.cu & exit /b 1 )

echo.
echo Step 8: Compiling CUDA physics sim...
nvcc -allow-unsupported-compiler -arch=%GPU_ARCH% -c ^
    -Xcompiler "%NVCC_FLAGS%" ^
    integrate_final_gjk.cu ^
    -I"%CUDA_PATH%\include" ^
    -I"..\GJK\gpu" ^
    -I"..\GJK" ^
    -I"." ^
    -o build\gpu_gjk_bridge.obj
if errorlevel 1 ( echo Error: integrate_final_gjk.cu & exit /b 1 )

echo.
echo Step 9: Compiling main...
cl /c %CL_FLAGS% /EHsc /std:c++17 ^
    /I"%GLFW_PATH%\include" ^
    /I"%GLAD_PATH%\include" ^
    /I"%GLM_INCL%" ^
    /I"%CUDA_PATH%\include" ^
    /I"." ^
    main_opengl.cpp /Fo:build\main_opengl.obj
if errorlevel 1 ( echo Error: main_opengl.cpp & exit /b 1 )

echo.
echo Step 10: Linking...
link %LINK_FLAGS% /OUT:gjk_visualizer_opengl.exe ^
    build\main_opengl.obj ^
    build\glad.obj ^
    build\input.obj ^
    build\camera.obj ^
    build\mesh_builder.obj ^
    build\opengl_renderer.obj ^
    build\gpu_gjk_bridge.obj ^
    build\openGJK_gpu.obj ^
    "%GLFW_PATH%\lib-vc2022\glfw3.lib" ^
    opengl32.lib gdi32.lib user32.lib shell32.lib ^
    "%CUDA_PATH%\lib\x64\cudart.lib"
if errorlevel 1 ( echo Error: Linking & exit /b 1 )

echo.
echo ================================================
echo Build successful! [%BUILD_TYPE%]
echo Executable: gjk_visualizer_opengl.exe
echo ================================================
