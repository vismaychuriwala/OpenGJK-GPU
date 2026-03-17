# Standalone build support: these only apply when this is the top-level CMakeLists.txt.
# When included via add_subdirectory() from the root they are skipped.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(gjk_visualizer LANGUAGES C CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# When built standalone (not via root add_subdirectory), build GJK ourselves
if(NOT TARGET GJK)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../GJK ${CMAKE_CURRENT_BINARY_DIR}/GJK)
    set(_GJK_INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
else()
    set(_GJK_INCLUDE_ROOT ${CMAKE_SOURCE_DIR})
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

# ---- GLAD (vendored in include/glad) ----
add_library(glad_viz STATIC include/glad/src/glad.c)
target_include_directories(glad_viz PUBLIC include/glad/include)

# ---- GLFW (pre-built binary) ----
set(GLFW_DIR "C:/glfw-3.4.bin.WIN64" CACHE PATH "GLFW installation root")
add_library(glfw_viz STATIC IMPORTED GLOBAL)
set_target_properties(glfw_viz PROPERTIES
    IMPORTED_LOCATION             "${GLFW_DIR}/lib-vc2022/glfw3.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_DIR}/include")

# ---- GLM (vendored in include/) ----
add_library(glm_viz INTERFACE)
target_include_directories(glm_viz INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Visualizer executable ----
add_executable(gjk_visualizer
    main_opengl.cpp
    rendering/input.cpp
    rendering/camera.cpp
    rendering/mesh_builder.cpp
    rendering/opengl_renderer.cpp
    integrate_final_gjk.cu
)

# Match the same CUDA arch logic as GJK/CMakeLists.txt
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(gjk_visualizer PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(gjk_visualizer PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(gjk_visualizer PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_compile_options(gjk_visualizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

target_include_directories(gjk_visualizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_GJK_INCLUDE_ROOT}/GJK/gpu
    ${_GJK_INCLUDE_ROOT}/GJK
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gjk_visualizer PRIVATE
    GJK
    glad_viz
    glfw_viz
    glm_viz
    OpenGL::GL
    CUDA::cudart
    gdi32 user32 shell32
)

# Copy shaders next to the executable after build
add_custom_command(TARGET gjk_visualizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:gjk_visualizer>/shaders
)
