cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0063 NEW)
project(OpenGJK-GPU_test
    LANGUAGES C CUDA CXX
    VERSION 1.0.0
    DESCRIPTION "GPU-accelerated openGJK"
)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Werror")
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Werror")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wunused-macros")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wunused-macros")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel")
    # Original openGJK has no Intel-specific flags
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    # Original openGJK has no MSVC-specific flags (uses CMake defaults)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(UNIX)

add_subdirectory(GJK)
add_subdirectory(examples/usage)

include_directories(.)

set(headers
    "examples/testing_helpers.hpp"
    "examples/common/timer.h"
    "examples/gpu/example.h"
    "examples/cpu/example.h"
    )

set(sources
    "examples/main.cpp"
    "examples/gpu/example.cu"
    "examples/cpu/example.cpp"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} GJK)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

message(STATUS "Version     : " ${CMAKE_PROJECT_VERSION})
message(STATUS "Build type  : " ${CMAKE_BUILD_TYPE})
