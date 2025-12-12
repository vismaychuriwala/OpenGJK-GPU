project(usage_examples
    LANGUAGES C CXX CUDA
    VERSION 1.0.0
)

# Force C++ compiler for C files (required for GPU API which uses C++ namespaces)
file(GLOB_RECURSE GJK_FILES "${CMAKE_CURRENT_SOURCE_DIR}/GJKUsage.c")
file(GLOB_RECURSE EPA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/EPAUsage.c")
SET_SOURCE_FILES_PROPERTIES(${GJK_FILES} ${EPA_FILES} PROPERTIES LANGUAGE CXX)

# GJK Usage Example
add_executable(GJKUsage 
    ${CMAKE_CURRENT_SOURCE_DIR}/GJKUsage.c
    ${CMAKE_SOURCE_DIR}/examples/gpu/example.cu
    ${CMAKE_SOURCE_DIR}/examples/cpu/example.cpp
)

# Link to GJK library (contains GPU kernels)
target_link_libraries(GJKUsage GJK)

# Add include directories
target_include_directories(GJKUsage PRIVATE ${CMAKE_SOURCE_DIR})

# Copy input files for GJK example after build (to the same directory as the executable)
add_custom_command(
    TARGET GJKUsage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/userP.dat
    $<TARGET_FILE_DIR:GJKUsage>/userP.dat
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/userQ.dat
    $<TARGET_FILE_DIR:GJKUsage>/userQ.dat
)

# EPA Usage Example
add_executable(EPAUsage 
    ${CMAKE_CURRENT_SOURCE_DIR}/EPAUsage.c
    ${CMAKE_SOURCE_DIR}/examples/gpu/example.cu
    ${CMAKE_SOURCE_DIR}/examples/cpu/example.cpp
)

# Link to GJK library (contains GPU kernels)
target_link_libraries(EPAUsage GJK)

# Add include directories
target_include_directories(EPAUsage PRIVATE ${CMAKE_SOURCE_DIR})
