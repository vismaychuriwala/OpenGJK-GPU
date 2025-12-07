project(example_lib_opengjk_ce
    LANGUAGES C CXX CUDA
    VERSION 1.0.0
)

# Force C++ compiler for C files (required for GPU API which uses C++ namespaces)
file(GLOB_RECURSE CFILES "${CMAKE_CURRENT_SOURCE_DIR}/main.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)

add_executable(example_lib_opengjk_ce 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    ${CMAKE_SOURCE_DIR}/examples/gpu/example.cu
    ${CMAKE_SOURCE_DIR}/examples/cpu/example.cpp
)

# Link to GJK library (contains GPU kernels)
target_link_libraries(example_lib_opengjk_ce GJK)

# Add include directories
target_include_directories(example_lib_opengjk_ce PRIVATE ${CMAKE_SOURCE_DIR})

# Copy input files for this example after build (to the same directory as the executable)
add_custom_command(
    TARGET example_lib_opengjk_ce POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/userP.dat
    $<TARGET_FILE_DIR:example_lib_opengjk_ce>/userP.dat
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/userQ.dat
    $<TARGET_FILE_DIR:example_lib_opengjk_ce>/userQ.dat
)
