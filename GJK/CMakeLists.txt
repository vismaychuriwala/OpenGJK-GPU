set(headers
    "gpu/openGJK.h"
    "cpu/openGJK.h"
    "common.h"
    )

set(sources
    "gpu/openGJK.cu"
    "cpu/openGJK.c"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})
add_library(GJK ${sources} ${headers})

if(UNIX)
    target_link_libraries(GJK PRIVATE m)
endif()
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(GJK PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(GJK PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(GJK PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(GJK PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
