function(sph_add_gpu_object target)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEFINITIONS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "sph_add_gpu_object: SOURCES required")
    endif()

    if(SPH_EXA_WITH_HIP)
        set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()

    if(SPH_EXA_WITH_CUDA OR SPH_EXA_WITH_HIP)
        add_library(${target} OBJECT ${ARG_SOURCES})
        target_include_directories(${target} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)

        if(ARG_DEFINITIONS)
            target_compile_definitions(${target} PRIVATE ${ARG_DEFINITIONS})
        endif()

        if(SPH_EXA_WITH_HIP)
            target_link_libraries(${target} PRIVATE roc::rocthrust hip::hipcub)
        endif()
    endif()
endfunction()
