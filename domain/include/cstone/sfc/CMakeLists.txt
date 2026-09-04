if(CSTONE_WITH_HIP)
    set_source_files_properties(sfc_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    add_library(sfc_obj OBJECT sfc_gpu.cu)
    target_include_directories(sfc_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(sfc_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()
