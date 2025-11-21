if(CMAKE_HIP_COMPILER)
    set_source_files_properties(sfc_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(sfc_obj OBJECT sfc_gpu.cu)
    target_include_directories(sfc_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(sfc_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()
