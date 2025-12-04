if(CMAKE_HIP_COMPILER)
    set_source_files_properties(rebalance_gpu.cu source_center_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(source_center_gpu_obj OBJECT rebalance_gpu.cu source_center_gpu.cu)
    target_include_directories(source_center_gpu_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(source_center_gpu_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()

