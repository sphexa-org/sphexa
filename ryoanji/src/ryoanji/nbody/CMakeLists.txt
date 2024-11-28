if(CMAKE_HIP_COMPILER)
    set_source_files_properties(upsweep_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(ryoanji_kernels upsweep_gpu.cu)
    target_include_directories(ryoanji_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src ${CSTONE_DIR})
endif()
