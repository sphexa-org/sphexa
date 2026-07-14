if(RYOANJI_WITH_HIP)
    set_source_files_properties(traversal_gpu.cu upsweep_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    add_library(ryoanji_kernels OBJECT traversal_gpu.cu upsweep_gpu.cu)
    target_include_directories(ryoanji_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src ${CSTONE_DIR})
endif()
