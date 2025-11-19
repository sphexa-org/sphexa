if(CMAKE_HIP_COMPILER)
    set_source_files_properties(primitives_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(primitives_gpu_obj OBJECT primitives_gpu.cu)
    target_include_directories(primitives_gpu_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(primitives_gpu_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()

