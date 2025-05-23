if(CMAKE_HIP_COMPILER)
    set_source_files_properties(device_vector.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gpu_utils_obj OBJECT device_vector.cu)
    target_include_directories(gpu_utils_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()