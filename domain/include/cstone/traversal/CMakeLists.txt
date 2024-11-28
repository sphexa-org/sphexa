if(CMAKE_HIP_COMPILER)
    set_source_files_properties(collisions_gpu.cu groups_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(traversal_obj OBJECT collisions_gpu.cu groups_gpu.cu)
    target_include_directories(traversal_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()
