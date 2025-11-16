if(CMAKE_HIP_COMPILER)
    set_source_files_properties(csarray_gpu.cu octree_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(octree_obj OBJECT csarray_gpu.cu octree_gpu.cu)
    target_include_directories(octree_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(octree_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()
