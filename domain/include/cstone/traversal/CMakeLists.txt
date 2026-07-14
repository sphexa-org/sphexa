if(CSTONE_WITH_HIP)
    set_source_files_properties(collisions_gpu.cu groups_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    add_library(traversal_obj OBJECT collisions_gpu.cu groups_gpu.cu)
    target_include_directories(traversal_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(traversal_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()
