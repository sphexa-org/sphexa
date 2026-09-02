if(CSTONE_WITH_HIP)
    set_source_files_properties(device_vector.cu PROPERTIES LANGUAGE HIP)
endif()

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    add_library(gpu_utils_obj OBJECT device_vector.cu)
    target_include_directories(gpu_utils_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(gpu_utils_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()
