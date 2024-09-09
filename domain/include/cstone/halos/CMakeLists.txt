if(CMAKE_HIP_COMPILER)
    set_source_files_properties(gather_halos_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gather_halos_obj OBJECT gather_halos_gpu.cu)
    target_include_directories(gather_halos_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()

add_library(halos_utility OBJECT halos_utility.cpp)
target_include_directories(halos_utility PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(halos_utility PRIVATE OpenMP::OpenMP_CXX)
