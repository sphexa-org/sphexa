if(CMAKE_HIP_COMPILER)
    set_source_files_properties(conserved_gpu.cu gpu_reductions.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(obs_gpu OBJECT conserved_gpu.cu gpu_reductions.cu)
    target_include_directories(obs_gpu PRIVATE ${CSTONE_DIR} ${SPH_DIR})
endif()

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(obs_gpu PUBLIC CUDA::cudart)
endif()

if(CMAKE_HIP_COMPILER)
    target_link_libraries(obs_gpu PUBLIC hip::host)
endif()

add_library(observables observables.cpp)
target_include_directories(observables PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR} ${SPH_DIR}
        ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(observables PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(observables_gpu $<TARGET_OBJECTS:obs_gpu> observables.cpp)
    target_compile_definitions(observables_gpu PRIVATE USE_CUDA)
    target_include_directories(observables_gpu PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR}
            ${SPH_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(observables_gpu PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(observables_gpu PUBLIC CUDA::cudart)
endif()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(observables_gpu PUBLIC hip::host)
    target_compile_definitions(observables_gpu PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(observables_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()