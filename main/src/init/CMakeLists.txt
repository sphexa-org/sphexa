add_library(sim_init sim_init.cpp)
target_include_directories(sim_init PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR} ${SPH_DIR}
        ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(sim_init PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
enableGrackle(sim_init)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(sim_init_gpu sim_init.cpp)
    target_compile_definitions(sim_init_gpu PRIVATE USE_CUDA)
    target_include_directories(sim_init_gpu PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR}
            ${SPH_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(sim_init_gpu PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
    enableGrackle(sim_init_gpu)
endif ()

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(sim_init_gpu PRIVATE CUDA::cudart)
endif ()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(sim_init_gpu PRIVATE hip::host)
    target_compile_definitions(sim_init_gpu PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(sim_init_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()
