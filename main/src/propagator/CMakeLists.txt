set(PROP_SOURCES nbody.cpp std_hydro_grackle.cpp std_hydro.cpp ve_hydro.cpp ve_hydro_bdt.cpp)

add_library(propagator ${PROP_SOURCES})
target_include_directories(propagator PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR}
        ${SPH_DIR} ${RYOANJI_DIR} ${MPI_CXX_INCLUDE_PATH} ${DISK_DIR})

target_link_libraries(propagator PRIVATE ${MPI_CXX_LIBRARIES} util OpenMP::OpenMP_CXX)
enableDiskPhysics(propagator FALSE)
enableGrackle(propagator)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(propagator_gpu ${PROP_SOURCES})
    target_compile_definitions(propagator_gpu PRIVATE USE_CUDA)
    target_include_directories(propagator_gpu PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR}
            ${SPH_DIR} ${RYOANJI_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(propagator_gpu PRIVATE ${MPI_CXX_LIBRARIES} cstone_gpu ryoanji sph_gpu util OpenMP::OpenMP_CXX)
    enableDiskPhysics(propagator_gpu TRUE)
    enableGrackle(propagator_gpu)
endif ()

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(propagator_gpu PRIVATE CUDA::cudart)
endif ()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(propagator_gpu PRIVATE hip::host)
    target_compile_definitions(propagator_gpu PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(propagator_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()
