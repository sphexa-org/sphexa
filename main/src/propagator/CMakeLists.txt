add_library(propagator_init nbody.cpp std_hydro_grackle.cpp std_hydro.cpp ve_hydro.cpp ve_hydro_bdt.cpp)
target_include_directories(propagator_init PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR} 
        ${SPH_DIR} ${RYOANJI_DIR} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(propagator_init PRIVATE ${MPI_CXX_LIBRARIES} util OpenMP::OpenMP_CXX)
enableGrackle(propagator_init)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(propagator_init_gpu nbody.cpp std_hydro_grackle.cpp std_hydro.cpp ve_hydro.cpp ve_hydro_bdt.cpp)
    target_compile_definitions(propagator_init_gpu PRIVATE USE_CUDA)
    target_include_directories(propagator_init_gpu PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR}
            ${SPH_DIR} ${RYOANJI_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(propagator_init_gpu PRIVATE ${MPI_CXX_LIBRARIES} ryoanji sph_gpu util OpenMP::OpenMP_CXX)
    enableGrackle(propagator_init_gpu)
endif ()

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(propagator_init_gpu PRIVATE ryoanji sph_gpu util CUDA::cudart)
endif ()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(propagator_init_gpu PRIVATE ryoanji sph_gpu util hip::host)
    target_compile_definitions(propagator_init_gpu PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(propagator_init_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()
