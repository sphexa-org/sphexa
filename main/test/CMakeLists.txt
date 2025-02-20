
set(UNIT_TESTS
        init/grid.cpp
        io/arg_parser.cpp
        io/h5part_wrapper.cpp
        observables/gravitational_waves.cpp
        sphexa/particles_data.cpp
        test_main.cpp)

if (SPH_EXA_WITH_H5HUT)
    set(exename frontend_units)
    add_executable(${exename} ${UNIT_TESTS})
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${SPH_DIR} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${exename} PRIVATE io ${MPI_CXX_LIBRARIES} GTest::gtest_main)
    enableH5hut(${exename})
    add_test(NAME FrontendUnits COMMAND ${exename})
endif ()

if (CMAKE_CUDA_COMPILER AND SPH_EXA_WITH_H5HUT)
    set(testname frontend_units_cuda)
    add_executable(${testname}
            cuda/gpu_particles_data.cpp
            test_main.cpp)
    target_include_directories(${testname} PRIVATE
                               ${COOLING_DIR}
                               ${SPH_DIR}
                               ${CSTONE_DIR}
                               ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${testname} PUBLIC cstone_gpu sph_gpu OpenMP::OpenMP_CXX CUDA::cudart GTest::gtest_main)
    add_test(NAME FrontendUnitsCuda COMMAND ${exename})
    unset(testname)
endif ()
