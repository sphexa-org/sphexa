
set(UNIT_TESTS
        init/grid.cpp
        io/arg_parser.cpp
        io/h5part_wrapper.cpp
        io/id_tag_utils.cpp
        observables/gravitational_waves.cpp
        sphexa/particles_data.cpp
        test_main.cpp)

if (SPH_EXA_WITH_H5HUT)
    set(testname frontend_units)
    add_executable(${testname} ${UNIT_TESTS})
    target_compile_options(${testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

    target_include_directories(${testname} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${SPH_DIR} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${testname} PRIVATE io ${MPI_CXX_LIBRARIES} GTest::gtest_main)
    enableH5hut(${testname})
    add_test(NAME ${testname} COMMAND ${testname})
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/main)
endif ()

if (CMAKE_CUDA_COMPILER AND SPH_EXA_WITH_H5HUT AND SPH_EXA_WITH_CUDA)
    set(testname frontend_units_cuda)
    add_executable(${testname}
            io/id_tag_utils_gpu.cu
            test_main.cpp)
    target_include_directories(${testname} PRIVATE
                               ${COOLING_DIR}
                               ${SPH_DIR}
                               ${CSTONE_DIR}
                               ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${testname} PUBLIC cstone_gpu io io_gpu sph_gpu OpenMP::OpenMP_CXX CUDA::cudart GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/main)
    unset(testname)
endif ()
