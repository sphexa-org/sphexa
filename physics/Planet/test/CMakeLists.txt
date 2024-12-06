set(UNIT_TESTS planet_test.cpp accretion.cpp test_main.cpp)
set(testname PlanetTests)
set(exename planet_tests)
set(MPI_TEST_DIR ${PROJECT_SOURCE_DIR}/domain/test/integration_mpi)

add_executable(${exename} ${UNIT_TESTS})
target_compile_options(${exename} PRIVATE -Wall -Wextra)
target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_include_directories(${exename} PRIVATE ${PLANET_DIR} ${CSTONE_DIR} ${SPH_DIR} ${PROJECT_SOURCE_DIR}/include ${MAIN_APP_DIR} ${MPI_TEST_DIR})

target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main)

set(cmd ${exename})
list(PREPEND cmd "${MPIEXEC_EXECUTABLE}" " " "${MPIEXEC_NUMPROC_FLAG}" " 2 ")
add_test(NAME ${testname} COMMAND ${cmd})
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/physics)
