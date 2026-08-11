
set(CSTONE_TEST_DIR ${CSTONE_DIR}/../test)
message(STATUS "cstone test dir " ${CSTONE_TEST_DIR})

function(addFrontendMpiTest source exename testname ranks)
    list(APPEND source ${CSTONE_TEST_DIR}/integration_mpi/test_main.cpp)
    addMpiTest("${source}" ${exename} ${testname} ${ranks})

    if (${ranks} LESS 10)
        set_tests_properties(${testname} PROPERTIES LABELS "0${ranks}r;cpu")
    else()
        set_tests_properties(${testname} PROPERTIES LABELS "${ranks}r;cpu")
    endif()

    target_include_directories(${exename} PRIVATE ${CSTONE_DIR} ${SPH_DIR} ${PROJECT_SOURCE_DIR}/main/src)
endfunction()

set(exename hdf5io)
set(testname HDF5IO)
if (SPH_EXA_WITH_H5HUT)
    addFrontendMpiTest(hdf5.cpp ${exename} ${testname} 2)

    if(DEFINED ENV{SPH_CTEST_WORKING_DIR})
        set_tests_properties(${testname} PROPERTIES WORKING_DIRECTORY "$ENV{SPH_CTEST_WORKING_DIR}")
    endif()

    target_link_libraries(${exename} PRIVATE io)
endif()

