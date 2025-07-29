
set(CSTONE_TEST_DIR ${CSTONE_DIR}/../test)
message(STATUS "cstone test dir " ${CSTONE_TEST_DIR})

function(addFrontendMpiTest source exename testname ranks)
    list(APPEND source ${CSTONE_TEST_DIR}/integration_mpi/test_main.cpp)
    addMpiTest("${source}" ${exename} ${testname} ${ranks})
    target_include_directories(${exename} PRIVATE ${CSTONE_DIR} ${SPH_DIR} ${PROJECT_SOURCE_DIR}/main/src)
endfunction()

if (SPH_EXA_WITH_H5PART)
    addFrontendMpiTest(hdf5.cpp hdf5io HDF5IO 2)
    target_link_libraries(hdf5io PRIVATE io)
endif()


