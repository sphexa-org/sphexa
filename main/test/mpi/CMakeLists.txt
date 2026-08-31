set(CSTONE_TEST_DIR ${CSTONE_DIR}/../test)
message(STATUS "cstone test dir " ${CSTONE_TEST_DIR})

if (SPH_EXA_WITH_H5HUT)
    main_add_mpi_test(hdf5io HDF5IO 2 "02r;cpu"
        SOURCES hdf5.cpp
        LINK io)

    if(DEFINED ENV{SPH_CTEST_WORKING_DIR})
        set_tests_properties(HDF5IO PROPERTIES WORKING_DIRECTORY "$ENV{SPH_CTEST_WORKING_DIR}")
    endif()
endif()
