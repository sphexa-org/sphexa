set(MPI_TEST_DIR ${CMAKE_SOURCE_DIR}/domain/test/integration_mpi)

disk_add_test(disk_tests
    SOURCES disk_test.cpp accretion.cpp test_main.cpp
    LABELS "02r;cpu"
    MPI)
