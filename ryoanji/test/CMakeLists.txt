list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(setup_GTest)
include(ryoanji_add_test)

set(RYOANJI_TEST_INCLUDE_DIRS ${CSTONE_DIR} ${CSTONE_TEST_DIR} ${PROJECT_SOURCE_DIR}/src)

if (RYOANJI_WITH_HIP)
    set_source_files_properties(demo.cu PROPERTIES LANGUAGE HIP)
endif()

if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    set(testname ryoanji_demo)
    add_executable(${testname} demo.cu)
    target_link_libraries(${testname} PRIVATE ryoanji_kernels cstone_gpu)
    set_target_properties(${testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${testname} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})
    # set_source_files_properties(demo.cu PROPERTIES COMPILE_FLAGS "-Xptxas -v")
    if (RYOANJI_WITH_HIP)
        target_link_libraries(${testname} PUBLIC roc::rocthrust hip::hipcub)
    endif()
    add_test(NAME ${testname} COMMAND ${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "01r;gpu")
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji)
endif ()


# multi-rank demonstrator
if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    set(testname ryoanji_demo_mpi)
    add_executable(${testname} demo_mpi.cpp)
    target_include_directories(${testname} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(${testname} PRIVATE cstone_gpu ryoanji ${MPI_CXX_LIBRARIES})

    if (RYOANJI_WITH_HIP)
        target_link_libraries(${testname} PRIVATE hip::host)
        set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
    endif ()

    if (RYOANJI_WITH_CUDA)
        target_link_libraries(${testname} PRIVATE CUDA::cudart)
    endif ()
    target_compile_options(${testname} PRIVATE -Wno-unknown-pragmas)
    add_test(NAME ${testname} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_PREFLAGS} 2 $<TARGET_FILE:${testname}>)
    set_tests_properties(${testname} PROPERTIES LABELS "02r;gpu")
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji)
    unset(testname)
endif ()

# CPU unit test suite
set(testname cpu_unit_tests)
add_executable(${testname}
        nbody/cartesian_qpole.cpp
        nbody/kernel.cpp
        nbody/traversal_cpu.cpp
        nbody/ewald_cpu.cpp
        test_main.cpp)

target_compile_options(${testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
target_include_directories(${testname} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(${testname} PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
add_test(NAME ${testname} COMMAND ${testname})
set_tests_properties(${testname} PROPERTIES LABELS "01r;cpu")
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji)

if (RYOANJI_WITH_HIP)
    set_source_files_properties(nbody/direct.cu nbody/warpscan.cu PROPERTIES LANGUAGE HIP)
endif()

# GPU unit test suite
if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    set(testname gpu_unit_tests)
    add_executable(${testname}
            nbody/ewald.cu
            nbody/direct.cu
            test_main.cpp)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${testname} PRIVATE cstone_thrust_incl OpenMP::OpenMP_CXX ryoanji GTest::gtest_main)
    if (RYOANJI_WITH_HIP)
        set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
    endif ()
    if (RYOANJI_WITH_CUDA)
        target_compile_options(${testname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif ()

    add_test(NAME ${testname} COMMAND ${testname})
    set_tests_properties(${testname} PROPERTIES LABELS "01r;gpu")
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji)
endif ()

add_subdirectory(interface)

