list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(setup_GTest)
include(ryoanji_add_test)

set(RYOANJI_TEST_INCLUDE_DIRS ${CSTONE_DIR} ${CSTONE_TEST_DIR} ${PROJECT_SOURCE_DIR}/src)

if (RYOANJI_WITH_HIP)
    set_source_files_properties(demo.cu
                                nbody/direct.cu
                                nbody/warpscan.cu
                                PROPERTIES LANGUAGE HIP)
endif()

if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    ryoanji_add_executable_test(ryoanji_demo
        SOURCES demo.cu
        LINK ryoanji_kernels;cstone_gpu
        LABELS "01r;gpu"
    )
    set_target_properties(ryoanji_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if (RYOANJI_WITH_HIP)
        target_link_libraries(ryoanji_demo PUBLIC roc::rocthrust hip::hipcub)
    endif()
endif()

if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    ryoanji_add_executable_test(ryoanji_demo_mpi
        SOURCES demo_mpi.cpp
        LINK cstone_gpu;ryoanji;${MPI_CXX_LIBRARIES}
        INCLUDE ${MPI_CXX_INCLUDE_PATH}
        LABELS "02r;gpu"
        MPI
    )

    if (RYOANJI_WITH_HIP)
        target_link_libraries(ryoanji_demo_mpi PRIVATE hip::host)
        set_target_properties(ryoanji_demo_mpi PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    if (RYOANJI_WITH_CUDA)
        target_link_libraries(ryoanji_demo_mpi PRIVATE CUDA::cudart)
    endif()

    target_compile_options(ryoanji_demo_mpi PRIVATE -Wno-unknown-pragmas)
endif()

ryoanji_add_executable_test(cpu_unit_tests
    SOURCES
        nbody/cartesian_qpole.cpp
        nbody/kernel.cpp
        nbody/traversal_cpu.cpp
        nbody/ewald_cpu.cpp
        test_main.cpp
    LINK GTest::gtest_main;OpenMP::OpenMP_CXX
    COMPILE_OPTIONS -Wall;-Wextra;-Wno-unknown-pragmas
    LABELS "01r;cpu"
)

if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    ryoanji_add_executable_test(gpu_unit_tests
        SOURCES
            nbody/ewald.cu
            nbody/direct.cu
            test_main.cpp
        LINK cstone_thrust_incl;OpenMP::OpenMP_CXX;ryoanji;GTest::gtest_main
        LABELS "01r;gpu"
    )

    if (RYOANJI_WITH_HIP)
        set_target_properties(gpu_unit_tests PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    if (RYOANJI_WITH_CUDA)
        target_compile_options(gpu_unit_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()
endif()

add_subdirectory(interface)
