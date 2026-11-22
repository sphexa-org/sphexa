ryoanji_add_executable_test(global_upsweep_cpu
    SOURCES global_upsweep_cpu.cpp
    LINK ${MPI_CXX_LIBRARIES}
    INCLUDE ${MPI_CXX_INCLUDE_PATH};${RYOANJI_TEST_INCLUDE_DIRS}
    COMPILE_OPTIONS -Wno-unknown-pragmas
    LABELS "10r;cpu"
    MPI RANKS 10
)

if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    ryoanji_add_executable_test(global_upsweep_gpu
        SOURCES global_upsweep_gpu.cpp
        LINK ${MPI_CXX_LIBRARIES};cstone_gpu;ryoanji
        INCLUDE ${MPI_CXX_INCLUDE_PATH};${RYOANJI_TEST_INCLUDE_DIRS}
        COMPILE_OPTIONS -Wno-unknown-pragmas
        LABELS "10r;gpu"
        MPI RANKS 10
    )

    if (RYOANJI_WITH_HIP)
        target_link_libraries(global_upsweep_gpu PRIVATE hip::host)
        set_target_properties(global_upsweep_gpu PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    if (RYOANJI_WITH_CUDA)
        target_link_libraries(global_upsweep_gpu PRIVATE CUDA::cudart)
    endif()
endif()

if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
    ryoanji_add_executable_test(global_forces_gpu
        SOURCES global_forces_gpu.cpp
        LINK ${MPI_CXX_LIBRARIES};cstone_gpu;ryoanji
        INCLUDE ${MPI_CXX_INCLUDE_PATH};${RYOANJI_TEST_INCLUDE_DIRS}
        COMPILE_OPTIONS -Wno-unknown-pragmas
        LABELS "4r;gpu"
        MPI RANKS 4
    )

    if (RYOANJI_WITH_HIP)
        target_link_libraries(global_forces_gpu PRIVATE hip::host)
        set_target_properties(global_forces_gpu PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    if (RYOANJI_WITH_CUDA)
        target_link_libraries(global_forces_gpu PRIVATE CUDA::cudart)
    endif()
endif()
