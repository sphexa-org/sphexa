include(ryoanji_add_test)

function(addRyoanjiMpiTest source exename testname ranks)
    add_executable(${exename} ${source})
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${RYOANJI_TEST_INCLUDE_DIRS})
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES})
    target_compile_options(${exename} PRIVATE -Wno-unknown-pragmas)
    ryoanji_add_test(${exename} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/)
endfunction()

addRyoanjiMpiTest(global_upsweep_cpu.cpp global_upsweep_cpu GlobalUpsweepCpu 10)

function(addRyoanjiGpuMpiTest source exename testname ranks)
    if (RYOANJI_WITH_CUDA OR RYOANJI_WITH_HIP)
        addRyoanjiMpiTest(${source} ${exename} ${testname} ${ranks})
        target_link_libraries(${exename} PRIVATE cstone_gpu ryoanji)
    endif()

    if (RYOANJI_WITH_HIP)
        target_link_libraries(${exename} PRIVATE hip::host)
        set_target_properties(${exename} PROPERTIES LINKER_LANGUAGE CXX)
    endif ()

    if (RYOANJI_WITH_CUDA)
        target_link_libraries(${exename} PRIVATE CUDA::cudart)
    endif ()
endfunction()

addRyoanjiGpuMpiTest(global_upsweep_gpu.cpp global_upsweep_gpu GlobalUpsweepGpu 10)
addRyoanjiGpuMpiTest(global_forces_gpu.cpp global_forces_gpu GlobalForcesGpu 6)
