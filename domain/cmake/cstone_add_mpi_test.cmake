include(cstone_add_test)

function(cstone_add_mpi_test target testname ranks labels)
    cmake_parse_arguments(ARG "GPU" "" "SOURCES;LINK;INCLUDE" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "cstone_add_mpi_test: SOURCES required")
    endif()

    list(APPEND ARG_SOURCES test_main.cpp)

    add_executable(${target} ${ARG_SOURCES})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_include_directories(${target} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(${target} PRIVATE -Wno-unknown-pragmas)
    target_link_libraries(${target} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main)

    if(ARG_INCLUDE)
        target_include_directories(${target} PRIVATE ${ARG_INCLUDE})
    endif()

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()

    cstone_add_test(${testname} EXECUTABLE ${target} RANKS ${ranks})
    set_tests_properties(${testname} PROPERTIES LABELS "${labels}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)

    if(ARG_GPU)
        target_link_libraries(${target} PRIVATE cstone_gpu)
        if(CSTONE_WITH_CUDA)
            target_link_libraries(${target} PRIVATE CUDA::cudart)
        endif()
        if(CSTONE_WITH_HIP)
            target_link_libraries(${target} PRIVATE hip::host)
            target_compile_definitions(${target} PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
            set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
        endif()
    endif()
endfunction()
