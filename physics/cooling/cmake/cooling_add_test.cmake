function(cooling_add_test target)
    cmake_parse_arguments(ARG "MPI" "RANKS" "SOURCES;LINK;INCLUDE;LABELS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "cooling_add_test: SOURCES required")
    endif()

    if(NOT ARG_LABELS)
        set(ARG_LABELS "1r;cpu")
    endif()

    add_executable(${target} ${ARG_SOURCES})

    target_compile_options(${target} PRIVATE -Wall -Wextra)
    target_compile_definitions(${target} PUBLIC SPH_EXA_HAVE_H5HUT)
    target_compile_definitions(${target} PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_compile_definitions(${target} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    target_include_directories(${target} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include ${MAIN_APP_DIR})

    if(ARG_INCLUDE)
        target_include_directories(${target} PRIVATE ${ARG_INCLUDE})
    endif()

    target_link_libraries(${target} PRIVATE grackle cooler OpenMP::OpenMP_CXX GTest::gtest_main)

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()

    if(ARG_MPI)
        if(NOT ARG_RANKS)
            set(ARG_RANKS 2)
        endif()
        add_test(NAME ${target}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS}
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>)
    else()
        add_test(NAME ${target} COMMAND ${target})
    endif()

    set_tests_properties(${target} PROPERTIES LABELS "${ARG_LABELS}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/physics)
endfunction()
