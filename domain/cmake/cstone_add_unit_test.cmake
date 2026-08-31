function(cstone_add_unit_test target)
    cmake_parse_arguments(ARG "OPENMP" "DESTINATION" "SOURCES;LINK;INCLUDE;LABELS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "cstone_add_unit_test: SOURCES required")
    endif()

    if(NOT ARG_LABELS)
        set(ARG_LABELS "01r;cpu")
    endif()

    if(NOT ARG_DESTINATION)
        set(ARG_DESTINATION unit)
    endif()

    add_executable(${target} ${ARG_SOURCES})

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)

    if(ARG_INCLUDE)
        target_include_directories(${target} PRIVATE ${ARG_INCLUDE})
    endif()

    target_link_libraries(${target} PRIVATE GTest::gtest_main)

    if(ARG_OPENMP)
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()

    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${ARG_LABELS}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/${ARG_DESTINATION})
endfunction()
