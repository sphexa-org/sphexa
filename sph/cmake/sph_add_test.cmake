function(sph_add_test target)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK;LABELS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "sph_add_test: SOURCES required")
    endif()

    if(NOT ARG_LABELS)
        set(ARG_LABELS "01r;cpu")
    endif()

    add_executable(${target} ${ARG_SOURCES})

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_include_directories(${target} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()

    target_link_libraries(${target} PRIVATE GTest::gtest_main)

    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${ARG_LABELS}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)
endfunction()
