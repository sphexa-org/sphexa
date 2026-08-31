function(main_add_test target)
    cmake_parse_arguments(ARG "H5HUT" "" "SOURCES;LINK;INCLUDE;LABELS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "main_add_test: SOURCES required")
    endif()

    if(NOT ARG_LABELS)
        set(ARG_LABELS "01r;cpu")
    endif()

    add_executable(${target} ${ARG_SOURCES})

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

    target_include_directories(${target} PRIVATE
        ${MPI_CXX_INCLUDE_PATH}
        ${SPH_DIR}
        ${CSTONE_DIR}
        ${PROJECT_SOURCE_DIR}/main/src)

    if(ARG_INCLUDE)
        target_include_directories(${target} PRIVATE ${ARG_INCLUDE})
    endif()

    target_link_libraries(${target} PRIVATE io ${MPI_CXX_LIBRARIES} GTest::gtest_main)

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()

    if(ARG_H5HUT)
        enableH5hut(${target})
    endif()

    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${ARG_LABELS}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/main)
endfunction()

include("${CMAKE_SOURCE_DIR}/domain/cmake/cstone_add_test.cmake")

function(main_add_mpi_test target testname ranks labels)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK;INCLUDE" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "main_add_mpi_test: SOURCES required")
    endif()

    list(APPEND ARG_SOURCES ${CSTONE_TEST_DIR}/integration_mpi/test_main.cpp)

    add_executable(${target} ${ARG_SOURCES})
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/test
        ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(${target} PRIVATE -Wno-unknown-pragmas)
    target_link_libraries(${target} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main)

    cstone_add_test(${testname} EXECUTABLE ${target} RANKS ${ranks})

    set_tests_properties(${testname} PROPERTIES LABELS "${labels}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)

    target_include_directories(${target} PRIVATE ${CSTONE_DIR} ${SPH_DIR} ${PROJECT_SOURCE_DIR}/main/src)

    if(ARG_INCLUDE)
        target_include_directories(${target} PRIVATE ${ARG_INCLUDE})
    endif()

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()
endfunction()
