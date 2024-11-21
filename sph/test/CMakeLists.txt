add_subdirectory(hydro_turb)

set(UNIT_TESTS
    positions.cpp
    std.cpp
    table_creation.cpp
    ve.cpp
    test_eos.cpp
)

set(testname sph_tests)
add_executable(${testname} ${UNIT_TESTS})

target_compile_options(${testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
target_include_directories(${testname} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${testname} PRIVATE GTest::gtest_main)
add_test(NAME ${testname} COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)
install(FILES ${PROJECT_SOURCE_DIR}/test/example_data.txt DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)

# create a symbolic link in the build directory to the example data file
add_custom_command(TARGET ${testname} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${PROJECT_SOURCE_DIR}/test/example_data.txt $<TARGET_FILE_DIR:${testname}>/example_data.txt)

set(UNIT_TESTS_GPU timestep.cu)

if(CMAKE_HIP_COMPILER)
    set_source_files_properties(${UNIT_TESTS_GPU} PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(testname sph_gpu_tests)
    add_executable(${testname} ${UNIT_TESTS_GPU})

    target_include_directories(${testname} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${testname} PRIVATE sph_gpu GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})

    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)
    install(FILES ${PROJECT_SOURCE_DIR}/test/example_data.txt DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)
endif()
