
set(testname coordinate_test)
add_executable(${testname} random.cpp test_main.cpp)
target_include_directories(${testname} PRIVATE ../../include)
target_link_libraries(${testname} GTest::gtest_main)
target_compile_options(${testname} PRIVATE -Wno-unknown-pragmas)
add_test(NAME ${testname} COMMAND ${testname})
set_tests_properties(${testname} PROPERTIES LABELS "01r;cpu")
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/coord_samples)
