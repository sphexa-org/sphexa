
set(UNIT_TESTS chemistry_data.cpp cooling.cpp)
set(testname cooling_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)
target_compile_definitions(${testname} PUBLIC SPH_EXA_HAVE_H5HUT)
target_compile_definitions(${testname} PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(${testname} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(${testname} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include ${MAIN_APP_DIR})

target_link_libraries(${testname} PRIVATE grackle)
target_link_libraries(${testname} PRIVATE cooler)
target_link_libraries(${testname} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${testname} PRIVATE GTest::gtest_main)

add_test(NAME ${testname} COMMAND ${testname})
set_tests_properties(${testname} PROPERTIES LABELS "01r;cpu")
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/physics)
