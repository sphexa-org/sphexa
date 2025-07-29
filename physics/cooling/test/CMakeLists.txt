
set(UNIT_TESTS chemistry_data.cpp cooling.cpp)
set(testname cooling_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)
target_compile_definitions(${testname} PUBLIC SPH_EXA_HAVE_H5PART)


target_include_directories(${testname} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include ${MAIN_APP_DIR})

target_link_libraries(${testname} PRIVATE grackle)
target_link_libraries(${testname} PRIVATE cooler)
target_link_libraries(${testname} PRIVATE GTest::gtest_main)
target_compile_definitions(${testname} PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(${testname} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME ${testname} COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/physics)
