set(exename iobench)
add_executable(${exename} iobench.cpp)
target_include_directories(${exename} PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_link_libraries(${exename} PRIVATE io propagator OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})

install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
