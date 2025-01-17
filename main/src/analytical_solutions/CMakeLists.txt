# temporary fix for hip build
if(NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR "bin")
endif(NOT DEFINED CMAKE_INSTALL_BINDIR)

add_subdirectory(sedov_solution)
add_subdirectory(turbulence)

install(PROGRAMS compare_noh.py compare_solutions.py DESTINATION ${CMAKE_INSTALL_BINDIR})
