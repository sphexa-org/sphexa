# temporary fix for hip build
if(NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR "bin")
endif(NOT DEFINED CMAKE_INSTALL_BINDIR)

install(PROGRAMS plot_power.py set_parms.py DESTINATION ${CMAKE_INSTALL_BINDIR})
