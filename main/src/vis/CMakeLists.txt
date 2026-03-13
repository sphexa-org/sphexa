set(INSITU
    None
    CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

if(INSITU STREQUAL "Catalyst")
  find_package(catalyst REQUIRED PATHS
               "$ENV{EBROOTPARAVIEW}/lib/cmake/paraview-5.10")
elseif(INSITU STREQUAL "Ascent")
  find_package(Ascent REQUIRED PATHS "$ENV{EBROOTASCENT}/lib/cmake/ascent")
endif()

add_library(vis INTERFACE)

target_include_directories(vis INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(INSITU STREQUAL "Catalyst")
  target_compile_definitions(vis INTERFACE SPH_EXA_USE_CATALYST2)
  target_link_libraries(vis INTERFACE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
  target_compile_definitions(vis INTERFACE SPH_EXA_USE_ASCENT)
  target_link_libraries(vis INTERFACE ascent::ascent_mpi)
endif()
