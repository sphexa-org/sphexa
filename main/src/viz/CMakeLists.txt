set(INSITU
    "None"
    CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

# Note: https://github.com/Alpine-DAV/ascent/pull/1677
option(
  INSITU_FORCE_EXPLICIT_CONNECTIVITY_LIST
  "Force usage of an explicit connectivity list on mesh. (Ascent bug workaround)"
  OFF)

if(INSITU STREQUAL "Catalyst")
  find_package(catalyst 2.0 REQUIRED)
elseif(INSITU STREQUAL "Ascent")
  find_package(Ascent 0.9.5 REQUIRED)
endif()

add_library(viz INTERFACE)
target_include_directories(
  viz INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_definitions(
  viz
  INTERFACE
    $<$<NOT:$<STREQUAL:${INSITU},None>>:SPHEXA_WITH_VISUALIZATION>
    $<$<BOOL:${INSITU_FORCE_EXPLICIT_CONNECTIVITY_LIST}>:FORCE_EXPLICIT_CONNECTIVITY_LIST>
)

if(INSITU STREQUAL "Catalyst")
  target_compile_definitions(viz INTERFACE SPH_EXA_USE_CATALYST2)
  target_link_libraries(viz INTERFACE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
  target_compile_definitions(viz INTERFACE SPH_EXA_USE_ASCENT)
  target_link_libraries(viz INTERFACE ascent::ascent_mpi)
endif()
