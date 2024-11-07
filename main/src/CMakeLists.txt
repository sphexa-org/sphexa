
set(SPH_EXA_INCLUDE_DIRS ${CSTONE_DIR} ${COOLING_DIR} ${RYOANJI_DIR} ${SPH_DIR}
    ${PROJECT_SOURCE_DIR}/main/src
    ${CMAKE_BINARY_DIR}/main/src
    ${MPI_CXX_INCLUDE_PATH})

function(enableGrackle exename)
    if(SPH_EXA_WITH_GRACKLE)
        target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_GRACKLE)
        target_compile_definitions(${exename} PUBLIC CONFIG_BFLOAT_8)
        target_link_libraries(${exename} PRIVATE grackle)
        target_link_libraries(${exename} PRIVATE cooler)
    endif()
endfunction()

add_subdirectory(init)
add_subdirectory(io)
add_subdirectory(observables)
add_subdirectory(propagator)
add_subdirectory(sphexa)
#add_subdirectory(util)

if (BUILD_ANALYTICAL)
    add_subdirectory(analytical_solutions)
endif ()

include(setup_gitinfo)
configure_file(
  version.h.in
  ${CMAKE_BINARY_DIR}/main/src/version.h
)
