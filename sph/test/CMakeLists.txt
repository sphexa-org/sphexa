list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(sph_add_test)

add_subdirectory(hydro_turb)

sph_add_test(sph_tests
    SOURCES positions.cpp std.cpp table_creation.cpp ve.cpp)

install(FILES ${PROJECT_SOURCE_DIR}/test/example_data.txt DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)

add_custom_command(TARGET sph_tests PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/test/example_data.txt $<TARGET_FILE_DIR:sph_tests>/example_data.txt)

if(SPH_EXA_WITH_CUDA OR SPH_EXA_WITH_HIP)
    sph_add_test(sph_gpu_tests
        SOURCES timestep.cu
        LINK sph_gpu
        LABELS "01r;gpu")

    if(SPH_EXA_WITH_HIP)
        set_target_properties(sph_gpu_tests PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(sph_gpu_tests PRIVATE cstone_thrust_incl)
    endif()
endif()
