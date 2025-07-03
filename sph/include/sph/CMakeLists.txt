add_subdirectory(hydro_std)
add_subdirectory(hydro_turb)
add_subdirectory(hydro_ve)

add_library(helmEOS STATIC helmholtz_eos.cpp)
target_include_directories(helmEOS PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)

set(TIMESTEP_GPU_KERNELS
        positions_gpu.cu ts_groups.cu update_h_gpu.cu
)

if (CMAKE_HIP_COMPILER)
    set_source_files_properties(${TIMESTEP_GPU_KERNELS} PROPERTIES LANGUAGE HIP)
endif ()

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(timesteps_gpu OBJECT ${TIMESTEP_GPU_KERNELS})
    target_include_directories(timesteps_gpu PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(timesteps_gpu PRIVATE USE_CUDA)

    add_library(sph_gpu
            $<TARGET_OBJECTS:timesteps_gpu>
            $<TARGET_OBJECTS:hydro_std_gpu>
            $<TARGET_OBJECTS:hydro_turb_gpu>
            $<TARGET_OBJECTS:hydro_ve_gpu>)
    set_target_properties(sph_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

set(DATA_URL "https://raw.githubusercontent.com/rmcabezon/net90/main/src/helm_table.dat")
set(DATA_FILE "${CMAKE_CURRENT_LIST_DIR}/helm_table.dat")

# Download helm_table.dat only if it does not exist
add_custom_command(
    OUTPUT ${DATA_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo "Downloading helm_table.dat from ${DATA_URL} to ${DATA_FILE}"
    COMMAND ${CMAKE_COMMAND} -E remove -f ${DATA_FILE}
    COMMAND ${CMAKE_COMMAND} -E download ${DATA_URL} ${DATA_FILE}
    COMMENT "Downloading helm_table.dat if it does not exist"
)

add_custom_target(helm_table_download ALL DEPENDS ${DATA_FILE})
