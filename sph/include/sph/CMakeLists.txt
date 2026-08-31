list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(sph_add_gpu_object)

add_subdirectory(hydro_std)
add_subdirectory(hydro_turb)
add_subdirectory(hydro_ve)

sph_add_gpu_object(timesteps_gpu
    SOURCES positions_gpu.cu ts_groups.cu find_neighbors_gpu.cu neighborhood_gpu.cu)

if(SPH_EXA_WITH_CUDA OR SPH_EXA_WITH_HIP)
    add_library(sph_gpu
            $<TARGET_OBJECTS:timesteps_gpu>
            $<TARGET_OBJECTS:hydro_std_gpu>
            $<TARGET_OBJECTS:hydro_turb_gpu>
            $<TARGET_OBJECTS:hydro_ve_gpu>)
    set_target_properties(sph_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
