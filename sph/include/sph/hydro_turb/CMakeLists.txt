
if(SPH_EXA_WITH_HIP)
    set_source_files_properties(stirring_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(SPH_EXA_WITH_CUDA OR SPH_EXA_WITH_HIP)
    add_library(hydro_turb_gpu OBJECT stirring_gpu.cu)
    target_include_directories(hydro_turb_gpu PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
endif()
