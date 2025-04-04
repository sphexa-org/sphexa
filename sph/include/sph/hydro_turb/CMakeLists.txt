
if(CMAKE_HIP_COMPILER)
    set_source_files_properties(stirring_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(hydro_turb_gpu OBJECT stirring_gpu.cu)
    target_include_directories(hydro_turb_gpu PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include ${SPH_EXA_DIR})
endif()
