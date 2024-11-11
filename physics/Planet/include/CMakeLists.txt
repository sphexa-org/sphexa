
if(CMAKE_HIP_COMPILER)
    set_source_files_properties(computeCentralForce_gpu.cu accretion_gpu.cu betaCooling_gpu.cu eos_polytropic_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(planet_gpu OBJECT computeCentralForce_gpu.cu accretion_gpu.cu betaCooling_gpu.cu eos_polytropic_gpu.cu)
    target_include_directories(planet_gpu PRIVATE ${CSTONE_DIR} ${SPH_DIR})
    target_compile_definitions(planet_gpu PRIVATE USE_CUDA)
    target_link_libraries(planet_gpu PUBLIC CUDA::cudart)
endif()
