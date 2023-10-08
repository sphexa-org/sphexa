
set(VE_KERNELS
        xmass_gpu.cu
        eos_gpu.cu
        ve_def_gradh_gpu.cu
        iad_divv_curlv_gpu.cu
        av_switches_gpu.cu
        momentum_energy_gpu.cu
        additional_fields.cu)

if(CMAKE_HIP_COMPILER)
    set_source_files_properties(${VE_KERNELS} PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(hydro_ve_gpu OBJECT ${VE_KERNELS})
    target_include_directories(hydro_ve_gpu PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(hydro_ve_gpu PRIVATE USE_CUDA)
endif()

