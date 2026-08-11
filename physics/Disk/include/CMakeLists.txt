
if (SPH_EXA_WITH_HIP)
    set_source_files_properties(central_force_gpu.cu accretion_gpu.cu beta_cooling_gpu.cu PROPERTIES LANGUAGE HIP)
endif ()

set(libname disk_gpu)
if (SPH_EXA_WITH_CUDA OR SPH_EXA_WITH_HIP)
    add_library(${libname} OBJECT central_force_gpu.cu accretion_gpu.cu beta_cooling_gpu.cu)
    target_include_directories(${libname} PRIVATE ${CSTONE_DIR} ${SPH_DIR})
    if (SPH_EXA_WITH_CUDA)
        target_link_libraries(${libname} PRIVATE CUDA::cudart)
    elseif (SPH_EXA_WITH_HIP)
        target_link_libraries(${libname} PRIVATE hip::host cstone_thrust_incl)
        target_compile_definitions(${libname} PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
        set_target_properties(${libname} PROPERTIES LINKER_LANGUAGE CXX)
    endif ()
endif ()

