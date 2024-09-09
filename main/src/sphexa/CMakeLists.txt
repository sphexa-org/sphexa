
function(enableInSituViz exename)
    if(INSITU STREQUAL "Catalyst")
        target_include_directories(${exename} PRIVATE ..)
        target_compile_definitions(${exename}
            PRIVATE
            "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
            SPH_EXA_USE_CATALYST2)
        target_link_libraries(${exename} PRIVATE catalyst::catalyst)
    elseif(INSITU STREQUAL "Ascent")
        target_include_directories(${exename} PRIVATE ..)
        target_compile_definitions(${exename}
            PRIVATE
            SPH_EXA_USE_ASCENT)
        target_link_libraries(${exename} PRIVATE ascent::ascent_mpi)
    endif()
endfunction()

function(enableGpuDirect exename)
    if(GPU_DIRECT)
        target_compile_definitions(${exename} PRIVATE USE_GPU_DIRECT)
    endif()
endfunction()

set(exename sphexa)
add_executable(${exename} sphexa.cpp)
target_include_directories(${exename} PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_link_libraries(${exename} PRIVATE io sim_init propagator_init observables cstone_gpu OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/physics/cooling/include/)

enableInSituViz(${exename})
enableGrackle(${exename})
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CMAKE_CUDA_COMPILER)
    add_executable(${exename}-cuda sphexa.cpp)
    target_include_directories(${exename}-cuda PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_include_directories(${exename}-cuda PUBLIC ${CMAKE_BINARY_DIR}/main/src)
    target_compile_definitions(${exename}-cuda PRIVATE USE_CUDA)
    target_link_libraries(${exename}-cuda PRIVATE cstone_gpu ryoanji sph_gpu io sim_init_gpu propagator_init_gpu observables_gpu OpenMP::OpenMP_CXX
        ${MPI_CXX_LIBRARIES} CUDA::cudart)
    enableInSituViz(${exename}-cuda)
    enableGpuDirect(${exename}-cuda)
    enableGrackle(${exename}-cuda)
    install(TARGETS ${exename}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif(CMAKE_HIP_COMPILER)
    add_executable(${exename}-hip sphexa.cpp)
    target_include_directories(${exename}-hip PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_compile_definitions(${exename}-hip PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    target_link_libraries(${exename}-hip PRIVATE cstone_gpu ryoanji sph_gpu io sim_init_gpu propagator_init_gpu observables_gpu OpenMP::OpenMP_CXX
        ${MPI_CXX_LIBRARIES} hip::host)
    set_target_properties(${exename}-hip PROPERTIES LINKER_LANGUAGE CXX)
    enableInSituViz(${exename}-hip)
    enableGpuDirect(${exename}-hip)
    enableGrackle(${exename}-hip)
    install(TARGETS ${exename}-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
