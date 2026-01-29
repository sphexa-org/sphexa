if(CMAKE_HIP_COMPILER)
    set_source_files_properties(ewald.cu multipole_holder.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    # workaround for compiling MPI with nvcc on cray systems
    if (MPI_CXX_INCLUDE_PATH STREQUAL "" AND DEFINED ENV{CRAY_MPICH2_DIR} OR DEFINED ENV{CRAY_MPICH_DIR})
        message(STATUS "Applying MPI include workaround on CRAY")
        set(MPI_CXX_INCLUDE_PATH $ENV{MPICH_DIR}/include)
    endif()
    message(STATUS "MPI include is " ${MPI_CXX_INCLUDE_PATH})

    # intermdiate object to prevent --hip-link from hipcub to propagate to the final link line
    add_library(ryoanji_obj OBJECT ewald.cu multipole_holder.cu)
    # link to cstone_gpu because of its INTERFACE compile definition
    target_link_libraries(ryoanji_obj PUBLIC cstone_gpu)
    target_include_directories(ryoanji_obj PUBLIC ${PROJECT_SOURCE_DIR}/src ${MPI_CXX_INCLUDE_PATH})
    add_library(ryoanji $<TARGET_OBJECTS:ryoanji_obj> $<TARGET_OBJECTS:ryoanji_kernels>)
    target_link_libraries(ryoanji PUBLIC cstone_gpu ${MPI_CXX_LIBRARIES})
endif()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(ryoanji_obj PRIVATE hip::hipcub)
endif ()
