cmake_minimum_required(VERSION 3.22)

project(cornerstone-octree CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(MPI)
find_package(OpenMP REQUIRED)

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERAFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
# i.e. we add e.g. the -fopenmp flag to the linker
get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS)
if (NOT OpenMP_CXX_INTERFACE_LINK_OPTIONS)
    # fill in using INTERFACE_COMPILE_OPTIONS
    get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OpenMP_CXX_INTERFACE_LINK_OPTIONS})
endif()

include(CTest)
include(CheckLanguage)

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(CSTONE_WITH_CUDA "Enable building for NVIDIA GPUs" ON)
option(CSTONE_WITH_HIP "Enable building for AMD GPUs" ON)
option(CSTONE_WITH_GPU_AWARE_MPI "Enable CUDA-aware MPI communication" OFF)

if(GPU_DIRECT)
    message(WARNING "Option GPU_DIRECT is deprecated and will be removed. Use -DCSTONE_WITH_GPU_AWARE_MPI=ON instead.")
    set(CSTONE_WITH_GPU_AWARE_MPI ON)
endif()

if(CSTONE_WITH_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        # Thrust and CUB are part of CUDAToolkit
        set(CSTONE_THRUST_LIBS "")
        set(CMAKE_CUDA_STANDARD 20)
    else()
        message(STATUS "No CUDA support")
        set(CSTONE_WITH_CUDA OFF)
    endif()
endif()

# interface target with just the THRUST and CUB includes, as linking those targets adds
# --hip-link --offload-arch, which doesn't work if a) we have to use GCC (mpicxx) for linking
# and b) we can't put HIP objects into a static archive when compiling GTest runners
add_library(cstone_thrust_incl INTERFACE)

if(CSTONE_WITH_HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        find_package(hip)
        find_package(rocthrust REQUIRED)
        find_package(hipcub REQUIRED)
        get_property(rocthrust_include TARGET roc::rocthrust PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        get_property(hipcub_include TARGET hip::hipcub PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        get_property(rocprim_include TARGET roc::rocprim PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(cstone_thrust_incl INTERFACE
            ${rocthrust_include} ${hipcub_include} ${rocprim_include})

        set(CSTONE_THRUST_LIBS roc::rocthrust hip::hipcub)

        set(CMAKE_HIP_STANDARD 20)
    else()
        message(STATUS "No HIP support")
        set(CSTONE_WITH_HIP OFF)
    endif()
endif()

if(CSTONE_WITH_HIP AND CSTONE_WITH_CUDA)
    message(FATAL_ERROR "CUDA and HIP cannot both be turned on")
endif()

add_subdirectory(include)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()

