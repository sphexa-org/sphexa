cmake_minimum_required(VERSION 3.22)

project(ryoanji CXX)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_TESTING "build unit and integration tests" ON)
option(GPU_DIRECT "Enable CUDA-aware MPI communication" OFF)

find_package(MPI)
find_package(OpenMP)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(RYOANJI_WITH_CUDA "Enable building for NVIDIA GPUs" ON)
option(RYOANJI_WITH_HIP "Enable building for AMD GPUs" ON)

if(RYOANJI_WITH_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        set(CMAKE_CUDA_STANDARD 20)
    else()
        message(STATUS "No CUDA support")
        set(RYOANJI_WITH_CUDA OFF)
    endif()
endif()

if(RYOANJI_WITH_HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
        enable_language(HIP)
        find_package(hip)
        set(CMAKE_HIP_STANDARD 20)
    else()
        message(STATUS "No HIP support")
        set(RYOANJI_WITH_HIP OFF)
    endif()
endif()

if(RYOANJI_WITH_HIP AND RYOANJI_WITH_CUDA)
    message(FATAL_ERROR "CUDA and HIP cannot both be turned on")
endif()

add_subdirectory(src)
if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
