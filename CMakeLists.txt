cmake_minimum_required(VERSION 3.22)

project(sphexa CXX C)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_MODULE_PATH 
     "${PROJECT_SOURCE_DIR}/cmake"
     "${PROJECT_SOURCE_DIR}/main/cmake"
)

option(BUILD_TESTING "build unit and integration tests" ON)
option(BUILD_ANALYTICAL "build analytical solution" ON)

option(SPH_EXA_WITH_CUDA "Enable building for NVIDIA GPUs" ON)
option(SPH_EXA_WITH_HIP "Enable building for AMD GPUs" ON)

set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)
set(CSTONE_TEST_DIR ${PROJECT_SOURCE_DIR}/domain/test)

if (BUILD_TESTING)
    include(setup_GTest)
endif()

find_package(MPI)
find_package(OpenMP)

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
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
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(SPH_EXA_WITH_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        set(CMAKE_CUDA_STANDARD 20)
    else()
        message(STATUS "No CUDA support")
        set(SPH_EXA_WITH_CUDA OFF)
    endif()
endif()

if(SPH_EXA_WITH_HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
        enable_language(HIP)
        find_package(hip)
        set(CMAKE_HIP_STANDARD 20)
    else()
        message(STATUS "No HIP support")
        set(SPH_EXA_WITH_HIP OFF)
    endif()
endif()

if(SPH_EXA_WITH_HIP AND SPH_EXA_WITH_CUDA)
    message(FATAL_ERROR "CUDA and HIP cannot both be turned on")
endif()

option(SPH_EXA_WITH_H5HUT "Enable HDF5 IO using the H5hut (H5part) library" ON)
if (SPH_EXA_WITH_H5HUT)
    set(SPH_EXA_HDF5_PARALLEL_MODE "independent" CACHE STRING "Parallel IO mode used by the H5Hut hdf5 IO")
    set_property(CACHE SPH_EXA_HDF5_PARALLEL_MODE PROPERTY STRINGS independent collective)
    find_package(HDF5)
    find_package(H5hut)
    if (HDF5_FOUND AND H5hut_FOUND)
    else()
      message(STATUS "No HDF5 support")
      set(SPH_EXA_WITH_H5hut OFF)
    endif ()
endif ()

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

if(INSITU STREQUAL "Catalyst")
    find_package(catalyst REQUIRED PATHS "$ENV{EBROOTPARAVIEW}/lib/cmake/paraview-5.10")
elseif(INSITU STREQUAL "Ascent")
	find_package(Ascent REQUIRED PATHS "$ENV{EBROOTASCENT}/lib/cmake/ascent")
endif()

option(SPH_EXA_WITH_GRACKLE "Enable radiative cooling with GRACKLE" OFF)
if (SPH_EXA_WITH_GRACKLE)
    add_subdirectory(extern/grackle)
endif()

option(SPH_EXA_WITH_DISKS "Enable disk physics and propagator" OFF)

add_subdirectory(domain)
add_subdirectory(ryoanji)
add_subdirectory(scripts)
add_subdirectory(sph)
add_subdirectory(physics)
add_subdirectory(main)
