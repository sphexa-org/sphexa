cmake_minimum_required(VERSION 3.17)

project(sphexa-cooling)
set(CMAKE_CXX_STANDARD 20)

set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
set(MAIN_APP_DIR ${CMAKE_SOURCE_DIR}/main/src)
set(SPH_DIR ${CMAKE_SOURCE_DIR}/sph/include)

#include(CTest)
#include(CheckLanguage)
add_subdirectory(include/cooling)

if (BUILD_TESTING)
    add_subdirectory(test)
endif ()
