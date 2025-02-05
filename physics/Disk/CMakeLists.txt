cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 20)

set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
set(MAIN_APP_DIR ${CMAKE_SOURCE_DIR}/main/src)
set(DISK_DIR ${CMAKE_SOURCE_DIR}/physics/Disk/include)
set(SPH_DIR ${PROJECT_SOURCE_DIR}/sph/include)

add_subdirectory(include)

if (BUILD_TESTING)
    add_subdirectory(test)
endif ()
