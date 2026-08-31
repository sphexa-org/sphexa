
set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
set(MAIN_APP_DIR ${CMAKE_SOURCE_DIR}/main/src)
set(DISK_DIR ${CMAKE_SOURCE_DIR}/physics/Disk/include)
set(SPH_DIR ${PROJECT_SOURCE_DIR}/sph/include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(disk_add_gpu_object)

add_subdirectory(include)

if (BUILD_TESTING)
    include(disk_add_test)
    add_subdirectory(test)
endif ()
