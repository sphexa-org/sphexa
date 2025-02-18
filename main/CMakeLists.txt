
set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)
set(COOLING_DIR ${PROJECT_SOURCE_DIR}/physics/cooling/include)
set(RYOANJI_DIR ${PROJECT_SOURCE_DIR}/ryoanji/src)
set(SPH_DIR ${PROJECT_SOURCE_DIR}/sph/include)
set(DISK_DIR ${PROJECT_SOURCE_DIR}/physics/Disk/include)

add_subdirectory(src)
if (BUILD_TESTING)
    add_subdirectory(test)
    add_subdirectory(test/mpi)
endif (BUILD_TESTING)
