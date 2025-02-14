
if (SPH_EXA_WITH_GRACKLE)
    add_subdirectory(cooling)
endif ()
if (SPH_EXA_WITH_DISKS)
    add_subdirectory(Disk)
endif ()
