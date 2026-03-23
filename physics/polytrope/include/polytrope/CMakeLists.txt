add_library(polytrope polytrope_profile.cpp)

target_include_directories(polytrope PRIVATE ${CSTONE_DIR} ${SPH_DIR}
        ${MPI_CXX_INCLUDE_PATH} ${POLYTROPE_DIR})

target_link_libraries(polytrope PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
