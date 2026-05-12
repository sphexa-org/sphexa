add_library(polytrope polytrope_profile.cpp)

target_include_directories(polytrope PRIVATE ${CSTONE_DIR})
target_link_libraries(polytrope OpenMP::OpenMP_CXX)
