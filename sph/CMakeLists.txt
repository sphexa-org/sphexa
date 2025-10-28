cmake_minimum_required(VERSION 3.22)

set(SPH_EXA_DIR ${PROJECT_SOURCE_DIR}/main/src)

project(sph CXX)

add_subdirectory(include)
if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
