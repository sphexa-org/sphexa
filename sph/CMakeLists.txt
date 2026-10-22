cmake_minimum_required(VERSION 3.22)

project(sph CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(include)
if (BUILD_TESTING)
  add_subdirectory(test)
endif ()
