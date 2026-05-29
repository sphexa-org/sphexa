include(GNUInstallDirs)
include(FetchContent)

# nanobind requires CMake FindPython targets (Python::Module, Python::Interpreter)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
set(CSTONE_NANOBIND_GIT_TAG "v2.2.0" CACHE STRING "nanobind git tag or commit to fetch")

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG ${CSTONE_NANOBIND_GIT_TAG}
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(cstone_sfc cstone_sfc_bindings.cpp)
target_include_directories(cstone_sfc PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS cstone_sfc LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
