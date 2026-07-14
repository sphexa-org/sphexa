include(GNUInstallDirs)

# nanobind requires CMake FindPython targets (Python::Module, Python::Interpreter)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

# Detect the nanobind pip package and import it into CMake.
# nanobind must be installed beforehand, e.g. via domain/requirements.txt:
#   pip install -r domain/requirements.txt
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(cstone_sfc cstone_sfc_bindings.cpp)
target_include_directories(cstone_sfc PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS cstone_sfc LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
