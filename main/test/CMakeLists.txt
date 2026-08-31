list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/main/cmake")
include(main_add_test)

set(UNIT_TESTS
        init/grid.cpp
        io/arg_parser.cpp
        io/h5part_wrapper.cpp
        io/id_tag_utils.cpp
        observables/gravitational_waves.cpp
        sphexa/particles_data.cpp
        test_main.cpp)

if (SPH_EXA_WITH_H5HUT)
    main_add_test(frontend_units
        SOURCES ${UNIT_TESTS}
        H5HUT)
endif ()

if (SPH_EXA_WITH_CUDA AND SPH_EXA_WITH_H5HUT)
    main_add_test(frontend_units_cuda
        SOURCES io/id_tag_utils_gpu.cu test_main.cpp
        LINK cstone_gpu io_gpu sph_gpu OpenMP::OpenMP_CXX CUDA::cudart
        INCLUDE ${COOLING_DIR}
        LABELS "01r;gpu")
endif ()
