include(cstone_add_test)

set(UNIT_TEST_SOURCES_GPU
        cuda/device_vector.cu
        domain/domaindecomp_gpu.cu
        halos/gather_halos_gpu.cu
        primitives/clz.cu
        primitives/concat_vector.cu
        primitives/gather.cu
        primitives/primitives_gpu.cu
        primitives/warpscan.cu
        focus/inject.cu
        sfc/common.cu
        traversal/groups.cu
        traversal/macs.cu
        tree/btree.cu
        tree/csarray.cu
        tree/octree.cu
)

if(CMAKE_HIP_COMPILER)
    set_source_files_properties(${UNIT_TEST_SOURCES_GPU} PROPERTIES LANGUAGE HIP)
endif()

set(testname component_units_cuda)

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gpu_unit_obj OBJECT ${UNIT_TEST_SOURCES_GPU})
    target_link_libraries(gpu_unit_obj PUBLIC OpenMP::OpenMP_CXX GTest::gtest_main)
    # We have to use GCC (mpicxx) as the linker because it was used for GTest
    # which then pulls in -lmpi_gnu_123 -lmpi_gtl_hsa not found by clang
    # We can't link to roc::rocthrust in this case, because it adds --hip-link --offload-arch
    # to the link line, which is no recognized by GCC
    # Workaround: we just add the interface includes defined by rocthrust and hipcub
    target_link_libraries(gpu_unit_obj PRIVATE cstone_gpu cstone_thrust_incl)
    target_include_directories(gpu_unit_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(gpu_unit_obj PRIVATE ${PROJECT_SOURCE_DIR}/test)

    add_executable(${testname} test_main.cpp)
    target_link_libraries(${testname} PRIVATE gpu_unit_obj)
    cstone_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_cuda)
endif()

if(CMAKE_HIP_COMPILER)
    set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
endif()

unset(testname)
