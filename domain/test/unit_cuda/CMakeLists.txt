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
        focus/octree_focus.cu
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
    target_link_libraries(gpu_unit_obj PUBLIC cstone_gpu OpenMP::OpenMP_CXX GTest::gtest_main)
    target_include_directories(gpu_unit_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(gpu_unit_obj PRIVATE ${PROJECT_SOURCE_DIR}/test)

    add_executable(${testname} test_main.cpp)
    target_link_libraries(${testname} PUBLIC gpu_unit_obj)
    cstone_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_cuda)
endif()

if(CMAKE_HIP_COMPILER)
    target_link_libraries(gpu_unit_obj PUBLIC hip::host)
    target_compile_definitions(gpu_unit_obj PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
endif()

unset(testname)
