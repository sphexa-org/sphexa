include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf)
cstone_add_performance_test(peers.cpp peers_perf)

if (CMAKE_CUDA_COMPILER)
    cstone_add_performance_test(hilbert.cpp hilbert_perf)
    target_link_libraries(hilbert_perf PRIVATE ThrustOMP)
endif ()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    cstone_add_cuda_performance_test(sph_density_driver.cu sph_density_test_gpu)
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu)
    cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu)
endif()
