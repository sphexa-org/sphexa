include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf)
cstone_add_performance_test(peers.cpp peers_perf)

# TODO(iomaganaris): reenable for CPU once thrust includes are fixed for CPU builds
if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_performance_test(hilbert.cpp hilbert_perf)
    # link to cudart to get thrust includes
    target_link_libraries(hilbert_perf PRIVATE CUDA::cudart)
endif ()

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_cuda_performance_test(sph_density_driver.cu sph_density_test_gpu)
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu)
    cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu)
endif()
