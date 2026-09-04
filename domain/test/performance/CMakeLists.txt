include(cstone_add_performance_test)

# NOTE: hilbert_perf[_gpu] and octree_perf_gpu are intentionally not run by the CI
# cstone_add_performance_test(octree.cpp octree_perf "1r;cpu")
cstone_add_performance_test(peers.cpp peers_perf "1r;cpu")

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_cuda_performance_test(sph_density_driver.cu sph_density_test_gpu "1r;gpu")
    # cstone_add_cuda_performance_test(octree.cu octree_perf_gpu "1r;gpu")
    # cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu "1r;gpu")
endif()
