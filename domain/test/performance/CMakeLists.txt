include(cstone_add_performance_test)

# hilbert.cpp, octree.cu and hilbert.cu are intentionally not tested
cstone_add_performance_test(peers.cpp peers_perf "1r;cpu")

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_cuda_performance_test(sph_density_driver.cu sph_density_test_gpu "1r;gpu")
endif()
