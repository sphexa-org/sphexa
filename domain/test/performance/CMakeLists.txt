include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf "01r;cpu")
cstone_add_performance_test(peers.cpp peers_perf "01r;cpu")

if (CSTONE_WITH_CUDA)
    cstone_add_performance_test(hilbert.cpp hilbert_perf "01r;gpu")
    # cudart seems to be required when compiling with some MPI compiler wrappers
    target_link_libraries(hilbert_perf PRIVATE ThrustOMP CUDA::cudart)
endif ()

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_cuda_performance_test(sph_density_driver.cu sph_density_test_gpu "01r;gpu")
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu "01r;gpu")
    cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu "01r;gpu")
endif()
