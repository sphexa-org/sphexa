include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf)
cstone_add_performance_test(peers.cpp peers_perf)

<<<<<<< HEAD
if (CMAKE_CUDA_COMPILER)
    cstone_add_performance_test(hilbert.cpp hilbert_perf CUDA::cudart)
    target_link_libraries(hilbert_perf PRIVATE ThrustOMP)
=======
if (CSTONE_WITH_CUDA)
    cstone_add_performance_test(hilbert.cpp hilbert_perf)
    # cudart seems to be required when compiling with some MPI compiler wrappers
    target_link_libraries(hilbert_perf PRIVATE ThrustOMP CUDA::cudart)
>>>>>>> upstream/develop
endif ()

if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_cuda_performance_test(sph_density_driver.cu sph_density_test_gpu)
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu)
    cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu)
endif()
