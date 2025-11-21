include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf)
cstone_add_performance_test(peers.cpp peers_perf)
cstone_add_performance_test(hilbert.cpp hilbert_perf)

if(CMAKE_CUDA_COMPILER)
    # link to cudart to get thrust includes
    target_link_libraries(hilbert_perf PRIVATE CUDA::cudart)
endif ()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    cstone_add_cuda_performance_test(neighbor_driver.cu neighbors_test_gpu)
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu)
    cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu)

    if (CMAKE_CUDA_COMPILER)
        set_source_files_properties(octree.cu PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)
    endif()
endif()
