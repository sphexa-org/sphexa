include(cstone_add_cuda_unit_test)

set(UNIT_TEST_SOURCES_GPU
        cuda/device_vector.cu
        cuda/memory.cu
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
        neighbors/compressneighbors.cu
        neighbors/reducearray.cu
        neighbors/ijloop.cu
        neighbors/ijloop_upsweep.cu)

cstone_add_cuda_unit_test(component_units_gpu SOURCES ${UNIT_TEST_SOURCES_GPU})
