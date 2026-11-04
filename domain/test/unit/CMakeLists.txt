include(cstone_add_unit_test)

set(UNIT_TESTS
        domain/domaindecomp.cpp
        domain/index_ranges.cpp
        domain/layout.cpp
        fields/field_get.cpp
        focus/octree_focus.cpp
        focus/source_center.cpp
        neighbors/findneighbors.cpp
        primitives/clz.cpp
        primitives/concat_vector.cpp
        primitives/gather.cpp
        primitives/math.cpp
        sfc/box.cpp
        sfc/common.cpp
        sfc/hilbert.cpp
        sfc/mixed_hilbert.cpp
        sfc/morton.cpp
        sfc/sfc.cpp
        traversal/boxoverlap.cpp
        traversal/collisions_a2a.cpp
        traversal/discovery.cpp
        traversal/macs.cpp
        traversal/peers.cpp
        traversal/traversal.cpp
        tree/btree.cpp
        tree/csarray.cpp
        tree/continuum.cpp
        tree/octree.cpp
        tree/cs_util.cpp
        util/array.cpp
        util/pack_buffers.cpp
        util/tuple_util.cpp
        util/type_list.cpp
        util/value_list.cpp
        test_main.cpp)

cstone_add_unit_test(component_units SOURCES ${UNIT_TESTS} neighbors/all_to_all.hpp INCLUDE ../)
cstone_add_unit_test(component_units_omp SOURCES ${UNIT_TESTS} INCLUDE ../ OPENMP)
