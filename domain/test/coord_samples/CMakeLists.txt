include(cstone_add_unit_test)

cstone_add_unit_test(coordinate_test
    SOURCES random.cpp test_main.cpp DESTINATION coord_samples LABELS "01r;cpu")
