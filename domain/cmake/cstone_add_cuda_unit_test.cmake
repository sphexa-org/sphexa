include(cstone_add_test)

function(cstone_add_cuda_unit_test target)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK;INCLUDE;LABELS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "cstone_add_cuda_unit_test: SOURCES required")
    endif()

    if(NOT ARG_LABELS)
        set(ARG_LABELS "01r;gpu")
    endif()

    if(CSTONE_WITH_HIP)
        set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()

    if(CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
        set(_obj_target ${target}_obj)

        add_library(${_obj_target} OBJECT ${ARG_SOURCES})
        target_link_libraries(${_obj_target} PUBLIC OpenMP::OpenMP_CXX GTest::gtest_main)
        # We have to use GCC (mpicxx) as the linker because it was used for GTest
        # which then pulls in -lmpi_gnu_123 -lmpi_gtl_hsa not found by clang
        # We can't link to roc::rocthrust in this case, because it adds --hip-link --offload-arch
        # to the link line, which is no recognized by GCC
        # Workaround: we just add the interface includes defined by rocthrust and hipcub
        target_link_libraries(${_obj_target} PRIVATE cstone_gpu cstone_thrust_incl)
        target_include_directories(${_obj_target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_include_directories(${_obj_target} PRIVATE ${PROJECT_SOURCE_DIR}/test)

        if(ARG_INCLUDE)
            target_include_directories(${_obj_target} PRIVATE ${ARG_INCLUDE})
        endif()

        add_executable(${target} test_main.cpp)
        target_link_libraries(${target} PRIVATE ${_obj_target})

        cstone_add_test(${target} EXECUTABLE ${target} RANKS 1)
        set_tests_properties(${target} PROPERTIES LABELS "${ARG_LABELS}")
        install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_gpu)

        if(CSTONE_WITH_HIP)
            set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
        endif()
    endif()
endfunction()
