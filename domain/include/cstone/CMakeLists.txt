add_subdirectory(cuda)
add_subdirectory(focus)
add_subdirectory(halos)
add_subdirectory(primitives)
add_subdirectory(sfc)
add_subdirectory(traversal)
add_subdirectory(tree)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(cstone_gpu $<TARGET_OBJECTS:gather_halos_obj>
                           $<TARGET_OBJECTS:octree_obj>
                           $<TARGET_OBJECTS:primitives_gpu_obj>
                           $<TARGET_OBJECTS:sfc_obj>
                           $<TARGET_OBJECTS:traversal_obj>
                           $<TARGET_OBJECTS:source_center_gpu_obj>
                           $<TARGET_OBJECTS:gpu_utils_obj>)

    if (CSTONE_WITH_GPU_AWARE_MPI)
        target_compile_definitions(cstone_gpu INTERFACE CSTONE_HAVE_GPU_AWARE_MPI)
    endif()
endif ()