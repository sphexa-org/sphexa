
function(enableH5Part exename)
    if(SPH_EXA_WITH_H5PART)
        target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
        target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)

        # necessary because the HDF5_INLUCDE_DIRS is empty on CRAY and HIP-compilation is done without the cray-wrapper
        target_include_directories(${exename} PUBLIC $ENV{HDF5_DIR}/include)
        target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
    endif()
endfunction()

add_library(io id_tag_utils.cpp ifile_io_ascii.cpp ifile_io_hdf5.cpp arg_parser.cpp)
target_include_directories(io PRIVATE ${CSTONE_DIR} ${SPH_EXA_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(io PRIVATE ${MPI_CXX_LIBRARIES})
enableH5Part(io)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(id_tag_utils_gpu OBJECT id_tag_utils.cu)
    target_include_directories(id_tag_utils_gpu PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_link_libraries(id_tag_utils_gpu PRIVATE CUDA::cudart)
endif()
