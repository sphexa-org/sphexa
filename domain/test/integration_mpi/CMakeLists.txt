include(cstone_add_mpi_test)

# workaround for compiling MPI with nvcc on cray systems
if (CSTONE_WITH_CUDA AND MPI_CXX_INCLUDE_PATH STREQUAL "" AND DEFINED ENV{CRAY_MPICH2_DIR} OR DEFINED ENV{CRAY_MPICH_DIR})
    message(STATUS "Applying MPI include workaround on CRAY")
    set(MPI_CXX_INCLUDE_PATH $ENV{MPICH_DIR}/include)
endif ()

cstone_add_mpi_test(globaloctree GlobalTreeTests 2 "2r;cpu" SOURCES globaloctree.cpp)
cstone_add_mpi_test(exchange_halos GlobalHaloExchange 2 "2r;cpu" SOURCES exchange_halos.cpp)
cstone_add_mpi_test(box_mpi GlobalBox 2 "2r;cpu" SOURCES box_mpi.cpp)
cstone_add_mpi_test(exchange_focus GlobalFocusExchange 2 "2r;cpu" SOURCES exchange_focus.cpp)
cstone_add_mpi_test(exchange_keys GlobalKeyExchange 2 "2r;cpu" SOURCES exchange_keys.cpp)
cstone_add_mpi_test(domain_2ranks GlobalDomain2Ranks 2 "2r;cpu" SOURCES domain_2ranks.cpp)
cstone_add_mpi_test(domain_resize GlobalDomainResize 2 "2r;cpu" SOURCES domain_resize.cpp)

cstone_add_mpi_test(treedomain GlobalDomainTreeIntregration 12 "12r;cpu" SOURCES treedomain.cpp)
cstone_add_mpi_test(exchange_general GeneralFocusExchange 12 "12r;cpu" SOURCES exchange_general.cpp)
cstone_add_mpi_test(exchange_domain GlobalDomainExchange 12 "12r;cpu" SOURCES exchange_domain.cpp)
cstone_add_mpi_test(focus_tree FocusTreeIntregration 12 "12r;cpu" SOURCES focus_tree.cpp)
cstone_add_mpi_test(domain_nranks GlobalDomainNRanks 12 "12r;cpu" SOURCES domain_nranks.cpp)

if (CSTONE_WITH_CUDA OR CSTONE_WITH_HIP)
    cstone_add_mpi_test(exchange_general_gpu ExchangeGeneralGpu 4 "4r;gpu" SOURCES exchange_general_gpu.cpp GPU)
    cstone_add_mpi_test(exchange_halos_gpu GlobalHaloExchangeGpu 2 "2r;gpu" SOURCES exchange_halos_gpu.cpp GPU)
    cstone_add_mpi_test(exchange_domain_gpu GlobalDomainExchangeGpu 2 "2r;gpu" SOURCES exchange_domain_gpu.cpp GPU)
    cstone_add_mpi_test(assignment_gpu AssignmentGpu 2 "2r;gpu" SOURCES assignment_gpu.cpp GPU)
    cstone_add_mpi_test(domain_gpu DomainGpu 2 "2r;gpu" SOURCES domain_gpu.cpp GPU)
endif ()
