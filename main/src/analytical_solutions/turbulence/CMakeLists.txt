set(SOURCES density_pdf.hpp density_pdf.cpp)
add_executable(density_pdf ${SOURCES})
target_include_directories(density_pdf PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src ${CMAKE_BINARY_DIR}/main/src ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(density_pdf io stdc++fs OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})

install(TARGETS density_pdf RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(setup_gitinfo)
configure_file(
        ${PROJECT_SOURCE_DIR}/main/src/version.h.in
        ${CMAKE_BINARY_DIR}/main/src/version.h
)
