cmake_minimum_required(VERSION 3.6)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(test_xgc)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)
find_package(MGARD REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

add_executable(test_cyclone test_cyclone.cpp)
add_executable(test_cyclone_rz test_cyclone_rz.cpp)
add_executable(test_cyclone_3d test_cyclone_3d.cpp)
add_executable(test_cyclone_prz test_cyclone_prz.cpp)
add_executable(test_cyclone_p3d test_cyclone_p3d.cpp)

target_link_libraries(test_cyclone adios2::adios2 mgard::mgard MPI::MPI_C)
target_link_libraries(test_cyclone_prz adios2::adios2 mgard::mgard MPI::MPI_C)
target_link_libraries(test_cyclone_rz adios2::adios2 mgard::mgard MPI::MPI_C)
target_link_libraries(test_cyclone_3d adios2::adios2 mgard::mgard MPI::MPI_C)
target_link_libraries(test_cyclone_p3d adios2::adios2 mgard::mgard MPI::MPI_C)
