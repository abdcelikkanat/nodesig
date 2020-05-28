cmake_minimum_required(VERSION 3.16)
project(cplusplus)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/Graph.cpp src/Model.cpp)
set(LIB_FILES lib/Graph.h lib/Model.h)

include_directories(lib)

include_directories(lib/eigen)

add_subdirectory(src)
add_subdirectory(tests)



add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SOURCE_FILES} ${LIB_FILES})
#target_link_libraries(${CMAKE_PROJECT_NAME}_run ${CMAKE_PROJECT_NAME}_lib)


#add_executable(main main.cpp src/Graph.cpp src/Matrix.cpp src/Model.cpp lib/Graph.h lib/Matrix.h lib/Model.h)
#add_executable(tests tests/test_graph.ddcpp src/Graph.cpp lib/Graph.h tests/test_sparse_matrix.cpp)


# Clone Eigen Lib
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/lib/eigen)
if(NOT EXISTS ${EIGEN_DIR})
    find_package(Git REQUIRED)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://gitlab.com/libeigen/eigen.git ${EIGEN_DIR}
    )
endif()