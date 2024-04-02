cmake_minimum_required(VERSION 3.5)

project(PPC-index)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lrt -march=native -g -O3")


# Add subdirectories
# add_subdirectory(configuration)
add_subdirectory(graph)
add_subdirectory(index)
add_subdirectory(nd)
add_subdirectory(utility)
add_subdirectory(src)
