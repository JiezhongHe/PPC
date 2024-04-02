set(INDEX_SRC cycle_counting.cpp feature_selector.cpp index.cpp path_counting.cpp)

# Add source files for the library
add_library(index STATIC ${INDEX_SRC})

# Set include directories for the target
target_include_directories(index PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})