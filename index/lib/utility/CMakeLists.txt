set(UTILS_SRC core_decomposition.cpp
    embedding.cpp utils.cpp
)

# Add source files for the library
add_library(utility STATIC ${UTILS_SRC})

# Set include directories for the target
target_include_directories(utility PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})