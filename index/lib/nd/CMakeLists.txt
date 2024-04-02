set(ND_SRC bucket.cpp computesetintersection.cpp 
    hierarchy.cpp k3-4.cpp kcore.cpp ktruss.cpp
    nd_interface.cpp nucleus_decomposition.cpp util.cpp
)

# Add source files for the library
add_library(nd STATIC ${ND_SRC})

# Set include directories for the target
target_include_directories(nd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})