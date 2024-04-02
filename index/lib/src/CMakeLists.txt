set(FILE_SRC
        auxiliary.cpp
        subgraph_enumeration.cpp
        subgraph_retrieval.cpp
)

function(add_exe target_file main_file)
        add_executable(${target_file} ${main_file} ${FILE_SRC})
        target_link_libraries(${target_file} graph)
        target_link_libraries(${target_file} index)
        target_link_libraries(${target_file} nd)
        target_link_libraries(${target_file} utility)
        target_link_libraries(${target_file} rt)
endfunction()

add_exe(debug.o debug.cpp)
# add_exe(run_enumeration.o run_enumeration.cpp)
add_exe(run_build_index.o run_build_index.cpp)
# add_exe(run_retrieval.o run_retrieval.cpp)

# Add source files for the executable
# add_executable(debug.o ${FILE_SRC})
# 
# # Link against the libraries
# # target_link_libraries(debug.o configuration)
# target_link_libraries(debug.o graph)
# target_link_libraries(debug.o index)
# target_link_libraries(debug.o nd)
# target_link_libraries(debug.o utility)
# target_link_libraries(debug.o rt)

# Set the output directory for built binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
