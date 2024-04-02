## Compile

```zsh
mkdir build
cd build
cmake ..
make
```


## Execute
A running example. The time limit is 300 seconds in default;

```zsh
cd build/src
./main.o --data ../../../dataset/matching/yeast/yeast.graph --query ../../../dataset/matching/yeast/queries/ --num 100000 --index ../../.index/yeast_128/
```

where:
- `--data`, the path of the data graph file
- `--query`, the path of the query graph file
- `--num`, number of results intended to find
- `--index`, the path containing the PPC-index, note that this directory must contain the PPC indices in four configurations `cycle_in_edge.index`, `cycle_in_vertex.index`, `path_in_edge.index` and `path_in_vertex.index`.

## Configuration
You can configure the enumeration process by adjusting the following macros in 'configuration/config.h'

| Macro | Description |
| :-----------------------------------------------: | :-------------: |
|TIME_LIMIT| set the time limit for the enumeration process|
|ENABLE_PRE_FILTERING| utilize PPC index for filtering|
|INDEX_ORDER| utilize PPC index for candidate ordering|
|COMPACT| PPC index in compact format to save the validation time |

## Format of the input graph
our graph is similar to the format of the negative sample. However, each file only contains one graph

```txt
t 2 1
v 0 2 1
v 1 3 1
e 0 1
```

where :
- `t 2 1`, 2 vertices and 1 edges
- `v 0 2 1`, vertex id is 0, label is 2, degree is 1
- `e 0 1`, an edge connection vertex 0 and 1