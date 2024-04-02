# PPC-index
Implementation of PPC-index

## Build
To build our project, run the following commands
`cd index/lib`
`mkdir build`
`cd build`
`cmake ..`
`make`

## Usage
### Index Construction
Firstly, run the following command to construct the PPC-index for yeast dataset

```bash
./run_build_index.o --data ../../../../dataset/enumeration/yeast/yeast.gr --se ../../../../dataset/enumeration/yeast/feature_finding/edge/ --sv ../../../../dataset/enumeration/yeast/feature_finding/vertex/ --fl 4 --fc 128 --output ../../../../dataset/enumeration/yeast/index --PV --PE --CV --CE
```

where:
- `--data`, the path of the input data graph
- `--se`, the path of the edge-anchored samples
- `--sv`, the path of the vertex-anchored samples
- `--fl`, the length of the generated features
- `--fc`, the size of the feature set
- `--output`, path of the constructed index
- `--PV, --PE, --CV, --CE`, the indictors that correspond to the PPC-PV, PPC-PE, PPC-CV, PPC-CE indices respectively. They are optional. If one of them, i.e., `--PV` is not put on, then PPC-PV will not be constructed.
- there are more optional input parameters that can be configured. See more details in the source codes.


### Subgraph Enumeration
Secondly, run the following command to conduct subgraph enumeration experiments

```bash
./run_enumeration.o --query ../../../../dataset/enumeration/yeast/queries/ --data ../../../../dataset/enumeration/yeast/yeast.gr --PV ../../../../dataset/enumeration/yeast/index/yeast.gr_128_4_1_path_vertex.index --PE ../../../../dataset/enumeration/yeast/index/yeast.gr_128_4_1_path_edge.index --CV ../../../../dataset/enumeration/yeast/index/yeast.gr_128_4_1_cycle_vertex.index --CE ../../../../dataset/enumeration/yeast/index/yeast.gr_128_4_1_cycle_edge.index --order 1
```

where:
- `query`, the path of the query graphs(multiple queries are being executed)
- `data`, the path of the data graph
- `--PV, --PE, --CV, --CE`, path of the PPC-indices generated before
- `--order`, whether enabling the candidates reordering mechanism, set 0 to disable
- there are more optional input parameters that can be configured. See more details in the source codes.(Note that `--residual` must be consistent to the building process, which are enabled both in `./run_build_index.o` and `run_enumeration.o` in default)
- if you get annoyed with the progress stream printed, you can go to `configuration/config.h` and comment the macro `PRINT_BUILD_PROGRESS`

## Format of the files
### format of the graphs
take `dataset/enumeration/yeast/yeast.gr` for example. It looks like:

```txt
# 0
v 0 0 1
v 1 1 9
e 3078 3110
e 3078 3111
# 1
...
```
where:
- `# ${graph_id}`, graph id starts from 0
- `v ${node_id} ${node_label} ${node_degree}`, ${node_degree} can be neglected
- `e ${src} ${dst}`
- multiple graphs can be recorded in the same file

### format of the samples
take `data/enumeration/yeast/query_vertex_8.gr`

```txt
# 0
v 0 2
v 1 19
e 0 1
q 4
d 164 0
```
where:
- `# 0`, is the sample id, and the following parts are the query graph
- `q ${query_node_id}`, is the query anchor node
- `d ${data_node_id} ${data_graph_id}`, is the data anchor node, for retrieval datasets, ${data_graph_id} may be larger than 0

## About Dataset
we only provide our queries in this project to keep our project in a reasonable size. The data graphs are provided in https://github.com/RapidsAtHKUST/SubgraphMatching and https://github.com/InfOmics/Dataset-GRAPES


