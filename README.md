# PPC-index
Implementation of PPC-index

## Build
Under the root directory, there are three directories `dataset`, `index` and `run_exp`. Our project is originally built with cython and invoked by the experimental scripts in `run_exp`. Recently we update the the source code and reorganize our project with cmake. Hence the scripts in `run_exp` are temporarily deprecated. In the future, we will update the source code to be compatible with python and the scripts in `run_exp`.

To build our project, run the following commands
`cd index/lib`
`mkdir build`
`cd build`
`cmake ..`
`make`

## Usage
After compiling the project, there are three executable binaries `run_build_index.o`, `run_enumeration.o` and `run_retrieval.o` under directory `build/src/`, which corresponds to `index construction`, `subgraph enumeration` and `subgraph retrieval` respectively.

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

After constructions, there are eight files including four indices and four feature files in the output directory `../../../../dataset/enumeration/yeast/index`. Note that if some of the target files are already generated, our construction program will skip those generated files to save the time cost.

### Index Application
the subgraph retrieval and matching sources are putted in the directories `retrieval` and `matching`.

### Sample Generation
In directory `dataset/query_generator` is our sample generator. It is coded in cython. To build the it. First run the following commands:
`cd dataset/query_generator/`
`python setup.py build_ext --inplace`

To use our generator
`python generate_queries.py --input ${data_graph} --query_size ${size} --num_queries ${num} --anchor ${anchor} --output ${output}`

where:
- `--input`, the data graph file(for subgraph retrieval datasets, multiple datagraphs are collected in the same file, our scripts will randomly pick up the datagraph and generate a sample for it)
- `--query_size`, number of vertices in each query
- `--num_queries`, number of samples
- `--output`, output directory
- `--pos_neg`, (optional), we only generate negative samples in default and these negative samples are not easily filtered by NLF and LDF rules.

## Format of the negative samples
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
we only provide our queries in this project to keep our project in a reasonable size. The dataset demonstrated above(`enumeration/yeast` and `retrieval/pdbs`) are provided with data graph. For other data graphs, please contact jiezhonghe@outlook.com or refer to https://github.com/RapidsAtHKUST/SubgraphMatching and https://github.com/InfOmics/Dataset-GRAPES


