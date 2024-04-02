# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.pair cimport pair
from libcpp.string cimport string

ctypedef unsigned int Vertex
ctypedef unsigned int Label
ctypedef unsigned short int Value
ctypedef Vertex* Vertex_ptr

cdef extern from "lib/embedding.hpp":
    cdef cppclass Tensor:
        int row_size
        int column_size
        Value* content

        Tensor(int row, int column, Value value=0) except +
        Value get_value(int i, int j) except +
        string to_string() except +


    cdef cppclass Index_manager:
        Index_manager(string filename_) except +
        Index_manager() except +

        void dump_tensor(Tensor* tensor) except +
        Tensor* load_graph_tensor(int graph_offset) except +
        void load_vertex_embedding(int graph_offset, Vertex v, vector[Value]& result) except +

    Tensor* merge_multi_Tensors(vector[Tensor*]& vec) except +

cdef extern from "lib/graph.hpp":
    cdef cppclass Graph:
        vector[Label] label_map
        vector[unordered_set[Vertex] ] adj
        vector[unordered_map[Vertex, Vertex]] nlf_data

        Graph() except +
        void print_graph() except +
        Graph(vector[vector[Vertex] ]& adj_, vector[Label]& label_map_) except +
        void set_up_edge_id_map() except +

cdef extern from "lib/feature_counter.hpp":
    cdef cppclass feature_counter:
        void count_for_edges(Graph& graph, Tensor**& result, int& result_size) except +
        void count_for_vertices(Graph& graph, Tensor**& result, int& result_size) except +
        vector[vector[Label]] get_features() except +
        bint get_residual() except +
        int get_feature_type() except +

cdef extern from "lib/cycle_counting.hpp":
    cdef cppclass Cycle_counter(feature_counter):
        Cycle_counter(bint enable_residual_, vector[vector[Label]]& label_paths_)
        void count_for_edges(Graph& graph, Tensor**& result, int& result_size) except +
        void count_for_vertices(Graph& graph, Tensor**& result, int& result_size) except +
        vector[vector[Label]] get_features() except +
        bint get_residual() except +
        int get_feature_type() except +

cdef extern from "lib/path_counting.hpp":
    cdef cppclass Path_counter(feature_counter):
        Path_counter(bint enable_residual_, vector[vector[Label]]& label_paths_)
        void count_for_edges(Graph& graph, Tensor**& result, int& result_size) except +
        void count_for_vertices(Graph& graph, Tensor**& result, int& result_size) except +
        vector[vector[Label]] get_features() except +
        bint get_residual() except +
        int get_feature_type() except +

cdef extern from "lib/index.hpp":
    cdef cppclass Index_constructer:
        Index_constructer(feature_counter* counter_) except +
        Index_constructer() except +
        Tensor* count_features(Graph& graph, int thread_num, int level) except +
        void construct_index_single_batch(Graph& graph, string index_file_name, int thread_num, int level) except +
        void construct_index_in_batch(Graph& graph, string index_file_name, int max_feature_size_per_batch, int thread_num, int level) except +

cdef extern from "lib/subgraph_retrieval.hpp":
    void sub_containment_vc(vector[Graph]& data_graphs, vector[Graph]& query_graphs, vector[int]& filtered_results, vector[double]& filtering_time) except +
    void sub_containment_vc_index(vector[Graph]& data_graphs, vector[Graph]& query_graphs, 
        vector[string]& index_edge_list, vector[string]& index_vertex_list, 
        vector[feature_counter*]& counter_edge_list, vector[feature_counter*]& counter_vertex_list,
        vector[int]& filtered_results, vector[double]& filtering_time
    ) except +
    void sub_containment_graph_level(vector[Graph]& data_graphs, vector[Graph]& query_graphs, 
        vector[string]& index_edge_list, vector[string]& index_vertex_list, 
        vector[feature_counter*]& counter_edge_list, vector[feature_counter*]& counter_vertex_list,
        vector[int]& filtered_results, vector[double]& filtering_time
    ) except +

cdef extern from "lib/subgraph_enumeration.hpp":
    void subgraph_enumeration(Graph* data_graphs, Graph* query_graphs, 
        long count_limit, long& result_count,
        double& enumeration_time, double& preprocessing_time,
        Tensor* query_vertex_emb, Tensor* data_vertex_emb,
        Tensor* query_edge_emb, Tensor* data_edge_emb,
        bint enable_order
    ) except +

cdef extern from "lib/feature_selector.hpp":
    cdef cppclass Feature_selector:
        Feature_selector() except +
        Feature_selector(int num_labels_, int feature_num_, int feature_length_, bint enable_cycle_, string tmp_path_) except +
        vector[vector[Label]] extract_for_singe_graph(vector[Graph]& sample_queries, vector[vector[Vertex]]& sample_query_anchors, vector[vector[Vertex]]& sample_data_anchors, Graph& data_graph, int level, int max_batch_size, int thread_num) except +
