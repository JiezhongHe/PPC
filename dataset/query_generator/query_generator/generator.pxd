# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string

ctypedef unsigned int vertex_id

cdef extern from "lib/graph.hpp":
    cdef cppclass graph_cpp:
        unordered_map[int, unordered_set[int] ] adj_list;
        unordered_map[int, int] label_map;
        graph_cpp() except +
        graph_cpp(unordered_map[int, unordered_set[int] ]& adj_list_, unordered_map[int, int]& label_map_) except +
        graph_cpp(const graph_cpp& g) except +
        
        void clear() except +
        void get_k_hop_neihgborhood(int anchor, int hops, unordered_set[int]& explored) except +
        void print() except +

cdef extern from "lib/generator.hpp":
    cdef cppclass graph_generator:
        graph_generator() except +
        void add_graph(const graph_cpp& graph) except +
        void generate_pos_query(int hops, int query_size, graph_cpp& query, bint check_cycle) except +
        void generate_query(int hops, int query_size, graph_cpp& query, int& query_anchor, int& data_id, int& data_anchor, bint check_cycle) except +
        void generate_neg_query_on_multigraph(int hops, int query_size, graph_cpp& query, int& query_anchor, int& data_id, int& data_anchor, bint check_cycle) except +
        void generate_query_edge(int hops, int query_size, graph_cpp& query, int& query_anchor, int& query_anchor_u, int& data_id, int& data_anchor, int& data_anchor_v, bint check_cycle) except +
        void generate_neg_query_on_multigraph_edge(int hops, int query_size, graph_cpp& query, int& query_anchor, int& query_anchor_u, int& data_id, int& data_anchor, int& data_anchor_v, bint check_cycle) except +

cdef extern from "lib/utils.hpp":
    cdef bint check_iso_in_k_hop(graph_cpp* data_graph, int data_anchor, graph_cpp* query_graph, int query_anchor, int hops) except +

