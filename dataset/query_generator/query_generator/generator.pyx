# distutils: language = c++
from libcpp.vector cimport vector
cimport generator

import networkx as nx
import ctypes as ct
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as inc

cdef class QueryGenerator:
    cdef generator.graph_generator gen

    def __init__(self, graph_list):
        cdef generator.graph_cpp g
        for graph in graph_list:
            g.clear()
            self.from_nx_to_cpp(graph, g)
            self.gen.add_graph(g)

    cdef from_nx_to_cpp(self, graph, generator.graph_cpp& graph_cpp):
        for v in graph.nodes():
            graph_cpp.label_map[v] = graph.nodes[v]['label']
            graph_cpp.adj_list[v] = [n for n in graph[v]]

    cdef from_cpp_to_nx(self, generator.graph_cpp& graph):
        cdef:
            unordered_map[int, int].iterator itm = graph.label_map.begin()
            unordered_map[int, unordered_set[int]].iterator ita = graph.adj_list.begin()
        G = nx.Graph()
        while itm != graph.label_map.end():
            G.add_node(deref(itm).first, label=deref(itm).second)
            inc(itm)
        
        
        while ita != graph.adj_list.end():
            neighbors = list(deref(ita).second)
            v = deref(ita).first
            for n in neighbors:
                if v < n:
                    G.add_edge(v, n)
            inc(ita)
        return G

    def generate_query(self, hops, query_size, check_cycle, is_vertex):
        cdef generator.graph_cpp query
        cdef int query_anchor
        cdef int query_anchor_u
        cdef int data_id
        cdef int data_anchor
        cdef int data_anchor_v
        cdef bint check_cycle_bint = check_cycle
        if is_vertex == True:
            self.gen.generate_query(hops, query_size, query, query_anchor, data_id, data_anchor, check_cycle_bint)
            return (self.from_cpp_to_nx(query), query_anchor, data_id, data_anchor)
        else:
            self.gen.generate_query_edge(hops, query_size, query, query_anchor, query_anchor_u, data_id, data_anchor, data_anchor_v, check_cycle_bint)
            return (self.from_cpp_to_nx(query), query_anchor, query_anchor_u, data_id, data_anchor, data_anchor_v)
        

    def generate_neg_query(self, hops, query_size, check_cycle, is_vertex):
        cdef generator.graph_cpp query
        cdef int query_anchor
        cdef int query_anchor_u
        cdef int data_id
        cdef int data_anchor
        cdef int data_anchor_v
        cdef bint check_cycle_bint = check_cycle

        if is_vertex == True:
            self.gen.generate_neg_query_on_multigraph(hops, query_size, query, query_anchor, data_id, data_anchor, check_cycle_bint)
            return (self.from_cpp_to_nx(query), query_anchor, data_id, data_anchor)
        else:
            self.gen.generate_neg_query_on_multigraph_edge(hops, query_size, query, query_anchor, query_anchor_u, data_id, data_anchor, data_anchor_v, check_cycle_bint)
            return (self.from_cpp_to_nx(query), query_anchor, query_anchor_u, data_id, data_anchor, data_anchor_v)

    def generate_pos_query(self, hops, query_size, check_cycle):
        cdef generator.graph_cpp query
        cdef int query_anchor
        cdef int data_id
        cdef int data_anchor
        cdef bint check_cycle_bint = check_cycle
        self.gen.generate_pos_query(hops, query_size, query, check_cycle_bint)

        return self.from_cpp_to_nx(query)

def check_iso_by_filtering(data_graph, data_anchor, query_graph, query_anchor, hops):
    cdef:
        generator.graph_cpp query
        generator.graph_cpp data
        
    gen = QueryGenerator([])
    gen.from_nx_to_cpp(data_graph, data)
    gen.from_nx_to_cpp(query_graph, query)

    return generator.check_iso_in_k_hop(&data, data_anchor, &query, query_anchor, hops)

