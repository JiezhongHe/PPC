# distutils: language = c++
from libcpp.vector cimport vector
cimport ppc_index

import networkx as nx
import ctypes as ct
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.pair cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as inc

import time
import os
import random
import json

cdef class TensorPy:
    cdef public vector[vector[int]] values
    cdef public int dim
    cdef vector[int] shape
    def __init__(self, t):
        cdef vector[int] inner
        if type(t) != list:
            print("input must be a list")
            return
        for l in t:
            inner.clear()
            for j in l:
                inner.push_back(j)
            self.values.push_back(inner)
        self.dim = len(t[0])
        for l in t:
            if len(l) != self.dim:
                print("inner dimension is not consistent")
                return
        self.shape = [len(t), len(t[0])]

    def get_values(self):
        return list(self.values)

    def concat_with(self, other):
        cdef:
            vector[vector[int]].iterator itmb = self.values.begin()
            vector[vector[int]].iterator itme = self.values.end()
        
        other_value = other.get_values()
        offset = 0
        while itmb != itme:
            for e in other_value[offset]:
                deref(itmb).push_back(e)
            inc(itmb)
            offset += 1
        self.shape[1] += len(other_value[0])

    def size(self):
        return [self.shape[0], self.shape[1]]

    def __repr__(self):
        string = "---------------\n"
        for i in range(self.shape[0]):
            string += str(self.values[i])+"\n"
        string += "shape:"+str((self.shape[0], self.shape[1]))
        return string
    
    def __str__(self):
        string = "---------------\n"
        for i in range(self.shape[0]):
            string += str(self.values[i])+"\n"
        string += "shape:"+str((self.shape[0], self.shape[1]))
        return string
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return TensorPy([self.values[key]])
        elif isinstance(key, slice):
            keys = list(range(self.__len__()))[key]
            return TensorPy([self.values[i] for i in keys])
        elif isinstance(key, list) and isinstance(key[0], int):
            return TensorPy([self.values[i] for i in key])
        else:
            raise TypeError

    def is_contained(self, TensorPy t):
        if list(self.shape) != list(t.shape):
            print("Shape is not consistent")
            return None
        result = []
        for i in range(t.shape[0]):
            contained = True
            for j in range(t.shape[1]):
                if self.values[i][j] > t.values[i][j]:
                    contained = False
                    break
            result.append(contained)
        return result

    def conflict_test(self, TensorPy t):
        if list(self.shape) != list(t.shape):
            print("Shape is not consistent")
            return None
        result = []
        for i in range(t.shape[0]):
            conflict_indices = []
            for j in range(t.shape[1]):
                if self.values[i][j] > t.values[i][j]:
                    conflict_indices.append(j)
            result.append(conflict_indices)
        return result

    def sum(self):
        values_sum = []
        for i in range(self.shape[1]):
            column = []
            for j in range(self.shape[0]):
                column.append(self.values[j][i])
            values_sum.append(sum(column))
        return TensorPy([values_sum])

cdef from_ptr_to_tensor_py(ppc_index.Tensor* tensor):
    content = []
    for i in range(tensor.row_size):
        values = []
        for j in range(tensor.column_size):

            values.append(tensor.get_value(i, j))
        content.append(values)
    return TensorPy(content)


cdef class FeatureCounter:
    cdef bint enable_residual
    cdef int feature_length
    cdef int feature_num
    cdef ppc_index.feature_counter* counter
    cdef vector[vector[Label]] label_paths

    def __init__(self, enable_residual, label_paths, feature_type):
        self.feature_length = len(label_paths[0])
        self.feature_num = len(label_paths)
        self.enable_residual = enable_residual

        if isinstance(label_paths, list):
            for path in label_paths:
                self.label_paths.push_back(path)
        else:
            print("label_paths should be lists")
            return

        if feature_type == 'path':
            self.counter = new ppc_index.Path_counter(self.enable_residual, self.label_paths)
        elif feature_type == 'cycle':
            self.counter = new ppc_index.Cycle_counter(self.enable_residual, self.label_paths)
        else:
            print("feature type must be 'path' or 'cycle'")

    def __del__(self):
        del self.counter

    def count(self, graph_nx, level, thread_num=1):
        cdef ppc_index.Graph graph
        cdef vector[vector[Vertex]] adj
        cdef vector[Label] label_map
        cdef ppc_index.Tensor* tensor
        cdef ppc_index.Index_constructer constructor

        constructor = ppc_index.Index_constructer(self.counter)
        start_offset = 0
        for i in range(len(graph_nx.nodes())):
            label_map.push_back(graph_nx.nodes[i]['label'])
            tmp = []
            for n in graph_nx[i]:
                tmp.append(n)
            adj.push_back(tmp)
        graph = ppc_index.Graph(adj, label_map)
        if level == 'edge':
            tensor = constructor.count_features(graph, thread_num, 1)
        elif level == 'vertex':
            tensor = constructor.count_features(graph, thread_num, 0)
        result = from_ptr_to_tensor_py(tensor)
        # release tensor
        del tensor
        return result

    def build_index(self, graphs_nx, index_file_name, level, thread_num=1, max_features_per_batch=64):
        cdef ppc_index.Graph graph
        cdef vector[vector[Vertex]] adj
        cdef vector[Label] label_map
        cdef ppc_index.Tensor* tensor
        cdef string filename
        cdef int max_features_per_batch_c
        cdef int thread_num_c
        cdef int level_c
        cdef ppc_index.Index_constructer constructor

        if level == 'vertex':
            level_c = 0
        elif level == 'edge':
            level_c = 1
        constructor = ppc_index.Index_constructer(self.counter)
        filename = index_file_name.encode('UTF-8')
        thread_num_c = thread_num
        max_features_per_batch_c = max_features_per_batch
        construction_time = 0
        feature_num = self.counter.get_features().size()
        print(index_file_name+"_"+str(self.counter.get_feature_type())+"_"+str(level_c))
        id = 0
        for graph_nx in graphs_nx:
            adj.clear()
            label_map.clear()
            for i in range(len(graph_nx.nodes())):
                label_map.push_back(graph_nx.nodes[i]['label'])
                tmp = []
                for n in graph_nx[i]:
                    tmp.append(n)
                adj.push_back(tmp)
            graph = ppc_index.Graph(adj, label_map)
            start = time.time()
            if feature_num <= max_features_per_batch_c:
                constructor.construct_index_single_batch(graph, filename, thread_num_c, level_c)
            else:
                constructor.construct_index_in_batch(graph, filename, max_features_per_batch_c, thread_num_c, level_c)

            id += 1
            end = time.time() 
            construction_time += end-start
        
        return construction_time


cdef class IndexParser:
    cdef string filename
    cdef ppc_index.Index_manager manager

    def __init__(self, filename_):
        self.filename = filename_.encode('UTF-8')
        self.manager = ppc_index.Index_manager(self.filename)

    def load_graph_embeddings(self, graph_id):
        cdef Tensor* tensor

        tensor = self.manager.load_graph_tensor(graph_id)
        result = from_ptr_to_tensor_py(tensor)
        del tensor
        return result

    def load_vertex_embeddings(self, graph_id, vertex_id):
        cdef Tensor* tensor
        cdef vector[Value] vec
        
        self.manager.load_vertex_embedding(graph_id, vertex_id, vec)
        result = TensorPy([list(vec)])
        return result


def subgraph_retrieval(data_graphs, query_graphs, mode, edge_index_files, vertex_index_files, edge_counter_paras, vertex_counter_paras):
    cdef vector[Graph] data_graph_list
    cdef vector[Graph] query_graph_list
    cdef vector[feature_counter*] counter_edge_list
    cdef vector[feature_counter*] counter_vertex_list
    cdef vector[string] edge_index_file_list
    cdef vector[string] vertex_index_file_list
    cdef ppc_index.feature_counter* counter

    cdef vector[double] filtering_time
    cdef vector[int] filtered_results
    
    cdef vector[vector[Label]] label_paths
    cdef vector[vector[Vertex]] adj
    cdef vector[Label] label_map
    
    for graph_nx in data_graphs:
        adj.clear()
        label_map.clear()
        for i in range(len(graph_nx.nodes())):
            label_map.push_back(graph_nx.nodes[i]['label'])
            tmp = []
            for n in graph_nx[i]:
                tmp.append(n)
            adj.push_back(tmp)
        data_graph_list.push_back(ppc_index.Graph(adj, label_map))
    for graph_nx in query_graphs:
        adj.clear()
        label_map.clear()
        for i in range(len(graph_nx.nodes())):
            label_map.push_back(graph_nx.nodes[i]['label'])
            tmp = [] 
            for n in graph_nx[i]:
                tmp.append(n)
            adj.push_back(tmp)
        query_graph_list.push_back(ppc_index.Graph(adj, label_map))
    
    for file in edge_index_files:
        edge_index_file_list.push_back(file.encode('UTF-8'))
    for file in vertex_index_files:
        vertex_index_file_list.push_back(file.encode('UTF-8'))
    
    for para in edge_counter_paras:
        feature_file = para[-1]
        features = []
        with open(feature_file) as f:
            features = json.load(f)

        # convert the features to c++ forms
        label_paths.clear()
        for path in features:
            label_paths.push_back(path)

        if para[3] == 1:
            counter = new ppc_index.Cycle_counter((para[2]==1), features)
        else:
            counter = new ppc_index.Path_counter((para[2]==1), features)

        counter_edge_list.push_back(counter)
    
    for para in vertex_counter_paras:
        feature_file = para[-1]
        features = []
        with open(feature_file) as f:
            features = json.load(f)

        # convert the features to c++ forms
        label_paths.clear()
        for path in features:
            label_paths.push_back(path)

        if para[3] == 1:
            counter = new ppc_index.Cycle_counter((para[2]==1), features)
        else:
            counter = new ppc_index.Path_counter((para[2]==1), features)

        counter_vertex_list.push_back(counter)

    if mode == 'vc':
        if len(edge_index_files) == 0 and len(vertex_index_files) == 0:
            sub_containment_vc(data_graph_list, query_graph_list, filtered_results, filtering_time)
        else:
            sub_containment_vc_index(data_graph_list, query_graph_list, edge_index_file_list, vertex_index_file_list, counter_edge_list, counter_vertex_list, filtered_results, filtering_time)
    elif mode == 'graph':
        sub_containment_graph_level(data_graph_list, query_graph_list, edge_index_file_list, vertex_index_file_list, counter_edge_list, counter_vertex_list, filtered_results, filtering_time)

    for i in range(counter_edge_list.size()):
        del counter_edge_list[i]
    for i in range(counter_vertex_list.size()):
        del counter_vertex_list[i]

    # analyse the results
    return list(filtered_results), list(filtering_time)

def run_subgraph_enumeration(query_file_name, data_graph_nx, query_graphs, order, count_limit, edge_index_files, vertex_index_files, edge_counter_paras, vertex_counter_paras):
    cdef Graph data_graph
    cdef Graph query_graph
    cdef vector[feature_counter*] counter_edge_list
    cdef vector[feature_counter*] counter_vertex_list
    cdef vector[string] edge_index_file_list
    cdef vector[string] vertex_index_file_list
    cdef ppc_index.feature_counter* counter

    cdef vector[double] filtering_time
    cdef vector[int] filtered_results
    
    cdef vector[vector[Label]] label_paths
    cdef vector[vector[Vertex]] adj
    cdef vector[Label] label_map

    cdef ppc_index.Index_manager manager
    cdef vector[Tensor*] emb_vec_tmp;
    cdef Tensor* query_vertex_emb, *data_vertex_emb, *query_edge_emb, *data_edge_emb;
    cdef ppc_index.Index_constructer constructor

    cdef bint order_c = order
    cdef long count_limit_c = count_limit
    cdef long result_count
    cdef double enumeration_time, preprocessing_time
    
    adj.clear()
    label_map.clear()
    for i in range(len(data_graph_nx.nodes())):
        label_map.push_back(data_graph_nx.nodes[i]['label'])
        tmp = []
        for n in data_graph_nx[i]:
            tmp.append(n)
        adj.push_back(tmp)
    data_graph = ppc_index.Graph(adj, label_map)
    
    for file in edge_index_files:
        edge_index_file_list.push_back(file.encode('UTF-8'))
    for file in vertex_index_files:
        vertex_index_file_list.push_back(file.encode('UTF-8'))
    
    for para in edge_counter_paras:
        feature_file = para[-1]
        features = []
        with open(feature_file) as f:
            features = json.load(f)

        # convert the features to c++ forms
        label_paths.clear()
        for path in features:
            label_paths.push_back(path)

        if para[3] == 1:
            counter = new ppc_index.Cycle_counter((para[2]==1), features)
        else:
            counter = new ppc_index.Path_counter((para[2]==1), features)

        counter_edge_list.push_back(counter)
    
    for para in vertex_counter_paras:
        feature_file = para[-1]
        features = []
        with open(feature_file) as f:
            features = json.load(f)

        # convert the features to c++ forms
        label_paths.clear()
        for path in features:
            label_paths.push_back(path)

        if para[3] == 1:
            counter = new ppc_index.Cycle_counter((para[2]==1), features)
        else:
            counter = new ppc_index.Path_counter((para[2]==1), features)

        counter_vertex_list.push_back(counter)

    # extracting the data_emb
    emb_vec_tmp.clear()
    for file in edge_index_file_list:
        manager = ppc_index.Index_manager(file)
        emb_vec_tmp.push_back(manager.load_graph_tensor(0))
    data_edge_emb = ppc_index.merge_multi_Tensors(emb_vec_tmp)
    for emb in emb_vec_tmp:
        del emb
    emb_vec_tmp.clear()
    for file in vertex_index_file_list:
        manager = ppc_index.Index_manager(file)
        emb_vec_tmp.push_back(manager.load_graph_tensor(0))
    data_vertex_emb = ppc_index.merge_multi_Tensors(emb_vec_tmp)
    for emb in emb_vec_tmp:
        del emb
    data_graph.set_up_edge_id_map()

    result = {}

    # start enumertion
    for id, graph_nx in enumerate(query_graphs):
        result[id] = {}
        adj.clear()
        label_map.clear()
        for i in range(len(graph_nx.nodes())):
            label_map.push_back(graph_nx.nodes[i]['label'])
            tmp = [] 
            for n in graph_nx[i]:
                tmp.append(n)
            adj.push_back(tmp)
        
        query_graph = ppc_index.Graph(adj, label_map)
        
        # extracting the query_emb
        start_time = time.time()
        emb_vec_tmp.clear()
        for counter in counter_edge_list:
            constructor = ppc_index.Index_constructer(counter)
            emb_vec_tmp.push_back(constructor.count_features(query_graph, 1, 1))
        query_edge_emb = ppc_index.merge_multi_Tensors(emb_vec_tmp)
        for emb in emb_vec_tmp:
            del emb
        emb_vec_tmp.clear()
        for counter in counter_vertex_list:
            constructor = ppc_index.Index_constructer(counter)
            emb_vec_tmp.push_back(constructor.count_features(query_graph, 1, 0))
        query_vertex_emb = ppc_index.merge_multi_Tensors(emb_vec_tmp)
        for emb in emb_vec_tmp:
            del emb
        end_time = time.time()
        result[id]["query_indexing"] = end_time-start_time
        
        ppc_index.subgraph_enumeration(&data_graph, &query_graph, count_limit_c, result_count, enumeration_time, preprocessing_time, query_vertex_emb, data_vertex_emb, query_edge_emb, data_edge_emb, order_c)
        
        result[id]["result"] = result_count
        result[id]["preprocessing_time"] = preprocessing_time
        result[id]["enumeration_time"] = enumeration_time
        print("----------- {0}:{1} -------------".format(query_file_name, id))
        print(result[id])

    return result


cdef class FeatureSelector:
    cdef Feature_selector* selector

    def __init__(self, num_labels, feature_num, feature_length, enable_cycle, tmp_path):
        cdef string tmp_path_cpp = tmp_path.encode('UTF-8')
        cdef int num_labels_c = num_labels
        cdef int feature_num_c = feature_num
        cdef int feature_length_c = feature_length
        cdef bint enable_cycle_c = enable_cycle
        self.selector = new Feature_selector(num_labels_c, feature_num_c, feature_length_c, enable_cycle_c, tmp_path_cpp)

    def extract_for_singe_graph(self, samples, data_graph_nx, level, max_batch_size, thread_num):
        cdef vector[vector[Label]] features
        cdef vector[ppc_index.Graph] query_graphs
        cdef vector[vector[Vertex]] query_anchors
        cdef vector[vector[Vertex]] data_anchors
        cdef ppc_index.Graph data_graph

        cdef vector[vector[Label]] label_paths
        cdef vector[vector[Vertex]] adj
        cdef vector[Label] label_map

        cdef int level_c
        cdef int max_batch_size_c
        cdef int thread_num_c

        cdef Feature_selector f

        if level == 'vertex':
            level_c = 0
        elif level == 'edge':
            level_c = 1
        max_batch_size_c = max_batch_size
        thread_num_c = thread_num

        q_anchors = []
        d_anchors = []
        for sample in samples:
            adj.clear()
            label_map.clear()
            for i in range(len(sample[0].nodes())):
                label_map.push_back(sample[0].nodes[i]['label'])
                tmp = []
                for n in sample[0][i]:
                    tmp.append(n)
                adj.push_back(tmp)
            query_graphs.push_back(ppc_index.Graph(adj, label_map))
            q_anchors.append(sample[1][0])
            d_anchors.append([sample[1][1][i][0] for i in range(len(sample[1][1]))])
        adj.clear()
        label_map.clear()
        for i in range(len(data_graph_nx.nodes())):
            label_map.push_back(data_graph_nx.nodes[i]['label'])
            tmp = []
            for n in data_graph_nx[i]:
                tmp.append(n)
            adj.push_back(tmp)
        data_graph = ppc_index.Graph(adj, label_map)
        
        for a in q_anchors:
            query_anchors.push_back(a)
        for a in d_anchors:
            data_anchors.push_back(a)
        
        features = self.selector.extract_for_singe_graph(query_graphs, query_anchors, data_anchors, data_graph, level_c, max_batch_size_c, thread_num_c)

        return list(features)

    def __del__(self):
        del self.selector