import argparse
import networkx as nx
import os
import ipdb
import torch
import numpy as np
import random

import sys
# sys.path.append("/home/yixin/jiezhonghe/project/NeuralSubgraphMatching/common")
# import sampler as sampler_cpp
sys.path.append("./query_generator")
import generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="data graph in unified format")
    parser.add_argument('--query_size', type=int, help="query size")
    parser.add_argument('--num_queries', type=int)
    # parser.add_argument('--enable_cycle', type=bool, default=False)
    parser.add_argument('--anchor', type=str, help="[vertex, edge]")
    parser.add_argument('--output', type=str, help="output directory")
    parser.add_argument('--pos_neg', type=str, default="neg", help="['pos', 'neg']")
    args = parser.parse_args()
    return args

def parse_all_graph(filename):
    graph_list = []
    with open(filename) as f:
        G = None
        for line in f.readlines():
            if line[0] == 't':
                if G is not None:
                    graph_list.append(G)
                G = nx.Graph()
            elif line[0] == 'v':
                ele = line.strip("\n").split(" ")
                G.add_node(int(ele[1]), label=int(ele[2]))
            elif line[0] == 'e':
                ele = line.strip("\n").split(" ")
                G.add_edge(int(ele[1]), int(ele[2]))
        if len(G.nodes()) > 0:
            graph_list.append(G)
    return graph_list

class generate_single_graph:
    def __init__(self, args):
        self.args = args
        self.graph = parse_all_graph(args.input)[0]

        self.isomorphism_checker = sampler_cpp.Sampler([])

        random.seed(0)

        self.label_dist = {}
        for n in self.graph.nodes():
            label = self.graph.nodes[n]['label']
            if label not in self.label_dist:
                self.label_dist[label] = []
            self.label_dist[label].append(n)

    def k_hop_neighbors(self, graph, hops, anchor):
        frontier = [(anchor, 0)]
        explored = set()
        while len(frontier) > 0:
            ele = frontier.pop(0)
            explored.add(ele[0])
            if ele[1] >= hops:
                continue
            for n in graph[ele[0]]:
                if n not in explored:
                    frontier.append((n, ele[1]+1))

        return list(explored)

    def random_walk_in_k_hop(self, graph, hops, anchor, size):
        frontier = [anchor]
        explored = set()
        while len(frontier) > 0 and len(explored)<size:
            idx = random.randint(0, len(frontier)-1)
            ele = frontier.pop(idx)
            distance = nx.shortest_path_length(graph, source=anchor, target=ele)
            if distance > hops:
                continue

            explored.add(ele)
            for n in graph[ele]:
                if n not in explored:
                    frontier.append(n)

        return list(explored)

    def check_nlf(self, query, anchor_query, graph, anchor_data):
        label_dist_query = {}
        label_dist_data = {}

        for n in query[anchor_query]:
            label = query.nodes[n]['label']
            if label not in label_dist_query:
                label_dist_query[label] = 0
            label_dist_query[label] += 1
        for n in graph[anchor_query]:
            label = graph.nodes[n]['label']
            if label not in label_dist_data:
                label_dist_data[label] = 0
            label_dist_data[label] += 1

        for l in label_dist_query:
            if l not in label_dist_data:
                return False
            if label_dist_query[l] > label_dist_data[l]:
                return False
        return True

    def generate_query(self, hops, query_size):
        while True:
            # choose a label evenly
            anchor_label = random.choice(list(self.label_dist.keys()))
            if len(self.label_dist[anchor_label]) < 2:
                continue
            
            # choose a data graph anchor
            anchor_data = random.choice(self.label_dist[anchor_label])

            # choose a query graph anchor
            anchor_query = anchor_data
            while anchor_query == anchor_data:
                anchor_query = random.choice(self.label_dist[anchor_label])

            # randomly generate a query graph
            query_nei = self.random_walk_in_k_hop(self.graph, hops, anchor_query, query_size)
            query_graph = nx.Graph(self.graph.subgraph(query_nei))

            # randomly remove some edges
            edges = list(query_graph.edges())
            to_delete = random.sample(list(query_graph.edges()), random.randint(0, len(query_graph.edges())//2))
            for e in to_delete:
                query_graph.remove_edge(e[0], e[1])
                if not nx.is_connected(query_graph):
                    query_graph.add_edge(e[0], e[1])

            # check by nlf
            if self.check_nlf(query_graph, anchor_query, self.graph, anchor_data) == False:
                continue
            
            data_nei = self.k_hop_neighbors(self.graph, hops, anchor_data)
            data_graph = self.graph.subgraph(data_nei)
            # ipdb.set_trace()
            # res = generator.check_iso_by_filtering(self.graph, anchor_data, query_graph, anchor_query, hops)
            if self.isomorphism_checker.check_sub_iso(data_graph, query_graph, [[anchor_data], [anchor_query]]):
                continue
            
            return (anchor_data, query_graph, anchor_query)
    
    def dump(self, generated_queries, filename):
        with open(filename, 'w') as f:
            for id, q in enumerate(generated_queries):
                f.write("# {0}\n".format(id))
                id_map = {}
                for v in q[1].nodes():
                    id_map[v] = len(id_map)
                    f.write("v {0} {1}\n".format(id_map[v], q[1].nodes[v]['label']))
                for e in q[1].edges():
                    f.write("e {0} {1}\n".format(id_map[e[0]], id_map[e[1]]))
                f.write("q {0}\n".format(id_map[q[2]]))
                f.write("d {0} 0\n".format(q[0])) # last 0 is the data graph id in original graph
            f.write("# -1\n")

def dump(generated_queries, filename):
    with open(filename, 'w') as f:
        for id, q in enumerate(generated_queries):
            f.write("# {0}\n".format(id))
            id_map = {}
            if len(q) == 4:
                query = q[0]
                query_anchor = q[1]
                data_id = q[2]
                data_anchor = q[3]
            else:
                query = q[0]
                query_anchor = q[1]
                query_anchor_u = q[2]
                data_id = q[3]
                data_anchor = q[4]
                data_anchor_v = q[5]
            for v in query.nodes():
                id_map[v] = len(id_map)
                f.write("v {0} {1}\n".format(id_map[v], query.nodes[v]['label']))
            for e in query.edges():
                f.write("e {0} {1}\n".format(id_map[e[0]], id_map[e[1]]))
            f.write("q {0}\n".format(id_map[query_anchor]))
            f.write("d {0} {1}\n".format(data_anchor, data_id)) # last 0 is the data graph id in original graph
            if len(q) > 4:
                f.write("q {0}\n".format(id_map[query_anchor_u]))
                f.write("d {0} {1}\n".format(data_anchor_v, data_id)) # last 0 is the data graph id in original graph
        f.write("# -1\n")

def dump_without_anchor(generated_queries, filename):
    with open(filename, 'w') as f:
        for id, q in enumerate(generated_queries):
            f.write("# {0}\n".format(id))
            id_map = {}
            for v in q.nodes():
                id_map[v] = len(id_map)
                f.write("v {0} {1}\n".format(id_map[v], q.nodes[v]['label']))
            for e in q.edges():
                f.write("e {0} {1}\n".format(id_map[e[0]], id_map[e[1]]))
        f.write("# -1\n")

# def dump(generated_queries, filename):
#     with open(filename, 'w') as f:
#         for id, q in enumerate(generated_queries):
#             f.write("# {0}\n".format(id))
#             id_map = {}
#             for v in q[1].nodes():
#                 f.write("v {0} {1}\n".format(v, q[1].nodes[v]['label']))
#             for e in q[1].edges():
#                 f.write("e {0} {1}\n".format(e[0], e[1]))
#             f.write("q {0}\n".format(q[2]))
#             f.write("d {0} 0\n".format(q[0])) # last 0 is the data graph id in original graph
#         f.write("# -1\n")

# if __name__ == '__main__':
#     args = parse_args()

#     single_graph_generator = generate_single_graph(args)
#     queries = []
#     for i in range(args.num_queries):
#         q = single_graph_generator.generate_query(4, args.query_size)
#         queries.append(q)
#         if i%10 == 0:
#             print("progress:"+str(i))
#     single_graph_generator.dump(queries, os.path.join(args.output, "query_{0}.gr".format(args.query_size)))

if __name__ == '__main__':
    args = parse_args()
    graphs = parse_all_graph(args.input)

    gen = generator.QueryGenerator(graphs)

    result_name = "query_{1}_{0}.gr".format(args.query_size, args.anchor)
    other = 0
    while os.path.exists(os.path.join(args.output, result_name)):
        result_name = "query_{2}_{0}_({1}).gr".format(args.query_size, other, args.anchor)
        other += 1
    
    is_vertex = False
    if args.anchor == 'edge':
        is_vertex = False
    else:
        is_vertex = True
    enable_cycle = False
    
    if args.pos_neg == "neg":
        queries = []
        print("done loading graph")
        for size in [8, 16, 24, 32, 40]:
            for i in range(args.num_queries):
                enable_cycle = not enable_cycle
                if len(graphs) > 1:
                    r = gen.generate_neg_query(4, size, enable_cycle, is_vertex)
                else:
                    r = gen.generate_query(4, size, enable_cycle, is_vertex)
                queries.append(r)
                print("finish:"+str(size)+":"+str(i))
            result_name = "query_{1}_{0}.gr".format(size, args.anchor)
            dump(queries, os.path.join(args.output, result_name))
        # for i in range(args.num_queries):
        #     enable_cycle = not enable_cycle
        #     if len(graphs) > 1:
        #         r = gen.generate_neg_query(4, args.query_size, enable_cycle, is_vertex)
        #     else:
        #         r = gen.generate_query(4, args.query_size, enable_cycle, is_vertex)
        #     queries.append(r)
        #     print("finish:"+str(i))
        # dump(queries, os.path.join(args.output, result_name))
    # else:
    #     queries = []
    #     print("done loading graph")
        
    #     for i in range(args.num_queries):
    #         r = gen.generate_pos_query(4, args.query_size, args.enable_cycle)
    #         queries.append(r)
    #         print("finish:"+str(i))
        # dump_without_anchor(queries, os.path.join(args.output, result_name))