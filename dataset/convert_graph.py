import argparse 
import os
import json
import ipdb
import torch
import struct
import time
import random


import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="query")
    parser.add_argument('--output', type=str, help="data")
    parser.add_argument('--name', type=str, help="name")
    args = parser.parse_args()
    return args

def parse_graph(filename):
    graph_list = []
    with open(filename) as f:
        G = None
        for line in f.readlines():
            if line[0] == '#':
                if G is not None:
                    graph_list.append(G)
                G = nx.Graph()
            elif line[0] == 'v':
                ele = line.strip("\n").split(" ")
                G.add_node(int(ele[1]), label=int(ele[2]))
            elif line[0] == 'e':
                ele = line.strip("\n").split(" ")
                G.add_edge(int(ele[1]), int(ele[2]))
        if len(G.nodes)>0:
            graph_list.append(G)
    return graph_list

def get_graph_count(graph_file):
    count = 0
    with open(graph_file) as f:
        for l in f.readlines():
            ele = int(l.split(" ")[-1])
            if l.startswith('#') and ele!=-1:
                count += 1
    return count

def dump_file_append(filename, graph_pair_list):
    base = 0
    if os.path.exists(filename):
        base = get_graph_count(filename)
    with open(filename, "a+") as f:
        for graph_pair in graph_pair_list:
            graph = graph_pair[0]
            f.write("# {0}\n".format(str(base)))
            for n in graph.nodes():
                f.write("v {0} {1}\n".format(n, graph.nodes[n]["label"]))
            for e in graph.edges():
                f.write("e {0} {1}\n".format(e[0], e[1]))
            for a_q, a_d in zip(graph_pair[1][0], graph_pair[1][1]):
                f.write("q {0}\n".format(a_q))
                f.write("d {0} {1}\n".format(a_d[0], a_d[1]))
            base += 1



args = parse_args()


graphs = parse_graph(args.input)
for id, graph in enumerate(graphs):
    output_file = os.path.join(args.output, "{0}_{1}.graph".format(args.name, id))
    with open(output_file, 'w') as f:
        f.write("t {0} {1}\n".format(len(graph.nodes()), len(graph.edges())))
        for i in range(len(graph.nodes())):
            f.write("v {0} {1} {2}\n".format(i, graph.nodes[i]['label'], graph.degree(i)))
        for i in range(len(graph.nodes())):
            for n in graph[i]:
                if i < n:
                    f.write("e {0} {1} 0\n".format(i, n))

