import argparse 
import os
import json
import ipdb
import torch
import struct
import time
import random


import networkx as nx

import sys
sys.path.append("./index")
from ppc_index import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help="query")
    parser.add_argument('--input_data', type=str, help="data")
    # [] means vanilla vc-framework
    parser.add_argument('--para', type=str, default="['64-4-1-1-1']") # feature_num feature_length residual feature_type(0 path, 1 cycle) level(0 vertex 1 edges)
    parser.add_argument("--mode", type=str, help="vc or graph") # vc mean vc framework using the edge, vertex support, or just vc if para=[], graph means graph level index by summing the supports of the indices from para
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
        if len(G.nodes()) > 0:
            graph_list.append(G)
    return graph_list

def parse_samples(filename):
    graph_list = []
    anchors = [[], []]
    with open(filename) as f:
        G = None
        for line in f.readlines():
            if line[0] == '#':
                if G is not None:
                    graph_list.append((G, [anchors[0][:], anchors[1][:]]))
                G = nx.Graph()
                anchors[0].clear()
                anchors[1].clear()
            elif line[0] == 'v':
                ele = line.strip("\n").split(" ")
                G.add_node(int(ele[1]), label=int(ele[2]))
            elif line[0] == 'e':
                ele = line.strip("\n").split(" ")
                G.add_edge(int(ele[1]), int(ele[2]))
            elif line[0] == 'q':
                ele = line.strip("\n").split(" ")
                anchors[0].append(int(ele[1]))
            elif line[0] == 'd':
                ele = line.strip("\n").split(" ")
                anchors[1].append((int(ele[1]), int(ele[2])))
        if len(G.nodes()) > 0:
            graph_list.append((G, [anchors[0][:], anchors[1][:]]))
    return graph_list

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

# def run_exp(parsed_args, index_path):
#     index_file_list = []
#     feature_file_list = []

#     paras = []
#     lines = eval(parsed_args.para)
#     for line in lines:
#         p = line.split('-')
#         paras.append([int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])])

#     index_files = []
#     for para in paras:
#         feature_type = 'path' if paras[3]==0 else 'cycle'
#         level = "edge" if paras[4] == 1 else "vertex"

#         name = parsed_args.input_data.split("/")[-1]+"_"+str(paras[0])+"_"+str(paras[1])+"_"+str(paras[2])+"_"+feature_type+"_"+level
#         level = "edge" if paras[4] == 1 else "vertex"
#         index_name = os.path.join(index_path, name+".index")
#         index_files.append(index_name)
#         if not os.path.exists(index_name):
#             print("index file:"+index_name+" does not exists!")
#             return
    
#     # start runing experiments
#     precisions = []
#     filtering_times = []
#     results = {}
#     data_graphs = parse_graph(parsed_args.input_data)
#     for file in os.listdir(parsed_args.input_query):
#         if file.endswith(".json"):
#             continue

#         query_graphs = parse_graph(os.path.join(parsed_args.input_query, file))
#         mode_code = 0 if parsed_args.mode == 'cv' else: 1 # 0 represents cv, 1 represents graph

#         filtering_results, filtering_time = subgraph_retrieval(index_files, mode_code, data_graphs, query_graphs, paras)

#     #     results_filtering, filtering_time = subgraph_retrieving(model_para_list, data_graphs, query_graphs, index_files)
#     #     # results_filtering, filtering_time = subgraph_retrieving_sum(model_para_list, data_graphs, query_graphs, index_files)
#     #     print(results_filtering)
#     #     print(filtering_time)
#     #     for qid in range(len(query_graphs)):
#     #         precisions.append(float(results[str(qid)]['number_of_matched_graphs'])/results_filtering[qid])
#     #         print("precision:{0}/{1}={2}".format(float(results[str(qid)]['number_of_matched_graphs']), results_filtering[qid], float(results[str(qid)]['number_of_matched_graphs'])/results_filtering[qid]))

#     #     filtering_times.append(filtering_time/len(query_graphs))

#     # print("average precision:"+str(sum(precisions)/float(len(precisions))))
#     # print("average filtering time:"+str(sum(filtering_times)/float(len(filtering_times))))


def parse_parameters(paras, index_path, parsed_args):
    index_files = []
    counter_paras = []
    for para in paras:
        feature_type = 'path' if para[3]==0 else 'cycle'
        level = "edge" if para[4] == 1 else "vertex"

        name = parsed_args.input_data.split("/")[-1]+"_"+str(para[0])+"_"+str(para[1])+"_"+str(para[2])+"_"+feature_type+"_"+level
        level = "edge" if para[4] == 1 else "vertex"
        index_name = os.path.join(index_path, name+".index")
        index_features = os.path.join(index_path, name+".features")
        index_files.append(index_name)
        counter_paras.append(para+[index_features])
        if not os.path.exists(index_name):
            print("index file:"+index_name+" does not exists!")
            return None
            
    return index_files, counter_paras

def run_exp(parsed_args, index_path):
    index_file_list = []
    feature_file_list = []

    lines = eval(parsed_args.para)
    edge_paras = []
    vertex_paras = []
    for line in lines:
        p = line.split('-')
        if int(p[4]) == 0:
            vertex_paras.append([int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])])
        else:
            edge_paras.append([int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])])

    edge_index_files = []
    vertex_index_file = []
    edge_counter_paras = []
    vertex_counter_paras = []
    # ipdb.set_trace()
    edge_index_files, edge_counter_paras = parse_parameters(edge_paras, index_path, parsed_args)
    vertex_index_file, vertex_counter_paras = parse_parameters(vertex_paras, index_path, parsed_args)

    precisions = []
    filtering_times = []
    global_results = {}
    data_graphs = parse_graph(parsed_args.input_data)
    for file in os.listdir(parsed_args.queries):
        if file.endswith(".json"):
            continue
        global_results[file] = {}
        ground_result = {}
        with open(os.path.join(parsed_args.queries, file+".json")) as f:
            ground_result = json.load(f)

        query_graphs = parse_graph(os.path.join(parsed_args.queries, file))
        
        filtering_results, filtering_time = subgraph_retrieval(data_graphs, query_graphs, args.mode, edge_index_files, vertex_index_file, edge_counter_paras, vertex_counter_paras)
        print(file)
        print(filtering_results)
        # analyse the results
        for id in range(len(query_graphs)):
            global_results[file][id] = {}
            precisions.append(ground_result[str(id)]["number_of_matched_graphs"]/float(filtering_results[id]))
            print("--------------")
            print("precision:{0}/{1}={2}".format(ground_result[str(id)]["number_of_matched_graphs"], filtering_results[id], ground_result[str(id)]["number_of_matched_graphs"]/float(filtering_results[id])))
            filtering_times.append(filtering_time[id])
            print("filtering time:{0}".format(filtering_time[id]))

            global_results[file][id]["candidates_after_filtering"] = filtering_results[id]
            global_results[file][id]["number_of_matched_graphs"] = ground_result[str(id)]["number_of_matched_graphs"]
            global_results[file][id]["filtering_time"] = filtering_time[id]
    print("*"*20)
    print("average precision:{0}".format(sum(precisions)/float(len(precisions))))
    print("average filtering time:{0}".format(sum(filtering_times)/float(len(filtering_times))))

    global_results["average_precision"] = sum(precisions)/float(len(precisions))
    global_results["average filtering time"] = sum(filtering_times)/float(len(filtering_times))

    result_file_name = parsed_args.input_data.split("/")[-1]+"_"+parsed_args.mode+"_"+"_".join(lines)+".json"
    with open(os.path.join(".result", result_file_name), 'w') as f:
        json.dump(global_results, f)

if __name__ == '__main__':
    args = parse_args()
    default_index_path = ".index"

    run_exp(args, default_index_path)