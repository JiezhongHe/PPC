#include <iostream>
#include "graph.hpp"
#include "embedding.hpp"
#include "cycle_counting.hpp"
#include "path_counting.hpp"
#include "subgraph_enumeration.hpp"
#include "subgraph_retrieval.hpp"
#include "feature_selector.hpp"
#include "index.hpp"
#include "utils.hpp"
#include "core_decomposition.hpp"

using namespace std;

void load_sample_list(string filename, vector<Graph>& result, vector<vector<Vertex>>& query_anchors, vector<vector<Vertex>>& data_anchors){
    ifstream in(filename);

    string line;
    unordered_map<Vertex, Label> label_map_unordered;
    unordered_map<Vertex, unordered_set<Vertex>> adj_unordered;
    vector<Label> label_map;
    vector<unordered_set<Vertex>> adj;
    vector<Vertex> q_anchors;
    vector<Vertex> d_anchors;
    while(getline(in, line)){
        if(line.c_str()[0] == '#'){
            if(label_map_unordered.size()>0){
                label_map.resize(label_map_unordered.size());
                for(auto v : label_map_unordered){
                    label_map[v.first] = v.second;
                }
                adj.resize(label_map_unordered.size());
                for(auto v : adj_unordered){
                    adj[v.first] = v.second;
                }
                Graph graph(adj, label_map);
                result.push_back(graph);
                data_anchors.push_back(d_anchors);
                query_anchors.push_back(q_anchors);
                d_anchors.clear();
                q_anchors.clear();
                label_map.clear();
                adj.clear();
            }
            label_map_unordered.clear();
            adj_unordered.clear();
        }else if(line.c_str()[0] == 'v'){
            stringstream ss(line);
            Vertex v, n;
            char c;
            ss>>c>>v>>n;
            label_map_unordered[v] = n;
        }else if(line.c_str()[0] == 'e'){
            stringstream ss(line);
            Vertex v, n;
            char c;
            ss>>c>>v>>n;
            if(adj_unordered.find(v) == adj_unordered.end()){
                adj_unordered.insert({v, {n}});
            }else{
                adj_unordered[v].insert(n);
            }
            if(adj_unordered.find(n) == adj_unordered.end()){
                adj_unordered.insert({n, {v}});
            }else{
                adj_unordered[n].insert(v);
            }
        }else if(line.c_str()[0] == 'q'){
            stringstream ss(line);
            Vertex u;
            char c;
            ss>>c>>u;
            q_anchors.push_back(u);
        }else if(line.c_str()[0] == 'd'){
            stringstream ss(line);
            Vertex v;
            char c;
            ss>>c>>v;
            d_anchors.push_back(v);
        }
    }
    if(label_map_unordered.size()>0){
        label_map.resize(label_map_unordered.size());
        for(auto v : label_map_unordered){
            label_map[v.first] = v.second;
        }
        adj.resize(label_map_unordered.size());
        for(auto v : adj_unordered){
            adj[v.first] = v.second;
        }
        Graph graph(adj, label_map);
        result.push_back(graph);
        data_anchors.push_back(d_anchors);
        query_anchors.push_back(q_anchors);
    }
}

// int main(){
    // vector<Vertex> label_map = {0, 1, 1, 2};
    // vector<vector<Vertex> > adj = {
    //     {1, 2},
    //     {0, 2, 3},
    //     {0, 1, 3},
    //     {1, 2}
    // };
    // vector<vector<Label> > label_paths = {
    //     {0, 1},
    //     {2, 1}
    // };
    // Graph graph(adj, label_map);
    // Cycle_counter counter(false, label_paths);
    // Tensor** result;
    // int result_size;
    // graph.construct_edge_common_neighbor();
    // counter.count_for_edges(graph, result, result_size);

    // vector<Vertex> label_map = {0, 1, 2, 2, 1};
    // vector<vector<Vertex> > adj = {
    //     {1, 4},
    //     {0, 2, 3, 4},
    //     {1, 3},
    //     {1, 2, 4},
    //     {0, 1, 3}
    // };
    // vector<vector<Label> > label_paths = {
    //     {0, 1, 2},
    //     {1, 1, 2},
    //     {2, 1, 2}
    // };

    // // vector<vector<Label> > label_paths;
    // // label_paths.resize(70);
    // // for(int i=0;i<70;++i){
    // //     label_paths[i].push_back(i);
    // //     label_paths[i].push_back(i);
    // // }

    // // // // vector<vector<Label> > label_paths = {{20, 36, 13, 26}, {13, 26, 6, 2}, {20, 36, 31, 13}, {9, 13, 17, 23}, {21, 26, 16, 16}, {27, 34, 27, 27}, {8, 20, 36, 6}, {21, 23, 23, 16}, {27, 34, 27, 30}, {30, 30, 30, 30}, {12, 14, 33, 8}, {6, 31, 31, 20}, {20, 36, 13, 36}, {16, 20, 2, 26}, {16, 20, 2, 36}, {12, 33, 33, 34}, {17, 27, 15, 11}, {13, 26, 6, 20}, {21, 21, 20, 1}, {12, 33, 34, 22}, {27, 34, 27, 23}, {2, 14, 20, 26}, {1, 32, 29, 3}, {2, 34, 20, 0}, {2, 34, 20, 13}, {9, 26, 17, 17}, {13, 26, 6, 6}, {20, 36, 31, 2}, {21, 26, 16, 11}, {14, 18, 25, 7}, {21, 21, 16, 11}, {4, 37, 32, 21}, {21, 23, 21, 1}, {21, 23, 21, 21}, {0, 22, 35, 38}, {8, 36, 26, 13}, {23, 35, 21, 37}, {23, 27, 33, 1}, {27, 34, 27, 17}, {34, 39, 28, 2}, {30, 30, 30, 31}, {1, 2, 13, 1}, {1, 22, 2, 16}, {2, 16, 13, 13}, {1, 2, 10, 6}, {8, 18, 23, 39}, {4, 9, 7, 15}, {20, 36, 13, 6}, {8, 8, 23, 31}, {8, 20, 25, 39}, {20, 36, 13, 33}, {16, 20, 2, 7}, {21, 23, 1, 23}, {16, 20, 2, 18}, {21, 23, 1, 27}, {8, 30, 0, 36}, {21, 21, 16, 20}, {9, 13, 14, 32}, {9, 13, 14, 33}, {9, 13, 30, 1}, {9, 13, 30, 2}, {9, 13, 30, 4}, {21, 23, 16, 35}, {9, 13, 30, 13}, {9, 13, 30, 16}, {9, 13, 30, 21}, {9, 13, 30, 27}, {16, 31, 20, 26}, {2, 34, 20, 38}, {20, 33, 26, 6}, {20, 33, 26, 13}, {16, 31, 29, 26}, {9, 13, 17, 14}, {17, 27, 15, 32}, {9, 26, 17, 1}, {17, 27, 22, 15}, {17, 27, 22, 17}, {9, 26, 17, 23}, {17, 27, 22, 22}, {17, 27, 22, 23}, {13, 26, 6, 30}, {20, 36, 31, 6}, {20, 36, 31, 11}, {20, 36, 31, 26}, {20, 36, 31, 22}, {20, 36, 31, 31}, {13, 23, 30, 17}, {13, 23, 30, 35}, {14, 18, 21, 1}, {14, 18, 21, 8}, {21, 21, 0, 3}, {8, 20, 36, 1}, {8, 20, 36, 8}, {21, 21, 16, 1}, {8, 20, 36, 36}, {21, 21, 16, 23}, {6, 26, 23, 10}, {6, 26, 23, 14}, {6, 23, 0, 2}, {20, 31, 13, 13}, {12, 33, 34, 19}, {4, 37, 32, 11}, {34, 39, 28, 37}, {4, 37, 32, 19}, {11, 38, 37, 0}, {11, 38, 37, 7}, {11, 38, 37, 11}, {11, 38, 37, 13}, {17, 27, 27, 9}, {8, 20, 33, 26}, {17, 27, 27, 11}, {17, 27, 27, 15}, {17, 27, 27, 28}, {20, 31, 6, 16}, {8, 20, 39, 36}, {20, 31, 6, 18}, {21, 23, 21, 0}, {22, 34, 33, 30}, {21, 23, 21, 6}, {0, 22, 35, 1}, {0, 22, 35, 7}, {0, 22, 35, 11}, {34, 39, 3, 14}, {21, 23, 21, 33}, {34, 39, 3, 32}, {34, 39, 3, 33}, {8, 36, 26, 22}, {23, 35, 21, 15}};
    // vector<vector<Label> > label_paths = {{20, 36, 13}, {13, 26, 6}, {20, 36, 31}, {9, 13, 17}, {21, 26, 16}, {27, 34, 27}, {8, 20, 36}, {21, 23, 23}, {27, 34, 27}, {30, 30, 30}, {12, 14, 33}, {6, 31, 31}, {20, 36, 13}, {16, 20, 2}, {16, 20, 2}, {12, 33, 33}, {17, 27, 15}, {13, 26, 6}, {21, 21, 20}, {12, 33, 34}, {27, 34, 27}, {2, 14, 20}, {1, 32, 29}, {2, 34, 20}, {2, 34, 20}, {9, 26, 17}, {13, 26, 6}, {20, 36, 31}, {21, 26, 16}, {14, 18, 25}, {21, 21, 16}, {4, 37, 32}, {21, 23, 21}, {21, 23, 21}, {0, 22, 35}, {8, 36, 26}, {23, 35, 21}, {23, 27, 33}, {27, 34, 27}, {34, 39, 28}, {30, 30, 30}, {1, 2, 13}, {1, 22, 2}, {2, 16, 13}, {1, 2, 10}, {8, 18, 23}, {4, 9, 7}, {20, 36, 13}, {8, 8, 23}, {8, 20, 25}, {20, 36, 13}, {16, 20, 2}, {21, 23, 1}, {16, 20, 2}, {21, 23, 1}, {8, 30, 0}, {21, 21, 16}, {9, 13, 14}, {9, 13, 14}, {9, 13, 30}, {9, 13, 30}, {9, 13, 30}, {21, 23, 16}, {9, 13, 30}};
    // vector<vector<Label>> label_paths = {{20, 36, 13, 26}, {13, 26, 6, 2}, {20, 36, 31, 13}, {9, 13, 17, 23}, {21, 26, 16, 16}, {27, 34, 27, 27}, {8, 20, 36, 6}, {21, 23, 23, 16}, {27, 34, 27, 30}, {30, 30, 30, 30}, {12, 14, 33, 8}, {6, 31, 31, 20}, {20, 36, 13, 36}, {16, 20, 2, 26}, {16, 20, 2, 36}, {12, 33, 33, 34}, {17, 27, 15, 11}, {13, 26, 6, 20}, {21, 21, 20, 1}, {12, 33, 34, 22}, {27, 34, 27, 23}, {2, 14, 20, 26}, {1, 32, 29, 3}, {2, 34, 20, 0}, {2, 34, 20, 13}, {9, 26, 17, 17}, {13, 26, 6, 6}, {20, 36, 31, 2}, {21, 26, 16, 11}, {14, 18, 25, 7}, {21, 21, 16, 11}, {4, 37, 32, 21}, {21, 23, 21, 1}, {21, 23, 21, 21}, {0, 22, 35, 38}, {8, 36, 26, 13}, {23, 35, 21, 37}, {23, 27, 33, 1}, {27, 34, 27, 17}, {34, 39, 28, 2}, {30, 30, 30, 31}, {1, 2, 13, 1}, {1, 22, 2, 16}, {2, 16, 13, 13}, {1, 2, 10, 6}, {8, 18, 23, 39}, {4, 9, 7, 15}, {20, 36, 13, 6}, {8, 8, 23, 31}, {8, 20, 25, 39}, {20, 36, 13, 33}, {16, 20, 2, 7}, {21, 23, 1, 23}, {16, 20, 2, 18}, {21, 23, 1, 27}, {8, 30, 0, 36}, {21, 21, 16, 20}, {9, 13, 14, 32}, {9, 13, 14, 33}, {9, 13, 30, 1}, {9, 13, 30, 2}, {9, 13, 30, 4}, {21, 23, 16, 35}, {9, 13, 30, 13}, {9, 13, 30, 16}, {9, 13, 30, 21}, {9, 13, 30, 27}, {16, 31, 20, 26}, {2, 34, 20, 38}, {20, 33, 26, 6}, {20, 33, 26, 13}, {16, 31, 29, 26}, {9, 13, 17, 14}, {17, 27, 15, 32}, {9, 26, 17, 1}, {17, 27, 22, 15}, {17, 27, 22, 17}, {9, 26, 17, 23}, {17, 27, 22, 22}, {17, 27, 22, 23}, {13, 26, 6, 30}, {20, 36, 31, 6}, {20, 36, 31, 11}, {20, 36, 31, 26}, {20, 36, 31, 22}, {20, 36, 31, 31}, {13, 23, 30, 17}, {13, 23, 30, 35}, {14, 18, 21, 1}, {14, 18, 21, 8}, {21, 21, 0, 3}, {8, 20, 36, 1}, {8, 20, 36, 8}, {21, 21, 16, 1}, {8, 20, 36, 36}, {21, 21, 16, 23}, {6, 26, 23, 10}, {6, 26, 23, 14}, {6, 23, 0, 2}, {20, 31, 13, 13}, {12, 33, 34, 19}, {4, 37, 32, 11}, {34, 39, 28, 37}, {4, 37, 32, 19}, {11, 38, 37, 0}, {11, 38, 37, 7}, {11, 38, 37, 11}, {11, 38, 37, 13}, {17, 27, 27, 9}, {8, 20, 33, 26}, {17, 27, 27, 11}, {17, 27, 27, 15}, {17, 27, 27, 28}, {20, 31, 6, 16}, {8, 20, 39, 36}, {20, 31, 6, 18}, {21, 23, 21, 0}, {22, 34, 33, 30}, {21, 23, 21, 6}, {0, 22, 35, 1}, {0, 22, 35, 7}, {0, 22, 35, 11}, {34, 39, 3, 14}, {21, 23, 21, 33}, {34, 39, 3, 32}, {34, 39, 3, 33}, {8, 36, 26, 22}, {23, 35, 21, 15}};
    // Graph graph(adj, label_map);
    // struct timeval start_t, end_t;
    // gettimeofday(&start_t, NULL);
    // Graph graph("../../../raw_dataset/Dataset/InMem/eu2005/data_graph/eu2005.gr");
    // // Graph graph("../../../raw_dataset/Dataset/InMem/yeast/data_graph/yeast.gr");
    // // cout<<"done loading graphs"<<endl;


    
    // // graph.construct_edge_common_neighbor();
    // // common_edge_neighbor_multi_threads(&graph, 8);
    // Cycle_counter counter(false, label_paths);
    // Index_constructer index(&counter);
    // index.construct_index_in_batch(graph, "test", 32, 5, 1);
    // index.construct_index_single_batch(graph, "tt", 1, 0);
    // Tensor* results = index.count_features(graph, 2, 0);
    // cout<<results->to_string()<<endl;
    // results = index.count_features(graph, 1, 1);
    // cout<<results->to_string()<<endl;
    // for(auto t:*results){
    //     cout<<t->to_string()<<endl;
    // }
    // results = counter.count_for_edges(graph);
    // for(auto t:*results){
    //     cout<<t->to_string()<<endl;
    // }
    // cout<<(*result)[0]->to_string()<<endl;
    // Index_constructer index(&counter);
    // index.construct_index_in_batch(graph, "batch_8", 2, 8, 1);
    // Index_manager manager("batch_8");
    // Tensor* result_8 = manager.load_graph_tensor(0);
    // cout<<result_8->to_string()<<endl;
    // Tensor* r = index.count_features(graph, 8, 1);
    // cout<<r->to_string()<<endl;

    // index.construct_index_in_batch(graph, "batch_8", 32, 8, 1);
    // index.construct_index_single_batch(graph, "test_8", 8, 1);
    // index.construct_index_single_batch(graph, "test_1", 1, 1);

    


    // Tensor* result = index.count_features(graph, 1, 0);
    // cout<<result->to_string()<<endl;
    // result = index.count_features(graph, 1, 1);
    // cout<<result->to_string()<<endl;
    // result = index.count_features(graph, 2, 1);
    // cout<<result->to_string()<<endl;
    
    // gettimeofday(&end_t, NULL);
    // cout<<"loading time:"<<get_time(start_t, end_t)<<endl;
    // gettimeofday(&start_t, NULL);
    // index.construct_index_in_batch(graph, "index_batched", 32, 8, 1);
    // index.construct_index_single_batch(graph, "index_vertex_8", 8, 0);
    // gettimeofday(&end_t, NULL);
    // cout<<"global finish time:"<<get_time(start_t, end_t)<<endl;
    // cout<<result->to_string()<<endl;
    // // // // Tensor* result = counter.count_for_edges(graph);
    // // // // // cout<<result->to_string()<<endl;
    // Index_manager manager("index");
    // // // manager.dump_tensor(result);
    // cout<<manager.load_graph_tensor(1)->to_string();
    // vector<Value> vec;
    // manager.load_vertex_embedding(0, 2, vec);
    // cout<<vector_to_string(vec);
    // vec.clear();
    // Index_manager manager1("index-residual");
    // manager1.load_vertex_embedding(0, 2, vec);
    // cout<<vector_to_string(vec);

    // Index_manager manager("batch_8");
    // Index_manager manager1("test_1");
    // bool info = true;
    // Tensor* result_8 = manager.load_graph_tensor(0);
    // Tensor* result_4 = manager.load_graph_tensor(0);
    // for(int i=0;i<result_8->row_size;++i){
    //     if(info == true){
    //         bool is_all_zero = true;
    //         for(int j=0;j<result_8->column_size;++j){
    //             if(result_8->content[i][j] != 0){
    //                 is_all_zero = false;
    //                 break;
    //             }
    //         }
    //         if(is_all_zero == false){
    //             cout<<"test1:{";
    //             for(int j=0;j<result_8->column_size;++j){
    //                 cout<<result_8->content[i][j]<<",";
    //             }
    //             cout<<"}"<<endl;
    //             cout<<"test2:{";
    //             for(int j=0;j<result_4->column_size;++j){
    //                 cout<<result_4->content[i][j]<<",";
    //             }
    //             cout<<"}"<<endl;
    //             info = false;
    //         }
    //     }
    //     Value* v8 = result_8->content[i];
    //     Value* v4 = result_4->content[i];
    //     for(int j=0;j<result_8->column_size;++j){
    //         if(v8[j] != v4[j]){
    //             cout<<"row:"<<i<<" are not the same"<<endl;
    //             return 0;
    //         }
    //     }
    // }
    // cout<<"the tensors are the same"<<endl;

    // struct timeval start_t, end_t;
    // gettimeofday(&start_t, NULL);
    // vector<string> tt = {"../test_index-8-32_0_edge", "../test_index-8-32_1_edge"};
    // merge_multi_index_files(tt, "test-10000");
    // gettimeofday(&end_t, NULL);
    // cout<<"finish time:"<<get_time(start_t, end_t)<<endl;

    // Graph graph("./test_data");
    // Graph query_graph("test_query");
    // candidate_auxiliary aux(&(graph), &(query_graph), NULL, NULL, NULL, NULL, false);
    // // vector<vector<Label>> path_features = load_label_path("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_64_4_1_path_edge.features");
    // // Path_counter counter = Path_counter(true, path_features);
    // // Index_constructer constructor(&counter);
    // // constructor.construct_index_single_batch(graph, "test", 1, 1);


    // test subgraph retrieval
    // vector<Graph> data_graphs;
    // vector<Graph> query_graphs;
    // load_graph_list("../../../raw_dataset/Dataset/GRAPES/ppigo/db/ppigo.gr", data_graphs);
    // load_graph_list("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/retrieval/ppigo/queries/query_vertex_24.gr", query_graphs);
    // vector<string> edge_list = {"/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_cycle_edge.index", "/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_path_edge.index"};
    // vector<string> vertex_list = {"/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_cycle_vertex.index", "/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_path_vertex.index"};
    
    // vector<vector<Label>> c_e_f = load_label_path("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_cycle_edge.features");
    // vector<vector<Label>> p_e_f = load_label_path("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_path_edge.features");
    // vector<vector<Label>> c_v_f = load_label_path("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_cycle_vertex.features");
    // vector<vector<Label>> p_v_f = load_label_path("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_path_vertex.features");

    // Cycle_counter* c_e = new Cycle_counter(true, c_e_f);
    // Path_counter* p_e = new Path_counter(true, p_e_f);
    // Cycle_counter* c_v = new Cycle_counter(true, c_v_f);
    // Path_counter* p_v = new Path_counter(true, p_v_f);

    // vector<feature_counter*> counter_edge_list = {c_e, p_e};
    // vector<feature_counter*> counter_vertex_list = {c_v, p_v};
    // vector<int> filtered_results;
    // vector<double> filtering_time;

    // // Index_constructer con(counter_vertex_list[1]);
    // // for(int i=0;i<data_graphs.size();++i){
    // //     con.construct_index_single_batch(data_graphs[i], "test", 1, 0);
    // // }
    // // Tensor* t_q = con.construct_index_in_batch(query_graphs[0], 1, 0);
    // // con.construct_index_single_batch(data_graphs[0], "test", 1, 0);
    // // Index_manager m("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/retrieval/.index/ppigo.gr_128_4_1_path_vertex.index");
    // // Tensor* t_d = m.load_graph_tensor(0);
    // // vector<Tensor*> t = m.load_all_graphs();

    // sub_containment_graph_level(data_graphs, query_graphs, 
    //     edge_list, vertex_list, 
    //     counter_edge_list, counter_vertex_list,
    //     filtered_results, filtering_time
    // );
    // for(auto r : filtered_results){
    //     cout<<r<<" ";
    // }
    // cout<<endl;
    // return 0;


    // test feature selector
    // Graph data_graph("../../../raw_dataset/Dataset/InMem/eu2005/data_graph/eu2005.gr");
    // vector<Graph> query_graphs;
    // vector<vector<Vertex>> query_anchors;
    // vector<vector<Vertex>> data_anchors;
    // // scan the number of labels
    // Label max_label = 0;
    // for(Vertex v=0;v<data_graph.label_map.size();++v){
    //     if(data_graph.label_map[v] > max_label){
    //         max_label = data_graph.label_map[v];
    //     }
    // }
    // load_sample_list("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/eu2005/feature_finding/edge/query_edge_8.gr", query_graphs, query_anchors, data_anchors);
    // load_sample_list("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/eu2005/feature_finding/edge/query_edge_16.gr", query_graphs, query_anchors, data_anchors);
    // load_sample_list("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/eu2005/feature_finding/edge/query_edge_24.gr", query_graphs, query_anchors, data_anchors);
    // load_sample_list("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/eu2005/feature_finding/edge/query_edge_32.gr", query_graphs, query_anchors, data_anchors);
    // load_sample_list("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/eu2005/feature_finding/edge/query_edge_40.gr", query_graphs, query_anchors, data_anchors);
    
    // Feature_selector fs(max_label+1, 128, 4, true, "../../run_exp/enumeration/.tmp/");
    // vector<vector<Label>> features = fs.extract_for_singe_graph(query_graphs, query_anchors, data_anchors, data_graph, 1, 32, 5);

    // dump_features(features, "eu2005.features");
    // return 0;
    // // vector<Graph> sample_queries = {query_graphs[0], query_graphs[1]};
    // // vector<vector<Vertex>> query_anchors = {{4,5},{4,5}};
    // // vector<vector<Vertex>> data_anchors = {{164,593}, {240,2384}};
    // // samples.push_back(tuple<Graph, vector<Vertex>, vector<Vertex>>(query_graphs[0], {4,5}, {164, 593}));
    // // samples.push_back(tuple<Graph, vector<Vertex>, vector<Vertex>>(query_graphs[0], {4,5}, {240, 2384}));

    // // Feature_selector fs(71, 6, 4, true, 1, ".");
    // // vector<vector<Label>> features = fs.extract_for_singe_graph(sample_queries, query_anchors, data_anchors, data_graph, 1, 64, 1);
    // // for(auto f : features){
    // //     cout<<"[";
    // //     for(auto e : f){
    // //         cout<<e<<", ";
    // //     }
    // //     cout<<"]"<<endl;
    // // }

    // // test
// }


int main(){
    // Graph query_graph("human_connected_test.gr");
    vector<Graph> query_graphs;
    vector<vector<Vertex>> query_anchors, data_anchors;
    load_sample_list("../../dataset/enumeration/wordnet/queries/query_edge_16.gr", query_graphs, query_anchors, data_anchors);
    Graph& query_graph = query_graphs[48];
    // Core composition(&query_graph);
    // cout<<"cores:"<<endl;
    // for(auto vec : composition.cores){
    //     cout<<"{";
    //     for(auto v : vec){
    //         cout<<v<<", ";
    //     }
    //     cout<<"}"<<endl;
    // }
    // cout<<"cut vertices:"<<endl;
    // cout<<"{";
    // for(auto v : composition.cutVertices){
    //     cout<<v<<", ";
    // }
    // cout<<"}"<<endl;

    string dataset("wordnet");
    string dataname("wordnet");
    string fnum("128");
    Graph data_graph(string("../../../raw_dataset/Dataset/InMem/")+dataset+string("/data_graph/")+dataname+string(".gr"));
    Index_manager vertex_cycle_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_vertex.index"));
    Index_manager edge_cycle_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_edge.index"));
    Index_manager vertex_path_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_vertex.index"));
    Index_manager edge_path_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_edge.index"));

    Tensor* vertex_cycle_d = vertex_cycle_manager.load_graph_tensor(0);
    Tensor* vertex_path_d = vertex_path_manager.load_graph_tensor(0);
    Tensor* edge_cycle_d = edge_cycle_manager.load_graph_tensor(0);
    Tensor* edge_path_d = edge_path_manager.load_graph_tensor(0);
    cout<<"done loading tensors"<<endl;
    vector<Tensor*> vertex_vec = {vertex_cycle_d,vertex_path_d};
    vector<Tensor*> edge_vec = {edge_cycle_d,edge_path_d};
    Tensor* vertex_emb_d = merge_multi_Tensors(vertex_vec);
    Tensor* edge_emb_d = merge_multi_Tensors(edge_vec);
    delete vertex_cycle_d,vertex_path_d,edge_cycle_d,edge_path_d;
    cout<<"done merging tensors"<<endl;
    // compute the embedding for queries
    vector<vector<Label>> vertex_cycle_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_vertex.features"));
    vector<vector<Label>> edge_cycle_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_edge.features"));
    vector<vector<Label>> vertex_path_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_vertex.features"));
    vector<vector<Label>> edge_path_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_edge.features"));
    
    Cycle_counter c_v_counter = Cycle_counter(true, vertex_cycle_features);
    Path_counter p_v_counter = Path_counter(true, vertex_path_features);
    Cycle_counter c_e_counter = Cycle_counter(true, edge_cycle_features);
    Path_counter p_e_counter = Path_counter(true, edge_path_features);

    Index_constructer c_v_con = Index_constructer(&c_v_counter);
    Index_constructer p_v_con = Index_constructer(&p_v_counter);
    Index_constructer c_e_con = Index_constructer(&c_e_counter);
    Index_constructer p_e_con = Index_constructer(&p_e_counter);

    struct timeval start_t, end_t;
    cout<<"start enumeration"<<endl;

    gettimeofday(&start_t, NULL);
    Tensor* vertex_cycle_q = c_v_con.count_features(query_graph, 1, 0);
    Tensor* vertex_path_q = p_v_con.count_features(query_graph, 1, 0);
    Tensor* edge_cycle_q = c_e_con.count_features(query_graph, 1, 1);
    Tensor* edge_path_q = p_e_con.count_features(query_graph, 1, 1);
    vertex_vec = {vertex_cycle_q,vertex_path_q};
    edge_vec = {edge_cycle_q,edge_path_q};
    Tensor* vertex_emb_q = merge_multi_Tensors(vertex_vec);
    Tensor* edge_emb_q = merge_multi_Tensors(edge_vec);
    delete vertex_cycle_q,vertex_path_q,edge_cycle_q,edge_path_q;
    gettimeofday(&end_t, NULL);
    // start enumeration
    long result_count;
    double enumeration_time, preprocessing_time;
    // cout<<vertex_emb_d->to_string()<<endl;
    subgraph_enumeration(&data_graph, &query_graph, 
        100000, result_count, 
        enumeration_time, preprocessing_time, 
        vertex_emb_q, vertex_emb_d,
        edge_emb_q, edge_emb_d,
        // NULL, NULL,
        false // whether order the candidates
    );
    cout<<"query_indexing:"<<get_time(start_t, end_t)<<" result:"<<result_count<<" preprocessing_time:"<<preprocessing_time<<" enumeration_time:"<<enumeration_time<<endl;
    delete vertex_emb_q, edge_emb_q;







    // exp: run the enumeration in batches
    // string dataset("eu2005");
    // string dataname("eu2005");
    // string fnum("128");
    // Graph data_graph(string("../../../raw_dataset/Dataset/InMem/")+dataset+string("/data_graph/")+dataname+string(".gr"));
    


    // Index_manager vertex_cycle_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_vertex.index"));
    // Index_manager edge_cycle_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_edge.index"));
    // Index_manager vertex_path_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_vertex.index"));
    // Index_manager edge_path_manager(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_edge.index"));

    // Tensor* vertex_cycle_d = vertex_cycle_manager.load_graph_tensor(0);
    // Tensor* vertex_path_d = vertex_path_manager.load_graph_tensor(0);
    // Tensor* edge_cycle_d = edge_cycle_manager.load_graph_tensor(0);
    // Tensor* edge_path_d = edge_path_manager.load_graph_tensor(0);
    // cout<<"done loading tensors"<<endl;
    // vector<Tensor*> vertex_vec = {vertex_cycle_d,vertex_path_d};
    // vector<Tensor*> edge_vec = {edge_cycle_d,edge_path_d};
    // Tensor* vertex_emb_d = merge_multi_Tensors(vertex_vec);
    // Tensor* edge_emb_d = merge_multi_Tensors(edge_vec);
    // delete vertex_cycle_d,vertex_path_d,edge_cycle_d,edge_path_d;
    // cout<<"done merging tensors"<<endl;
    // // compute the embedding for queries
    // vector<vector<Label>> vertex_cycle_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_vertex.features"));
    // vector<vector<Label>> edge_cycle_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_cycle_edge.features"));
    // vector<vector<Label>> vertex_path_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_vertex.features"));
    // vector<vector<Label>> edge_path_features = load_label_path(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/run_exp/enumeration/.index/")+dataname+string(".gr_")+fnum+string("_4_1_path_edge.features"));
    
    // Cycle_counter c_v_counter = Cycle_counter(true, vertex_cycle_features);
    // Path_counter p_v_counter = Path_counter(true, vertex_path_features);
    // Cycle_counter c_e_counter = Cycle_counter(true, edge_cycle_features);
    // Path_counter p_e_counter = Path_counter(true, edge_path_features);

    // Index_constructer c_v_con = Index_constructer(&c_v_counter);
    // Index_constructer p_v_con = Index_constructer(&p_v_counter);
    // Index_constructer c_e_con = Index_constructer(&c_e_counter);
    // Index_constructer p_e_con = Index_constructer(&p_e_counter);

    // vector<string> query_files = {
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_edge_8.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_edge_16.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_edge_24.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_edge_32.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_edge_40.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_vertex_8.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_vertex_16.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_vertex_24.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_vertex_32.gr"),
    //     string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_vertex_40.gr")
    // };
    // struct timeval start_t, end_t;
    // cout<<"start enumeration"<<endl;
    // fstream fout("youtube_enumeration_result");
    // fout<<"query\tindexing_time\tresult\tpreprocessing_time\tenumeration_time"<<endl;
    // for(auto filename : query_files){
    //     vector<Graph> query_graphs;
    //     vector<vector<Vertex>> query_anchors, data_anchors;
    //     load_sample_list(filename, query_graphs, query_anchors, data_anchors);
    //     // generate the query embeddings 
    //     for(int id=0;id<query_graphs.size();++id){
    //         gettimeofday(&start_t, NULL);
    //         Tensor* vertex_cycle_q = c_v_con.count_features(query_graphs[id], 1, 0);
    //         Tensor* vertex_path_q = p_v_con.count_features(query_graphs[id], 1, 0);
    //         Tensor* edge_cycle_q = c_e_con.count_features(query_graphs[id], 1, 1);
    //         Tensor* edge_path_q = p_e_con.count_features(query_graphs[id], 1, 1);
    //         vertex_vec = {vertex_cycle_q,vertex_path_q};
    //         edge_vec = {edge_cycle_q,edge_path_q};
    //         Tensor* vertex_emb_q = merge_multi_Tensors(vertex_vec);
    //         Tensor* edge_emb_q = merge_multi_Tensors(edge_vec);
    //         delete vertex_cycle_q,vertex_path_q,edge_cycle_q,edge_path_q;
    //         gettimeofday(&end_t, NULL);
    //         // start enumeration
    //         long result_count;
    //         double enumeration_time, preprocessing_time;
    //         // cout<<vertex_emb_d->to_string()<<endl;
    //         subgraph_enumeration(&data_graph, &query_graphs[id], 
    //             100000, result_count, 
    //             enumeration_time, preprocessing_time, 
    //             vertex_emb_q, vertex_emb_d,
    //             edge_emb_q, edge_emb_d,
    //             // NULL, NULL,
    //             false // whether order the candidates
    //         );
    //         string tag = filename+string("-")+to_string(id);
    //         cout<<tag<<" : "<<"query_indexing:"<<get_time(start_t, end_t)<<" result:"<<result_count<<" preprocessing_time:"<<preprocessing_time<<" enumeration_time:"<<enumeration_time<<endl;
    //         fout<<tag<<"\t"<<get_time(start_t, end_t)<<"\t"<<result_count<<"\t"<<preprocessing_time<<"\t"<<enumeration_time<<endl;
    //         delete vertex_emb_q, edge_emb_q;
    //     }
    // }

    // get the candidate info
    // uint64_t vertex_total_num, edge_total_num;
    // vertex_total_num = 0;
    // edge_total_num = 0;
    // int total_samples = 0;
    // for(auto filename : query_files){
    //     vector<Graph> query_graphs;
    //     vector<vector<Vertex>> query_anchors, data_anchors;
    //     load_sample_list(filename, query_graphs, query_anchors, data_anchors);
    //     // generate the query embeddings 
    //     for(int id=0;id<query_graphs.size();++id){
    //         Tensor* vertex_cycle_q = c_v_con.count_features(query_graphs[id], 1, 0);
    //         Tensor* vertex_path_q = p_v_con.count_features(query_graphs[id], 1, 0);
    //         Tensor* edge_cycle_q = c_e_con.count_features(query_graphs[id], 1, 1);
    //         Tensor* edge_path_q = p_e_con.count_features(query_graphs[id], 1, 1);
    //         vertex_vec = {vertex_cycle_q,vertex_path_q};
    //         edge_vec = {edge_cycle_q,edge_path_q};
    //         Tensor* vertex_emb_q = merge_multi_Tensors(vertex_vec);
    //         Tensor* edge_emb_q = merge_multi_Tensors(edge_vec);
    //         delete vertex_cycle_q,vertex_path_q,edge_cycle_q,edge_path_q;
    //         // start enumeration
    //         long result_count;
    //         uint32_t vertex_num, edge_num;
    //         // cout<<vertex_emb_d->to_string()<<endl;
    //         subgraph_enumeration_get_candidate_info(&data_graph, &query_graphs[id], 
    //             vertex_emb_q, vertex_emb_d,
    //             // edge_emb_q, edge_emb_d,
    //             NULL, NULL,
    //             vertex_num, edge_num
    //         );
    //         string tag = filename+string("-")+to_string(id);
    //         cout<<tag<<"\t"<<vertex_num<<"\t"<<edge_num<<endl;
    //         delete vertex_emb_q, edge_emb_q;
    //         total_samples ++;
    //         vertex_total_num += vertex_num;
    //         edge_total_num += edge_num;
    //     }
    // }
    // cout<<"average vertex candidates:"<<vertex_total_num/float(total_samples)<<endl;
    // cout<<"average edge candidates:"<<edge_total_num/float(total_samples)<<endl;
    // cout<<"start enumeration"<<endl;
    // long result_count;
    // double enumeration_time, preprocessing_time;
    // // cout<<vertex_emb_d->to_string()<<endl;
    // subgraph_enumeration(&data_graph, &query_graphs[41], 
    //     100000, result_count, 
    //     enumeration_time, preprocessing_time, 
    //     vertex_emb_q, vertex_emb_d,
    //     // edge_emb_q, edge_emb_d,
    //     NULL, NULL,
    //     // NULL, NULL,
    //     false // whether order the candidates
    // );

    // cout<<"result:"<<result_count<<" enumeration time:"<<enumeration_time<<" preprocessing time:"<<preprocessing_time<<endl;

    return 0;
}