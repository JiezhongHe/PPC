#include <iostream>
#include "graph.hpp"
#include "embedding.hpp"
#include "cycle_counting.hpp"
#include "path_counting.hpp"
#include "subgraph_enumeration_for_sample.hpp"
#include "subgraph_retrieval.hpp"
#include "feature_selector.hpp"
#include "index.hpp"
#include "utils.hpp"

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


int main(){
    string dataset("yeast");
    string dataname("yeast");
    string fnum("128");
    int id=17;
    string result_sample_file("yeast_edge_24_17_sample");
    vector<Graph> query_graphs;
    vector<vector<Vertex>> query_anchors, data_anchors;
    load_sample_list(string("/home/yixin/jiezhonghe/project/VLDB/PPC-index/dataset/enumeration/")+dataset+string("/queries/query_edge_24.gr"), query_graphs, query_anchors, data_anchors);
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

    Tensor* vertex_cycle_q = c_v_con.count_features(query_graphs[id], 1, 0);
    Tensor* vertex_path_q = p_v_con.count_features(query_graphs[id], 1, 0);
    Tensor* edge_cycle_q = c_e_con.count_features(query_graphs[id], 1, 1);
    Tensor* edge_path_q = p_e_con.count_features(query_graphs[id], 1, 1);
    vertex_vec = {vertex_cycle_q,vertex_path_q};
    edge_vec = {edge_cycle_q,edge_path_q};
    Tensor* vertex_emb_q = merge_multi_Tensors(vertex_vec);
    Tensor* edge_emb_q = merge_multi_Tensors(edge_vec);
    delete vertex_cycle_q,vertex_path_q,edge_cycle_q,edge_path_q;
    
    cout<<"start enumeration"<<endl;
    long result_count;
    double enumeration_time, preprocessing_time;
    // cout<<vertex_emb_d->to_string()<<endl;
    subgraph_enumeration_sample(&data_graph, &query_graphs[id], 
        100000, result_count, 
        enumeration_time, preprocessing_time, 
        vertex_emb_q, vertex_emb_d,
        edge_emb_q, edge_emb_d,
        // NULL, NULL,
        // NULL, NULL,
        true, // whether order the candidates
        result_sample_file
    );

    // cout<<"result:"<<result_count<<" enumeration time:"<<enumeration_time<<" preprocessing time:"<<preprocessing_time<<endl;

    return 0;
}