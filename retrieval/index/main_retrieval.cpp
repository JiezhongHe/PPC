#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <getopt.h>
#include <unistd.h>
#include <unordered_set>
#include <limits>
#include "../graph/graph.h"
#include "preprocessor.h"
#include "query_plan_generator.h"
#include "subgraph_enumeration.h"
#include "../index/embedding.h"
#include "../index/cycle_counting.h"
#include "../index/path_counting.h"
#include "../index/index.h"
// #include "../utility/utils.h"
// #include "../model/model.h"
// #include "../graph_decomposition/decomposition.h"

using namespace std;

void getFiles(string path, vector<string>& filenames){
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder does not exist"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0){
        if(strcmp(ptr->d_name, ".")!=0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path+"/"+ptr->d_name);
        }
    }
    closedir(pDir);
}

#if GNN_PRUNING_MARGIN > 0
void load_gnn_embedding(string embedding_path, unordered_map<string, pair<vector<int>, vector<float>>>& loaded_embedding){ //first->shape, second->embedding
    if(embedding_path[embedding_path.size()-1] == '/'){
        embedding_path = embedding_path.substr(0, embedding_path.size()-1);
    }
    string embedding_file = embedding_path + string("/embedding.bin");
    string offset_map_file = embedding_path + string("/offset_map");

    vector<string> query_file_names;
    string filename;
    ifstream fin_offset(offset_map_file);
    while(fin_offset>>filename){
        query_file_names.push_back(filename);
    }

    ifstream fin(embedding_file, ios::binary);
    for(int i=0;i<query_file_names.size();++i){

        int shape_len;
        fin.read((char*)&shape_len, sizeof(int));
        int* shape = new int [shape_len];
        fin.read((char*)shape, sizeof(int)*shape_len);
        int card = 1;
        vector<int> shape_vec;
        for(int i=0;i<shape_len;i++){
            card = card*shape[i];
            shape_vec.push_back(shape[i]);
        }
        vector<float> data;
        data.resize(card);
        fin.read((char*)&data[0], card*sizeof(float));

        loaded_embedding.insert({query_file_names[i], {shape_vec, data}});
    }
}
#endif

void stringsplit(string str, char split, vector<string>& result){
    istringstream iss(str);
    string token;
    while(getline(iss, token, split)){
        result.push_back(token);
    }
}

struct Param{
    string query_path;
    string data_file;
    string VP_path;
    string VC_path;
    string EP_path;
    string EC_path;
    string index_path;
    uint32_t num;
};

static struct Param parsed_input_para;

static const struct option long_options[] = {
    {"query", required_argument, NULL, 'q'},
    {"data", required_argument, NULL, 'd'},
    {"index", required_argument, NULL, 'i'},
    {"help", no_argument, NULL, '?'},
};

void parse_args(int argc, char** argv){
    int opt;
    int options_index=0;
    string suffix;
    parsed_input_para.num = std::numeric_limits<uint32_t>::max();
    while((opt=getopt_long_only(argc, argv, "q:d:n:x:y:z:l:?", long_options, &options_index)) != -1){
        switch (opt)
        {
        case 0:
            break;
        case 'q':
            parsed_input_para.query_path = string(optarg);
            break;
        case 'd':
            parsed_input_para.data_file = string(optarg);
            break;
        case 'x':
            parsed_input_para.VP_path = string(optarg);
            break;
        case 'y':
            parsed_input_para.VC_path = string(optarg);
            break;
        case 'z':
            parsed_input_para.EP_path = string(optarg);
            break;
        case 'l':
            parsed_input_para.EC_path = string(optarg);
            break;
        case 'i':
            parsed_input_para.index_path = string(optarg);
            break;
        case '?':
            cout<<"------------------ args list ------------------------"<<endl;
            cout<<"--query\tpath of the query graph"<<endl;
            cout<<"--data\tpath of the data graph"<<endl;
            break;
        default:
            break;
        }
    }
}


void print_input_args(){
    cout<<"input_data_graph:"<<parsed_input_para.data_file<<endl;
    cout<<"input_query: "<<parsed_input_para.query_path<<endl;
}


int main(int argc, char** argv){
    parse_args(argc, argv);
    if(parsed_input_para.data_file[parsed_input_para.data_file.size()-1] == '/'){
        parsed_input_para.data_file = parsed_input_para.data_file.substr(0, parsed_input_para.data_file.size()-1);
    }
    vector<string> data_files;
    getFiles(parsed_input_para.data_file, data_files);
    int max_id = data_files.size();
    data_files.clear();
    for(int id=0;id<max_id;++id){
        data_files.push_back(parsed_input_para.data_file+string("/graph_")+to_string(id)+string(".graph"));
    }
    vector<Graph*> data_graphs;
#if COMPACT == 0
    vector<Tensor*> data_edge_index;
    vector<Tensor*> data_vertex_index;
#else
    vector<CompactTensor*> data_edge_index;
    vector<CompactTensor*> data_vertex_index;
#endif

#if ENABLE_PRE_FILTERING==1
    if(parsed_input_para.index_path[parsed_input_para.index_path.size()-1] == '/'){
        parsed_input_para.index_path = parsed_input_para.index_path.substr(0, parsed_input_para.index_path.size()-1);
    }
    parsed_input_para.VC_path = parsed_input_para.index_path+string("/cycle_in_vertex.index");
    parsed_input_para.EC_path = parsed_input_para.index_path+string("/cycle_in_edge.index");
    parsed_input_para.VP_path = parsed_input_para.index_path+string("/path_in_vertex.index");
    parsed_input_para.EP_path = parsed_input_para.index_path+string("/path_in_edge.index");

    vector<vector<Label>> vc_features = load_label_path(parsed_input_para.VC_path.substr(0, parsed_input_para.VC_path.size()-5)+string("features"));
    vector<vector<Label>> vp_features = load_label_path(parsed_input_para.VP_path.substr(0, parsed_input_para.VP_path.size()-5)+string("features"));
    vector<vector<Label>> ec_features = load_label_path(parsed_input_para.EC_path.substr(0, parsed_input_para.EC_path.size()-5)+string("features"));
    vector<vector<Label>> ep_features = load_label_path(parsed_input_para.EP_path.substr(0, parsed_input_para.EP_path.size()-5)+string("features"));

    Cycle_counter vc_counter = Cycle_counter(true, vc_features);
    Cycle_counter ec_counter = Cycle_counter(true, ec_features);
    Path_counter vp_counter = Path_counter(true, vp_features);
    Path_counter ep_counter = Path_counter(true, ep_features);

    Index_constructer vc_con = Index_constructer(&vc_counter);
    Index_constructer ec_con = Index_constructer(&ec_counter);
    Index_constructer vp_con = Index_constructer(&vp_counter);
    Index_constructer ep_con = Index_constructer(&ep_counter);

    Index_manager vc_manager(parsed_input_para.VC_path);
    Index_manager ec_manager(parsed_input_para.EC_path);
    Index_manager vp_manager(parsed_input_para.VP_path);
    Index_manager ep_manager(parsed_input_para.EP_path);

    Tensor** query_emb_result;
    int query_emb_result_size;
#endif

    int id = 0;
    for(auto file : data_files){
        Graph* graph = new Graph(true);
        graph->loadGraphFromFile(file);
        graph->set_up_edge_id_map();
        data_graphs.push_back(graph);

#if ENABLE_PRE_FILTERING==1
#if COMPACT == 0
        Tensor* vc_d = vc_manager.load_graph_tensor(0);
        Tensor* vp_d = vp_manager.load_graph_tensor(0);
        vector<Tensor*> vd = {vc_d, vp_d};
        data_vertex_emb = merge_multi_Tensors(vd);
        data_vertex_index.push_back(data_vertex_emb_comp);
        delete vc_d, vp_d;
        Tensor* ec_d = ec_manager.load_graph_tensor(0);
        Tensor* ep_d = ep_manager.load_graph_tensor(0);
        vector<Tensor*> ed = {ec_d, ep_d};
        data_edge_emb = merge_multi_Tensors(ed);
        data_edge_index.push_back(data_edge_emb_comp);
        delete ec_d, ep_d;
#else
        CompactTensor* vc_d = vc_manager.load_graph_compact_tensor(id);
        CompactTensor* vp_d = vp_manager.load_graph_compact_tensor(id);
        data_vertex_emb_comp = merge_bi_CompactTensors(vc_d, vp_d);
        data_vertex_index.push_back(data_vertex_emb_comp);
        delete vc_d, vp_d;
        CompactTensor* ec_d = ec_manager.load_graph_compact_tensor(id);
        CompactTensor* ep_d = ep_manager.load_graph_compact_tensor(id);
        data_edge_emb_comp = merge_bi_CompactTensors(ec_d, ep_d);
        data_edge_index.push_back(data_edge_emb_comp);
        delete ec_d, ep_d;
    // exit(0);
#endif
#endif
        id ++;
    }


    vector<string> query_files;
    getFiles(parsed_input_para.query_path, query_files);
    int qid = 0;
    for(auto file : query_files){
        Graph* query_graph = new Graph(true);
        query_graph->loadGraphFromFile(file);
        query_graph->buildCoreTable();
        query_graph->set_up_edge_id_map();

#if ENABLE_PRE_FILTERING==1
#if COMPACT ==0
        Tensor* vc_q = vc_con.count_features(*query_graph, 1, 0);
        Tensor* vp_q = vp_con.count_features(*query_graph, 1, 0);
        Tensor* ec_q = ec_con.count_features(*query_graph, 1, 1);
        Tensor* ep_q = ep_con.count_features(*query_graph, 1, 1);
        vector<Tensor*> vq_tmp = {vc_q, vp_q};
        vector<Tensor*> eq_tmp = {ec_q, ep_q};
        query_vertex_emb = merge_multi_Tensors(vq_tmp);
        query_edge_emb = merge_multi_Tensors(eq_tmp);
#else
        Tensor* vc_q = vc_con.count_features(*query_graph, 1, 0);
        Tensor* vp_q = vp_con.count_features(*query_graph, 1, 0);
        Tensor* ec_q = ec_con.count_features(*query_graph, 1, 1);
        Tensor* ep_q = ep_con.count_features(*query_graph, 1, 1);
        query_vertex_emb_comp = merge_bi_CompactTensors(vc_q, vp_q);
        query_edge_emb_comp = merge_bi_CompactTensors(ec_q, ep_q);
#endif
#endif

        int did = 0;
        int contained_graphs_pred_by_index = 0;
        int contained_graphs_ground_truth = 0;
        float total_val_time = 0;
        for(auto data_graph : data_graphs){
#if ENABLE_PRE_FILTERING==1
#if COMPACT ==0
            data_edge_emb = data_edge_index[did];
            data_vertex_emb = data_vertex_index[did];
#else
            data_edge_emb_comp = data_edge_index[did];
            data_vertex_emb_comp = data_vertex_index[did];
#endif
#endif
            preprocessor* pp_ = new preprocessor();
            catalog* storage_ = new catalog(query_graph, data_graph);
            pp_->execute(query_graph, data_graph, storage_, true);
            float processing_time = NANOSECTOSEC(pp_->preprocess_time_);
            total_val_time += processing_time;
            bool is_contained = true;
            for(int u=0;u<query_graph->getVerticesCount();++u){
                if(storage_->num_candidates_[u] == 0){
                    is_contained = false;
                    break;
                }
            }
            if(is_contained == true){
                contained_graphs_pred_by_index ++;
            }

            // deriving the ground truth
            // if(is_contained == true){
                SubgraphEnum subgraph_enum(data_graph);
                subgraph_enum.match(query_graph, string("nd"), 1, 300);
                if(subgraph_enum.emb_count_ == 0){
                    contained_graphs_ground_truth ++;
                }
            // }
            
            did ++;
            delete pp_;
            delete storage_;
        }
        cout<<contained_graphs_ground_truth<<endl;
        cout<<file<<":"<<qid<<" precision:"<<contained_graphs_ground_truth/(float)contained_graphs_pred_by_index<<" :validation_time:"<<total_val_time/did<<endl;
        qid ++;
        delete query_graph;
    }

    return 0;
}
