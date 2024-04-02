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
    // {"VP", required_argument, NULL, 'x'},
    // {"VC", required_argument, NULL, 'y'},
    // {"EP", required_argument, NULL, 'z'},
    // {"EC", required_argument, NULL, 'l'},
    {"num", required_argument, NULL, 'n'},
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
        case 'n':
            if(string(optarg).compare(string("MAX")) != 0){
                parsed_input_para.num = atoi(optarg);
            }
            break;
        case '?':
            cout<<"------------------ args list ------------------------"<<endl;
            cout<<"--query\tpath of the query graph"<<endl;
            cout<<"--data\tpath of the data graph"<<endl;
            cout<<"--num\tnumber of results to be found"<<endl;
            break;
        default:
            break;
        }
    }
}


void print_input_args(){
    cout<<"input_data_graph:"<<parsed_input_para.data_file<<endl;
    cout<<"input_query: "<<parsed_input_para.query_path<<endl;
    cout<<"maximun number of results intend to find: "<<parsed_input_para.num<<endl;
}

// int main(int argc, char** argv){
//     parse_args(argc, argv);

//     cout<<"Loading Graphs"<<endl;
//     auto start = std::chrono::high_resolution_clock::now();
//     Graph* data_graph = new Graph(true);
//     Graph* query_graph = new Graph(true);
//     data_graph->loadGraphFromFile(parsed_input_para.data_file);
//     query_graph->loadGraphFromFile(parsed_input_para.query_path);
//     query_graph->buildCoreTable();
//     auto end = std::chrono::high_resolution_clock::now();

//     cout<<"Graph Loading Time (s):"<<NANOSECTOSEC(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())<<endl;
    
//     std::cout << "Query Graph Meta Information" << std::endl;
//     query_graph->printGraphMetaData();
//     std::cout << "-----" << std::endl;
//     std::cout << "Data Graph Meta Information" << std::endl;
//     data_graph->printGraphMetaData();

//     cout<<"Start Querying"<<endl;
//     // long result_count;
//     SubgraphEnum subgraph_enum(data_graph);
//     double enumeration_time, preprocessing_time, ordering_time;
//     long long state_count=0;
//     subgraph_enum.match(query_graph, string("nd"), parsed_input_para.num, 300);
//     enumeration_time = subgraph_enum.enumeration_time_;
//     preprocessing_time = subgraph_enum.preprocessing_time_;
//     ordering_time = subgraph_enum.ordering_time_;
//     state_count = subgraph_enum.state_count_;
//     long result_count = subgraph_enum.emb_count_;

//     cout<<"========== RESULT INFO ============"<<endl;
//     cout<<"Results #:"<<result_count<<endl;
//     cout<<"Total Query Time (s):\t"<<subgraph_enum.query_time_<<endl;
//     cout<<"Enumeration Time (s):\t"<<subgraph_enum.enumeration_time_<<endl;
//     cout<<"Preprocessing Time (s):\t"<<subgraph_enum.preprocessing_time_<<endl;
//     cout<<"Ordering Time (s):\t"<<subgraph_enum.ordering_time_<<endl;
//     cout<<"Order Adjustment Time (s):\t"<<subgraph_enum.order_adjust_time_<<endl;
//     cout<<"State Count:\t"<<subgraph_enum.state_count_<<endl;
// #if PRINT_MEM_INFO == 1
//     cout<<"Peak Memory (MB):\t"<<subgraph_enum.peak_memory_<<endl;
// #endif

// #if PRINT_LEAF_STATE == 1
//     cout<<"============ LEAF STATE DISTRIBUTION ============"<<endl;
//     cout<<" CONFLICT:"<<subgraph_enum.leaf_states_counter_[CONFLICT]<<endl;
//     cout<<" EMPTYSET:"<<subgraph_enum.leaf_states_counter_[EMPTYSET]<<endl;
//     cout<<" SUCCESSOR_EQ_CACHE:"<<subgraph_enum.leaf_states_counter_[SUCCESSOR_EQ_CACHE]<<endl;
//     cout<<" SUBTREE_REDUCTION:"<<subgraph_enum.leaf_states_counter_[SUBTREE_REDUCTION]<<endl;
//     cout<<" PGHOLE_FILTERING:"<<subgraph_enum.leaf_states_counter_[PGHOLE_FILTERING]<<endl;
//     cout<<" FAILING_SETS:"<<subgraph_enum.leaf_states_counter_[FAILING_SETS]<<endl;
//     cout<<" RESULT:"<<subgraph_enum.leaf_states_counter_[RESULT]<<endl;
// #endif

//     return 0;
// }


int main(int argc, char** argv){
    parse_args(argc, argv);
    Graph* data_graph = new Graph(true);
    
    data_graph->loadGraphFromFile(parsed_input_para.data_file);
    data_graph->set_up_edge_id_map();

    vector<string> query_files;
    getFiles(parsed_input_para.query_path, query_files);
    
    SubgraphEnum subgraph_enum(data_graph);
#if ENABLE_PRE_FILTERING==1 || GNN_PRUNING_MARGIN==1
    if(parsed_input_para.index_path[parsed_input_para.index_path.size()-1] == '/'){
        parsed_input_para.index_path = parsed_input_para.index_path.substr(0, parsed_input_para.index_path.size()-1);
    }
    parsed_input_para.VC_path = parsed_input_para.index_path+string("/cycle_in_vertex.index");
    parsed_input_para.EC_path = parsed_input_para.index_path+string("/cycle_in_edge.index");
    parsed_input_para.VP_path = parsed_input_para.index_path+string("/path_in_vertex.index");
    parsed_input_para.EP_path = parsed_input_para.index_path+string("/path_in_edge.index");
#endif

#if ENABLE_PRE_FILTERING==1
#if COMPACT == 0
    // loading data index
    Index_manager vc_manager(parsed_input_para.VC_path);
    Index_manager ec_manager(parsed_input_para.EC_path);
    Index_manager vp_manager(parsed_input_para.VP_path);
    Index_manager ep_manager(parsed_input_para.EP_path);
    cout<<"start loading vertex tensors"<<endl;
    Tensor* vc_d = vc_manager.load_graph_tensor(0);
    Tensor* vp_d = vp_manager.load_graph_tensor(0);
    vector<Tensor*> vd = {vc_d, vp_d};
    cout<<"start merging vertex tensors"<<endl;
    data_vertex_emb = merge_multi_Tensors(vd);
    delete vc_d, vp_d;
    cout<<"start loading edge tensors"<<endl;
    Tensor* ec_d = ec_manager.load_graph_tensor(0);
    Tensor* ep_d = ep_manager.load_graph_tensor(0);
    vector<Tensor*> ed = {ec_d, ep_d};
    cout<<"start merging edge tensors"<<endl;
    data_edge_emb = merge_multi_Tensors(ed);
    delete ec_d, ep_d;
    cout<<"done loading tensors"<<endl;
#else
    Index_manager vc_manager(parsed_input_para.VC_path);
    Index_manager ec_manager(parsed_input_para.EC_path);
    Index_manager vp_manager(parsed_input_para.VP_path);
    Index_manager ep_manager(parsed_input_para.EP_path);
    cout<<"start loading vertex tensors"<<endl;
    CompactTensor* vc_d = vc_manager.load_graph_compact_tensor(0);
    CompactTensor* vp_d = vp_manager.load_graph_compact_tensor(0);
    cout<<"start merging vertex tensors"<<endl;
    data_vertex_emb_comp = merge_bi_CompactTensors(vc_d, vp_d);
    // data_vertex_emb = merge_multi_Tensors(vd);
    delete vc_d, vp_d;
    cout<<"start loading edge tensors"<<endl;
    CompactTensor* ec_d = ec_manager.load_graph_compact_tensor(0);
    CompactTensor* ep_d = ep_manager.load_graph_compact_tensor(0);
    cout<<"start merging edge tensors"<<endl;
    data_edge_emb_comp = merge_bi_CompactTensors(ec_d, ep_d);
    delete ec_d, ep_d;
    cout<<"done loading tensors"<<endl;
    // exit(0);
#endif


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

    Tensor** query_emb_result;
    int query_emb_result_size;
#endif

#if GNN_PRUNING_MARGIN > 0
    unordered_map<string, pair<vector<int>, vector<float>>> embedding_map;
    load_gnn_embedding(parsed_input_para.index_path, embedding_map);
    data_gnn_emb = embedding_map["data_graph"];
#endif

    // setup the index files
    // query_files = {"../../../dataset/yeast/query_graph/query_dense_32_22.graph"};
    // query_files = {"../../../../result/2024-3-10/patents/query_32_10"};
    // query_files = {
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_dense_32_133.graph",
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_dense_32_159.graph",
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_sparse_32_57.graph",
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_dense_32_21.graph",
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_dense_32_62.graph",
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_sparse_32_73.graph",
    //     "../../../../../raw_dataset/Dataset-In-mem/patents/query_graph/query_dense_32_150.graph",
    // };
    int file_id = 0;
    for(auto file : query_files){
        Graph* query_graph = new Graph(true);
        query_graph->loadGraphFromFile(file);
        query_graph->buildCoreTable();
        query_graph->set_up_edge_id_map();


        auto start = std::chrono::high_resolution_clock::now();

#if GNN_PRUNING_MARGIN > 0
        vector<string> splitstr;
        stringsplit(file, '/', splitstr);
        string query_name = *(splitstr.rbegin());
        query_gnn_emb = embedding_map[query_name];
#endif

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
        
        auto end = std::chrono::high_resolution_clock::now();
        float query_preocessing_time = NANOSECTOSEC(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
#if ENABLE_PRE_FILTERING==1
        delete vc_q, vp_q, ec_q, ep_q;
#endif
        // if(file_id<=191){
        //     file_id ++;
        //     continue;
        // }
        // cout<<file<<endl;
        double enumeration_time, preprocessing_time, ordering_time;
        long long state_count=0;
        subgraph_enum.match(query_graph, string("nd"), parsed_input_para.num, 300);
        enumeration_time = subgraph_enum.enumeration_time_;
        preprocessing_time = subgraph_enum.preprocessing_time_;
        ordering_time = subgraph_enum.ordering_time_;
        state_count = subgraph_enum.state_count_;
        long result_count = subgraph_enum.emb_count_;
        cout<<file<<":"<<file_id<<": results:"<<result_count<<" query_emb_time:"<<query_preocessing_time<<" query_time:"<<subgraph_enum.query_time_<<" enumeration_time:"<<enumeration_time<<" preprocessing_time:"<<preprocessing_time<<" ordering_time:"<<ordering_time<<" order_adjust_time:"<<subgraph_enum.order_adjust_time_<<" state_count:"<<state_count
#if PRINT_MEM_INFO == 1
        <<" peak_memory:"<<subgraph_enum.peak_memory_
#endif  
#if PRINT_LEAF_STATE == 1
        <<" CONFLICT:"<<subgraph_enum.leaf_states_counter_[CONFLICT]
        <<" EMPTYSET:"<<subgraph_enum.leaf_states_counter_[EMPTYSET]
        <<" SUCCESSOR_EQ_CACHE:"<<subgraph_enum.leaf_states_counter_[SUCCESSOR_EQ_CACHE]
        <<" SUBTREE_REDUCTION:"<<subgraph_enum.leaf_states_counter_[SUBTREE_REDUCTION]
        <<" PGHOLE_FILTERING:"<<subgraph_enum.leaf_states_counter_[PGHOLE_FILTERING]
        <<" FAILING_SETS:"<<subgraph_enum.leaf_states_counter_[FAILING_SETS]
        <<" RESULT:"<<subgraph_enum.leaf_states_counter_[RESULT]
#endif
        <<endl;
        // if(file_id == 6){
        //     exit(0);
        // }
        file_id++;
        delete query_graph;

        // delete query_vertex_emb;
        // delete query_edge_emb;
        delete query_vertex_emb_comp;
        delete query_edge_emb_comp;
    }

    
    // std::cout << "Query Graph Meta Information" << std::endl;
    // query_graph->printGraphMetaData();
    // std::cout << "-----" << std::endl;
    // std::cout << "Data Graph Meta Information" << std::endl;
    // data_graph->printGraphMetaData();

    // long result_count;
    
    return 0;
}
