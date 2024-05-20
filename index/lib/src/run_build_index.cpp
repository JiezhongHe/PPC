#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../utility/utils.h"
#include "../utility/core_decomposition.h"
#include "../index/feature_selector.h"

void load_sample_list(string filename, vector<Graph>& result, vector<vector<Vertex>>& query_anchors, vector<vector<Vertex>>& data_anchors, vector<Vertex>& data_graph_ids){
    ifstream in(filename);

    string line;
    unordered_map<Vertex, Label> label_map_unordered;
    unordered_map<Vertex, unordered_set<Vertex>> adj_unordered;
    vector<Label> label_map;
    vector<unordered_set<Vertex>> adj;
    vector<Vertex> q_anchors;
    vector<Vertex> d_anchors;
    Vertex data_graph_id;
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
                data_graph_ids.push_back(data_graph_id);
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
            ss>>c>>v>>data_graph_id;
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
        data_graph_ids.push_back(data_graph_id);
    }
}

void my_delete_directory(string path){
    string cmd = string("rm -r ")+path;
    system(cmd.c_str());
}

void my_create_directory(string path, bool overwrite){
    if(access(path.c_str(), 0) == -1){
        mkdir(path.c_str(), S_IRWXU);
    }else{
        if(overwrite == true){
            my_delete_directory(path.c_str());
            mkdir(path.c_str(), S_IRWXU);
        }
    }
}


struct Param{
    string data_file;
    string sample_file_e;
    string sample_file_v;
    int feature_length;
    int feature_count;
    string output;
    string PV_data_index;
    string PV_feature;
    string PE_data_index;
    string PE_feature;
    string CV_data_index;
    string CV_feature;
    string CE_data_index;
    string CE_feature;
    string tmp_dir;
    string dataset_name;
    int thread_count;
    int batch_size; // number of features computed at a time. If memory is not enough, set a smaller value
    Vertex max_label;
    bool enable_residual;
};

static struct Param parsed_input_para;

static const struct option long_options[] = {
    {"data", required_argument, NULL, 'd'},
    {"se", required_argument, NULL, 'e'},
    {"sv", required_argument, NULL, 'v'},
    {"PV", no_argument, NULL, 'x'},
    {"PE", no_argument, NULL, 'y'},
    {"CV", no_argument, NULL, 'm'},
    {"CE", no_argument, NULL, 'n'},
    {"residual", required_argument, NULL, 'r'},
    {"fl", required_argument, NULL, 'f'},
    {"fc", required_argument, NULL, 'c'},
    {"thread", required_argument, NULL, 't'},
    {"output", required_argument, NULL, 'o'},
    {"batch_size", required_argument, NULL, 'b'},
    {"help", no_argument, NULL, '?'},
};

void print_args(){
    cout<<"data:\t"<<parsed_input_para.data_file<<endl;
    cout<<"se:\t"<<parsed_input_para.sample_file_e<<endl;
    cout<<"sv:\t"<<parsed_input_para.sample_file_v<<endl;
    cout<<"feature_length:\t"<<parsed_input_para.feature_length<<endl;
    cout<<"feature_count:\t"<<parsed_input_para.feature_count<<endl;
    cout<<"thread_count:\t"<<parsed_input_para.thread_count<<endl;
    cout<<"residual:\t"<<parsed_input_para.enable_residual<<endl;
    cout<<"output:\t"<<parsed_input_para.output<<endl;
    cout<<"batch_size:\t"<<parsed_input_para.batch_size<<endl;
    cout<<"PV(output path):\t"<<parsed_input_para.PV_data_index<<"\t"<<parsed_input_para.PV_feature<<endl;
    cout<<"PE(output path):\t"<<parsed_input_para.PE_data_index<<"\t"<<parsed_input_para.PE_feature<<endl;
    cout<<"CV(output path):\t"<<parsed_input_para.CV_data_index<<"\t"<<parsed_input_para.CV_feature<<endl;
    cout<<"CE(output path):\t"<<parsed_input_para.CE_data_index<<"\t"<<parsed_input_para.CE_feature<<endl;
}

vector<string> GetFiles(string path, string suffix){
    if(path[path.size()-1] == '/'){
        path = path.substr(0, path.size()-1);
    }

    const char* src_dir = path.c_str();
    const char* ext = suffix.c_str();
    vector<string> result;
    string directory(src_dir);
    string m_ext(ext);

    DIR *dir = opendir(src_dir);
    if ( dir == NULL )
    {
        printf("[ERROR] %s is not a directory or not exist!", src_dir);
        return result;
    }
 
    struct dirent* d_ent = NULL;
 
    char dot[3] = ".";
    char dotdot[6] = "..";
 
    while ( (d_ent = readdir(dir)) != NULL )
    {
        if ( (strcmp(d_ent->d_name, dot) != 0) && (strcmp(d_ent->d_name, dotdot) != 0) )
        {
            if ( d_ent->d_type != DT_DIR)
            {
                string d_name(d_ent->d_name);
                if (strcmp(d_name.c_str () + d_name.length () - m_ext.length(), m_ext.c_str ()) == 0)
                {
                    result.push_back(path+string("/")+string(d_ent->d_name));
                }
            }
        }
    }
 
    // sort the returned files
    sort(result.begin(), result.end());
 
    closedir(dir);
    return result;
}

void parse_args(int argc, char** argv){
    int opt;
    int options_index=0;
    string suffix;
    parsed_input_para.enable_residual = 1;
    parsed_input_para.thread_count = 1;
    parsed_input_para.tmp_dir = ".tmp";
    parsed_input_para.batch_size = 128; 
    bitset<4> index_type;
    index_type.reset();
    while((opt=getopt_long_only(argc, argv, "q:d:x:y:m:n:?", long_options, &options_index)) != -1){
        switch (opt)
        {
        case 0:
            break;
        case 'd':
            parsed_input_para.data_file = string(optarg);
            break;
        case 'e':
            parsed_input_para.sample_file_e = string(optarg);
            break;
        case 'v':
            parsed_input_para.sample_file_v = string(optarg);
            break;
        case 'r':
            parsed_input_para.enable_residual = atoi(optarg);
            break;
        case 'f':
            parsed_input_para.feature_length = atoi(optarg);
            break;
        case 'c':
            parsed_input_para.feature_count = atoi(optarg);
            break;
        case 't':
            parsed_input_para.thread_count = atoi(optarg);
            break;
        case 'o':
            parsed_input_para.output = string(optarg);
            break;
        case 'b':
            parsed_input_para.batch_size = atoi(optarg);
            break;
        case 'x':
            index_type.set(0);
            break;
        case 'y':
            index_type.set(1);
            break;
        case 'm':
            index_type.set(2);
            break;
        case 'n':
            index_type.set(3);
            break;
        case '?':
            cout<<"------------------ args list ------------------------"<<endl;
            cout<<"--data\tpath of the data graph"<<endl;
            cout<<"--se\tpath of the edge anchored samples"<<endl;
            cout<<"--sv\tpath of the vertex anchored samples"<<endl;
            cout<<"--output\tpath of the output index and features"<<endl;
            cout<<"--fl\tlength of the counted path/cycle features"<<endl;
            cout<<"--fc\tsize of the path/cycle feature size"<<endl;
            cout<<"--residual\t0/1 whether enable the residual mechanism during counting(default 1)"<<endl;
            cout<<"--thread\tnumber of threads utilized to build the index(default 1)"<<endl;
            cout<<"--batch_size\tnumber of features computed at one time(default 128)"<<endl;
            cout<<"--PV\t choose whether or not build the PPC-PV index(optional)"<<endl;
            cout<<"--PE\t choose whether or not build the PPC-PE index(optional)"<<endl;
            cout<<"--CV\t choose whether or not build the PPC-CV index(optional)"<<endl;
            cout<<"--CE\t choose whether or not build the PPC-CE index(optional)"<<endl;
            break;
        default:
            break;
        }
    }
    // get the name of the dataset
    istringstream iss(parsed_input_para.data_file);
    while(getline(iss, parsed_input_para.dataset_name, '/'));

    if(parsed_input_para.output[parsed_input_para.output.size()-1] == '/'){
        parsed_input_para.output = parsed_input_para.output.substr(0, parsed_input_para.output.size()-1);
    }
    // generate the corresponding output file
    if(index_type.test(0)){
        parsed_input_para.PV_data_index = parsed_input_para.output+string("/")+ string("path_in_vertex.index");
        parsed_input_para.PV_feature = parsed_input_para.output+string("/")+string("path_in_vertex.features");
    }
    if(index_type.test(1)){
        parsed_input_para.PE_data_index = parsed_input_para.output+string("/")+string("path_in_edge.index");
        parsed_input_para.PE_feature = parsed_input_para.output+string("/")+string("path_in_edge.features");
    }
    if(index_type.test(2)){
        parsed_input_para.CV_data_index = parsed_input_para.output+string("/")+string("cycle_in_vertex.index");
        parsed_input_para.CV_feature = parsed_input_para.output+string("/")+string("cycle_in_vertex.features");
    }
    if(index_type.test(3)){
        parsed_input_para.CE_data_index = parsed_input_para.output+string("/")+string("cycle_in_edge.index");
        parsed_input_para.CE_feature = parsed_input_para.output+string("/")+string("cycle_in_edge.features");
    }
}

void generate_features(vector<Graph>& data_graphs, string feature_file, vector<string> sample_files, bool enable_cycle, int level){
    vector<Graph> query_graphs;
    vector<vector<Vertex>> query_anchors;
    vector<vector<Vertex>> data_anchors;
    vector<Vertex> data_graph_ids;
    my_create_directory(parsed_input_para.tmp_dir, true);

    for(auto& sample : sample_files){
        load_sample_list(sample, query_graphs, query_anchors, data_anchors, data_graph_ids);
    }

    Feature_selector fs(parsed_input_para.max_label+1, parsed_input_para.feature_count, parsed_input_para.feature_length, enable_cycle, parsed_input_para.tmp_dir);
    vector<vector<Label>> features;
    if(data_graphs.size() == 1){
        features = fs.extract_for_singe_graph(query_graphs, query_anchors, data_anchors, data_graphs[0], level, parsed_input_para.batch_size, parsed_input_para.thread_count);
    }else{
        features = fs.extract_for_multi_graph(query_graphs, query_anchors, data_anchors, data_graph_ids, data_graphs, level, parsed_input_para.batch_size, parsed_input_para.thread_count);
    }
    
    dump_features(features, feature_file);
    my_delete_directory(parsed_input_para.tmp_dir);
}

bool file_exists(string file){
    ifstream fin(file.c_str());
    return fin.good();
}

void* thread_multi_graph(void* args){
    tuple<Graph**, Tensor**, int, Index_constructer*, int> input = *((tuple<Graph**, Tensor**, int, Index_constructer*, int>*)args);
    Graph** data_graphs = std::get<0>(input);
    Tensor** result = std::get<1>(input);
    int data_count = std::get<2>(input);
    Index_constructer* index = std::get<3>(input);
    int level = std::get<4>(input);
    for(int i=0;i<data_count;++i){
        result[i] = index->count_features(*(data_graphs[i]), 1, level);
    }
    return NULL;
}


int main(int argc, char** argv){
    parse_args(argc, argv);
    print_args();

    my_create_directory(parsed_input_para.output, false);
    vector<Graph> data_graphs;
    vector<vector<Vertex>> query_anchors, data_anchors; // useless
    vector<Vertex> data_graph_ids;
    load_sample_list(parsed_input_para.data_file, data_graphs, query_anchors, data_anchors, data_graph_ids);

    // get the largest label_id
    parsed_input_para.max_label = 0;
    for(auto& data_graph : data_graphs){
        
        for(Vertex v=0;v<data_graph.label_map.size();++v){
            if(data_graph.label_map[v] > parsed_input_para.max_label){
                parsed_input_para.max_label = data_graph.label_map[v];
            }
        }
    }

    vector<double> building_time(4, 0);
    
    // generate features
    vector<string> vertex_anchored_samples = GetFiles(parsed_input_para.sample_file_v, string(".gr"));
    vector<string> edge_anchored_samples = GetFiles(parsed_input_para.sample_file_e, string(".gr"));
    struct timeval start_t, end_t;

    // int upper_limit = 128;
    
    if(!parsed_input_para.PV_data_index.empty()){
        cout<<"start generating features for PPC-PV"<<endl;
        if(file_exists(parsed_input_para.PV_data_index)){
            cout<<"PPC-PV already built"<<endl;
        }else{
            if(file_exists(parsed_input_para.PV_feature)){
                cout<<"PPC-PV features are already generated"<<endl;
            }else{
                cout<<"start generating features for PPC-PV"<<endl;
                generate_features(data_graphs, parsed_input_para.PV_feature, vertex_anchored_samples, false, 0);
                cout<<"finished generating features for PPC-PV"<<endl;
            }
            cout<<"start building PPC-PV"<<endl;
            // load features
            gettimeofday(&start_t, NULL);
            vector<vector<Label>> features_tmp = load_label_path(parsed_input_para.PV_feature);
            vector<vector<Label>> features(&(features_tmp[0]), &(features_tmp[0]));
            Path_counter counter(parsed_input_para.enable_residual, features);
            Index_constructer index(&counter);
            for(auto& data_graph : data_graphs){
                if(features.size() <= parsed_input_para.batch_size){
                    index.construct_index_single_batch(data_graph, parsed_input_para.PV_data_index, parsed_input_para.thread_count, 0);
                }else{
                    index.construct_index_in_batch(data_graph, parsed_input_para.PV_data_index, parsed_input_para.batch_size, parsed_input_para.thread_count, 0);
                }
                building_time[0] += index.build_time;
            }
            cout<<"finish building PPC-PV:"<<building_time[0]<<endl;
        }
    }

    if(!parsed_input_para.PE_data_index.empty()){
        cout<<"start generating features for PPC-PE"<<endl;
        if(file_exists(parsed_input_para.PE_data_index)){
            cout<<"PPC-PE already built"<<endl;
        }else{
            if(file_exists(parsed_input_para.PE_feature)){
                cout<<"PPC-PE features are already generated"<<endl;
            }else{
                cout<<"start generating features for PPC-PE"<<endl;
                generate_features(data_graphs, parsed_input_para.PE_feature, edge_anchored_samples, false, 1);
                cout<<"finished generating features for PPC-PE"<<endl;
            }
            cout<<"start building PPC-PE"<<endl;
            // load features
            gettimeofday(&start_t, NULL);
            vector<vector<Label>> features_tmp = load_label_path(parsed_input_para.PE_feature);
            vector<vector<Label>> features(&(features_tmp[0]), &(features_tmp[0]));
            Path_counter counter(parsed_input_para.enable_residual, features);
            Index_constructer index(&counter);
            for(auto& data_graph : data_graphs){
                if(features.size() <= parsed_input_para.batch_size){
                    index.construct_index_single_batch(data_graph, parsed_input_para.PE_data_index, parsed_input_para.thread_count, 1);
                }else{
                    index.construct_index_in_batch(data_graph, parsed_input_para.PE_data_index, parsed_input_para.batch_size, parsed_input_para.thread_count, 1);
                }
                building_time[1] += index.build_time;
            }
            gettimeofday(&end_t, NULL);
            cout<<"finish building PPC-PE:"<<building_time[1]<<endl;
        }
    }

    if(!parsed_input_para.CV_data_index.empty()){
        cout<<"start generating features for PPC-CV"<<endl;
        if(file_exists(parsed_input_para.CV_data_index)){
            cout<<"PPC-CV already built"<<endl;
        }else{
            if(file_exists(parsed_input_para.CV_feature)){
                cout<<"PPC-CV features are already generated"<<endl;
            }else{
                cout<<"start generating features for PPC-CV"<<endl;
                generate_features(data_graphs, parsed_input_para.CV_feature, vertex_anchored_samples, true, 0);
                cout<<"finished generating features for PPC-CV"<<endl;
            }
            cout<<"start building PPC-CV"<<endl;
            // load features
            gettimeofday(&start_t, NULL);
            vector<vector<Label>> features_tmp = load_label_path(parsed_input_para.CV_feature);
            vector<vector<Label>> features(&(features_tmp[0]), &(features_tmp[0]));
            Cycle_counter counter(parsed_input_para.enable_residual, features);
            Index_constructer index(&counter);
            for(auto& data_graph : data_graphs){
                if(features.size() <= parsed_input_para.batch_size){
                    index.construct_index_single_batch(data_graph, parsed_input_para.CV_data_index, parsed_input_para.thread_count, 0);
                }else{
                    index.construct_index_in_batch(data_graph, parsed_input_para.CV_data_index, parsed_input_para.batch_size, parsed_input_para.thread_count, 0);
                }
                building_time[2] += index.build_time;
            }
            gettimeofday(&end_t, NULL);
            cout<<"finish building PPC-CV:"<<building_time[2]<<endl;
        }
    }

    if(!parsed_input_para.CE_data_index.empty()){
        cout<<"start generating features for PPC-CE"<<endl;
        if(file_exists(parsed_input_para.CE_data_index)){
            cout<<"PPC-CE already built"<<endl;
        }else{
            if(file_exists(parsed_input_para.CE_feature)){
                cout<<"PPC-CE features are already generated"<<endl;
            }else{
                cout<<"start generating features for PPC-CE"<<endl;
                generate_features(data_graphs, parsed_input_para.CE_feature, vertex_anchored_samples, true, 1);
                cout<<"finished generating features for PPC-CE"<<endl;
            }
            cout<<"start building PPC-CE"<<endl;
            // load features
            gettimeofday(&start_t, NULL);
            vector<vector<Label>> features_tmp = load_label_path(parsed_input_para.CE_feature);
            vector<vector<Label>> features(&(features_tmp[0]), &(features_tmp[0]));
            Cycle_counter counter(parsed_input_para.enable_residual, features);
            Index_constructer index(&counter);
            for(auto& data_graph : data_graphs){
                if(features.size() <= parsed_input_para.batch_size){
                    index.construct_index_single_batch(data_graph, parsed_input_para.CE_data_index, parsed_input_para.thread_count, 1);
                }else{
                    index.construct_index_in_batch(data_graph, parsed_input_para.CE_data_index, parsed_input_para.batch_size, parsed_input_para.thread_count, 1);
                }
                building_time[3] += index.build_time;
            }
            gettimeofday(&end_t, NULL);
            cout<<"finish building PPC-CE:"<<building_time[3]<<endl;
            
        }
    }
    
    cout<<"================= build info ==========="<<endl;
    if(building_time[0] == 0){
        cout<<"construction time for PPC-PV:"<<building_time[0]<<"(already generated)"<<endl;
    }else{
        cout<<"construction time for PPC-PV:"<<building_time[0]<<endl;
    }
    if(building_time[1] == 0){
        cout<<"construction time for PPC-PE:"<<building_time[1]<<"(already generated)"<<endl;
    }else{
        cout<<"construction time for PPC-PE:"<<building_time[1]<<endl;
    }
    if(building_time[2] == 0){
        cout<<"construction time for PPC-CV:"<<building_time[2]<<"(already generated)"<<endl;
    }else{
        cout<<"construction time for PPC-CV:"<<building_time[2]<<endl;
    }
    if(building_time[3] == 0){
        cout<<"construction time for PPC-CE:"<<building_time[3]<<"(already generated)"<<endl;
    }else{
        cout<<"construction time for PPC-CE:"<<building_time[3]<<endl;
    }
}
