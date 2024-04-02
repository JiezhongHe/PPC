#pragma once
#include <stdint.h>
#include "nd_interface.h"
#include "computesetintersection.h"
#include <vector>
#include <unordered_set>

class query_plan_generator {
public:
    static void generate_query_plan_with_nd(Graph *query_graph, vector<uint32_t>& vertex_candidate_info, vector<unordered_map<Vertex, uint32_t>>& edge_candidate_info, std::vector<std::vector<uint32_t>>& vertex_orders);

private:
    static void construct_density_tree(std::vector<nd_tree_node>& density_tree, std::vector<nd_tree_node>& k12_tree,
                                       std::vector<nd_tree_node>& k23_tree, std::vector<nd_tree_node>& k34_tree);

    static void merge_tree(std::vector<nd_tree_node>& density_tree);

    static void traversal_density_tree(Graph *query_graph, vector<uint32_t>& vertex_candidate_info, vector<unordered_map<Vertex, uint32_t>>& edge_candidate_info,
                                       std::vector<nd_tree_node> &density_tree,
                                       std::vector<std::vector<uint32_t>> &vertex_orders,
                                       std::vector<std::vector<uint32_t>> &node_orders);

    static void traversal_node(Graph *query_graph, vector<uint32_t>& vertex_candidate_info, vector<unordered_map<Vertex, uint32_t>>& edge_candidate_info, std::vector<nd_tree_node> &density_tree,
                                  std::vector<uint32_t> &vertex_order, std::vector<uint32_t> &node_order,
                                  vector<bool> &visited_vertex, std::vector<bool> &visited_node, nd_tree_node &cur_node,
                                  std::unordered_set<uint32_t> &extendable_vertex);


    static void eliminate_node(std::vector<nd_tree_node>& density_tree, std::vector<nd_tree_node>& src_tree,
                               std::vector<nd_tree_node>& target_tree);

    static void greedy_expand(Graph *query_graph, vector<uint32_t>& vertex_candidate_info, vector<unordered_map<Vertex, uint32_t>>& edge_candidate_info, std::vector<uint32_t> &vertex_order,
                              std::vector<bool> &visited_vertex, std::unordered_set<uint32_t> &extendable_vertex,
                              uint32_t bn_cnt_threshold);

    static void update_extendable_vertex(Graph *query_graph, uint32_t u,
                                         std::unordered_set<uint32_t> &extendable_vertex,
                                         vector<bool> &visited_vertex);

    static void connectivity_shortest_path(Graph *query_graph, vector<uint32_t>& vertex_candidate_info, vector<unordered_map<Vertex, uint32_t>>& edge_candidate_info, std::vector<uint32_t> &vertex_order,
                                           std::vector<bool> &visited_vertex, nd_tree_node &cur_node,
                                           std::vector<uint32_t> &prev, std::vector<double> &dist);

    static double connectivity_common_neighbors(std::vector<bool> &visited_vertex, nd_tree_node &child_node);

    static void query_plan_correctness_check(Graph *query_graph, std::vector<uint32_t> &vertex_order);
};

