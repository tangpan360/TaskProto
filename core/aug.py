"""
    https://github.com/Shen-Lab/GraphCL
"""

import copy
import random

import dgl
import torch
import numpy as np


def aug_drop_node(graph, root, drop_percent=0.2):
    """
        drop non-root nodes
    """
    num = graph.number_of_nodes()  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = copy.deepcopy(graph)
    all_node_list = [i for i in range(num) if i != root]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph.remove_nodes(drop_node_list)
    aug_graph = add_self_loop_if_not_in(aug_graph)
    return aug_graph


def aug_drop_node_list(graph_list, labels, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        aug_graph = aug_drop_node(graph_list[i], labels[i], drop_percent)
        aug_list.append(aug_graph)
    return aug_list


def aug_random_walk(graph, root, drop_percent=0.2):
    aug_graph = copy.deepcopy(graph)
    # Reverse the graph to facilitate walking from the starting node
    rg = dgl.reverse(aug_graph, copy_ndata=False, copy_edata=False)
    num_edge = rg.number_of_edges()  # number of edges of one graph
    retain_num = num_edge - int(num_edge * drop_percent)  # number of retain edges
    # Perform a random walk and return edge IDs
    trace = dgl.sampling.random_walk(rg, [root], length=retain_num, return_eids=False)[0]
    nodes = trace.flatten().unique()
    subgraph = dgl.node_subgraph(aug_graph, nodes, store_ids=False)
    subgraph = add_self_loop_if_not_in(subgraph)
    return subgraph



def aug_loss_modality(graph, drop_percent=0.2):
    aug_graph = copy.deepcopy(graph)
    num = aug_graph.number_of_nodes()  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = copy.deepcopy(graph)
    all_node_list = [i for i in range(num)]
    loss_node_list = random.sample(all_node_list, drop_num)
    metrics, traces, logs = aug_graph.ndata['metrics'], aug_graph.ndata['traces'], aug_graph.ndata['logs']
    for node in loss_node_list:
        # randomly remove one modality
        i = random.randint(0,2)
        if i == 0:
            metrics[node] = torch.zeros(metrics.shape[1])
        elif i == 1:
            traces[node] = torch.zeros(traces.shape[1])
        else:
            logs[node] = torch.zeros(logs.shape[1])
    aug_graph.ndata['metrics'] = metrics
    aug_graph.ndata['traces'] = traces
    aug_graph.ndata['logs'] = logs

    aug_graph = add_self_loop_if_not_in(aug_graph)
    return aug_graph


    # # set feats to zero if the node is not in the subgraph
    # subg_nodes = subgraph.nodes().tolist()
    # all_nodes = aug_graph.nodes().tolist()
    # not_in_subg_nodes = list(set(all_nodes) - set(subg_nodes))
    # metrics, traces, logs = aug_graph.ndata['metrics'], aug_graph.ndata['traces'], aug_graph.ndata['logs']
    # for node in not_in_subg_nodes:
    #     # randomly remove one modality
    #     i = random.randint(0,2)
    #     if i == 0:
    #         metrics[node] = torch.zeros(metrics.shape[1])
    #     elif i == 1:
    #         traces[node] = torch.zeros(traces.shape[1])
    #     else:
    #         logs[node] = torch.zeros(logs.shape[1])
    # aug_graph.ndata['metrics'] = metrics
    # aug_graph.ndata['traces'] = traces
    # aug_graph.ndata['logs'] = logs
    


def aug_random_walk_list(graph_list, labels, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        sub_graph = aug_random_walk(graph_list[i], labels[i], drop_percent)
        aug_list.append(sub_graph)
    return aug_list


def add_self_loop_if_not_in(graph):
    in_degrees = graph.in_degrees()
    zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
    for node in zero_indegree_nodes:
        graph.add_edges(node, node)
    return graph


def aug_importance_aware_drop(graph, root, drop_percent=0.2, use_degree=True, use_distance=True):
    """
    参数化重要性感知节点删除
    
    支持四种增强策略:
    - use_degree=True, use_distance=False: 仅基于度数重要性
    - use_degree=False, use_distance=True: 仅基于距离根因的重要性
    - use_degree=True, use_distance=True: 度数+距离综合重要性（推荐）
    - use_degree=False, use_distance=False: 随机删除
    
    Args:
        graph: DGL图对象
        root: 根因节点ID
        drop_percent: 删除节点的比例
        use_degree: 是否考虑节点度数（入度+出度）
        use_distance: 是否考虑距离根因的距离
        
    Returns:
        aug_graph: 增强后的图
    """
    
    num = graph.number_of_nodes()
    drop_num = int(num * drop_percent)
    aug_graph = copy.deepcopy(graph)
    
    # 如果两个参数都为False，使用随机删除
    if not use_degree and not use_distance:
        print(f"[随机增强] 两个参数都为False，使用随机删除")
        all_node_list = [i for i in range(num) if i != root]
        if len(all_node_list) >= drop_num:
            drop_node_list = np.random.choice(all_node_list, size=drop_num, replace=False)
            aug_graph.remove_nodes(drop_node_list.tolist())
        aug_graph = add_self_loop_if_not_in(aug_graph)
        return aug_graph
    
    # 计算重要性
    importance = np.zeros(num, dtype=float)
    strategy_name = []
    
    # 1. 度数重要性
    if use_degree:
        adj_matrix = graph.adj().to_dense().numpy()
        in_degree = np.sum(adj_matrix, axis=0)   # 入度 (被依赖数)
        out_degree = np.sum(adj_matrix, axis=1)  # 出度 (依赖数)
        
        # 度数重要性 = 入度权重高 + 出度权重低
        degree_importance = 0.7 * in_degree + 0.3 * out_degree
        importance += degree_importance
        strategy_name.append("度数")
    
    # 2. 距离重要性
    if use_distance:
        distances = _compute_distances_from_root(graph, root)
        
        # 距离保护权重 (距离越近权重越高)
        distance_protection = np.zeros(num, dtype=float)
        for i in range(num):
            if distances[i] == -1:  # 不可达
                distance_protection[i] = 0.0
            elif distances[i] == 0:  # 根因
                distance_protection[i] = 3.0
            elif distances[i] == 1:  # 直接邻居
                distance_protection[i] = 2.0  
            elif distances[i] == 2:  # 二度邻居
                distance_protection[i] = 1.0
            else:  # 远距离
                distance_protection[i] = 0.5
        
        importance += distance_protection
        strategy_name.append("距离")
    
    # 归一化重要性
    if np.max(importance) > 0:
        importance = importance / (np.max(importance) + 1e-8)
    
    # 计算删除概率 (重要性越高，删除概率越低)
    drop_prob = 1.0 - importance
    drop_prob[root] = 0.0  # 根因节点不删除
    
    # 根据概率选择要删除的节点
    available_nodes = [i for i in range(num) if i != root]
    if len(available_nodes) >= drop_num:
        # 按概率加权采样
        weights = drop_prob[available_nodes]
        weights = weights / (weights.sum() + 1e-8)
        
        drop_indices = np.random.choice(
            available_nodes, 
            size=drop_num, 
            replace=False, 
            p=weights
        )
        
        strategy_desc = "+".join(strategy_name)
        # print(f"[{strategy_desc}重要性感知增强] 删除 {len(drop_indices)} 个低重要性节点: {drop_indices.tolist()}")
        aug_graph.remove_nodes(drop_indices.tolist())
    
    # 确保连通性
    aug_graph = add_self_loop_if_not_in(aug_graph)
    
    return aug_graph


def _compute_distances_from_root(graph, root):
    """
    计算从根因节点到所有其他节点的最短路径距离
    
    Args:
        graph: DGL图对象
        root: 根因节点ID
        
    Returns:
        distances: 长度为N的列表，distances[i]是节点i到root的距离，-1表示不可达
    """
    from collections import deque
    
    num = graph.number_of_nodes()
    adj_matrix = graph.adj().to_dense().numpy()
    
    distances = [-1] * num  # -1表示不可达
    distances[root] = 0
    
    queue = deque([root])
    
    # BFS计算最短路径
    while queue:
        current = queue.popleft()
        
        # 检查所有邻居 (双向：in和out)
        for neighbor in range(num):
            if distances[neighbor] == -1:  # 未访问
                # 检查是否有边连接 (任意方向)
                if adj_matrix[current][neighbor] == 1 or adj_matrix[neighbor][current] == 1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    
    return distances