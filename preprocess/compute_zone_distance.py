# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/10 10:11 上午

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from typing import List, Dict, Tuple


def compute_every_two_zone_distance():
    """
    计算任意两个区域之间的距离
    """
    # 63个地区
    manhattan_zone_id_set = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75,
                             79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125,
                             127, 128, 137, 140, 141, 142, 143, 144, 148, 151,
                             152, 158, 161, 162, 163, 164, 166, 170, 186, 209,
                             211, 224, 229, 230, 231, 232, 233, 234, 236, 237,
                             238, 239, 243, 244, 246, 249, 261, 262, 263]
    weekend = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
    chunk_size = 100000
    distance_array = np.ones(shape=(264, 264), dtype=np.float) * -1
    distance_dict: Dict[Tuple:List] = dict()
    shortest_distance = np.ones(shape=(264, 264), dtype=np.float) * -1
    for i in range(1, 31):
        if i in weekend:
            continue
        for csv_iterator in pd.read_table("./raw_data/temp/Manhattan/order_data_{:03d}.csv".format(i), chunksize=chunk_size, #  chunksize分块读入
                                          iterator=True):
            for line in csv_iterator.values:
                s = line[0].split(',')
                zone_a = int(s[3])
                zone_b = int(s[4])
                if zone_a == zone_b:
                    print("exist same zone transition : ", zone_a, zone_b)
                distance = float(s[1])
                if (zone_a, zone_b) not in distance_dict:
                    distance_dict[(zone_a, zone_b)] = list()
                    distance_dict[(zone_b, zone_a)] = list()
                distance_dict[(zone_a, zone_b)].append(distance)
                distance_dict[(zone_b, zone_a)].append(distance)
    for i in manhattan_zone_id_set:
        for j in manhattan_zone_id_set:
            if i == j:
                continue
            if (i, j) in distance_dict:
                distance_array[i][j] = np.mean(distance_dict[(i, j)])
                distance_array[j][i] = np.mean(distance_dict[(i, j)])  # 求平均值
    # 不在的订单内的两区域距离通过最短路径算法得到
    with open('./network_data/near_zone.pkl', 'rb') as file:
        near_zone_dict = pickle.load(file)
    # 建立图
    graph = nx.Graph()
    graph.add_nodes_from(manhattan_zone_id_set)
    for node in manhattan_zone_id_set:
        near_zone_set = near_zone_dict[node]
        for near_node in near_zone_set:
            if distance_array[node][near_node] != -1:
                graph.add_edge(node, near_node, weight=distance_array[node][near_node])
            else:
                print(node, near_node)
                print("相邻的订单居然没有")
    # 计算每两个区域的最短路径
    for i in manhattan_zone_id_set:
        for j in manhattan_zone_id_set:
            if i == j:
                continue
            shortest_distance[i][j] = nx.dijkstra_path_length(graph, i, j)
    with open('./network_data/shortest_distance.pkl', 'wb') as file:
        pickle.dump(shortest_distance, file)

compute_every_two_zone_distance()
