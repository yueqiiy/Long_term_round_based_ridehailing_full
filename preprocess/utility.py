# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/2 10:04 上午

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from typing import List, Dict, Tuple


# 计算每个区域的邻近区域
def compute_near_zone():
    manhattan_zone_id = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74,
                         75, 79, 87, 88, 90, 100, 107, 113, 114, 116,
                         120, 125, 127, 128, 137, 140, 141, 142, 143,
                         144, 148, 151, 152, 158, 161, 162, 163, 164,
                         166, 170, 186, 209, 211, 224, 229, 230, 231,
                         232, 233, 234, 236, 237, 238, 239, 243, 244,
                         246, 249, 261, 262, 263]
    near_zone = dict()
    near_zone[4] = [79, 148, 224, 232]
    near_zone[12] = [13, 88, 261]
    near_zone[13] = [12, 231, 261]
    near_zone[24] = [41, 43, 151, 166]
    near_zone[41] = [24, 42, 43, 74, 75, 152, 166]
    near_zone[42] = [41, 74, 116, 120, 152, 166]
    near_zone[43] = [24, 41, 75, 236, 237, 163, 142, 239, 238, 151]
    near_zone[45] = [144, 148, 209, 231, 232]
    near_zone[48] = [50, 143, 142, 163, 230, 100, 68, 246]
    near_zone[50] = [143, 142, 48, 68, 246]
    near_zone[68] = [246, 48, 100, 186, 90, 249, 158]
    near_zone[74] = [42, 41, 75]
    near_zone[75] = [74, 41, 43, 236, 263, 262]
    near_zone[79] = [107, 224, 4, 232, 148, 144, 114, 113, 234]
    near_zone[87] = [209, 261, 88]
    near_zone[88] = [87, 261, 12]
    near_zone[90] = [68, 186, 234, 113, 249]
    near_zone[100] = [48, 230, 161, 164, 186, 68]
    near_zone[107] = [164, 170, 137, 224, 79, 113, 234]
    near_zone[113] = [249, 90, 234, 107, 79, 114]
    near_zone[114] = [249, 113, 79, 148, 144, 211, 125]
    near_zone[116] = [244, 120, 42, 152]
    near_zone[120] = [127, 243, 244, 116, 42]
    near_zone[125] = [158, 249, 114, 211, 231]
    near_zone[127] = [128, 243, 120]
    near_zone[128] = [127, 243]
    near_zone[137] = [233, 170, 107, 224]
    near_zone[140] = [262, 263, 141, 229]
    near_zone[141] = [237, 236, 263, 262, 140, 229, 162]
    near_zone[142] = [143, 239, 43, 163, 48, 50]
    near_zone[143] = [239, 142, 48, 50]
    near_zone[144] = [211, 114, 79, 148, 45, 231]
    near_zone[148] = [144, 114, 79, 4, 232, 45]
    near_zone[151] = [24, 43, 238]
    near_zone[152] = [116, 42, 41, 166]
    near_zone[158] = [246, 68, 249, 125]
    near_zone[161] = [230, 163, 162, 170, 164, 100]
    near_zone[162] = [161, 163, 237, 141, 229, 233, 170]
    near_zone[163] = [142, 43, 237, 162, 161, 230, 48]
    near_zone[164] = [100, 230, 161, 170, 107, 234, 186]
    near_zone[166] = [152, 42, 41, 24]
    near_zone[170] = [164, 161, 162, 233, 137, 107, 234]
    near_zone[186] = [68, 100, 164, 234, 90]
    near_zone[209] = [261, 231, 45, 87]
    near_zone[211] = [125, 249, 114, 144, 231]
    near_zone[224] = [137, 107, 79, 4]
    near_zone[229] = [140, 141, 237, 162, 233]
    near_zone[230] = [48, 163, 161, 164, 100]
    near_zone[231] = [125, 211, 144, 45, 209, 261, 13]
    near_zone[232] = [4, 79, 148, 45]
    near_zone[233] = [229, 162, 170, 137]
    near_zone[234] = [90, 186, 164, 170, 107, 79, 113, 249]
    near_zone[236] = [43, 75, 263, 141, 237]
    near_zone[237] = [43, 236, 141, 229, 162, 163]
    near_zone[238] = [151, 43, 239]
    near_zone[239] = [238, 43, 142, 143]
    near_zone[243] = [128, 127, 120, 244]
    near_zone[244] = [243, 120, 116]
    near_zone[246] = [50, 48, 68, 158]
    near_zone[249] = [158, 68, 90, 234, 113, 114, 125]
    near_zone[261] = [13, 231, 209, 87, 88, 12]
    near_zone[262] = [263, 141, 140]
    near_zone[263] = [75, 236, 141, 140, 262]
    with open('./network_data/near_zone.pkl', 'wb') as file:
        pickle.dump(near_zone, file)


# 返回区域的邻近区域
def near_zone_id(zone_id: int) -> List[int]:
    manhattan_zone_id = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74,
                             75, 79, 87, 88, 90, 100, 107, 113, 114, 116,
                             120, 125, 127, 128, 137, 140, 141, 142, 143,
                             144, 148, 151, 152, 158, 161, 162, 163, 164,
                             166, 170, 186, 209, 211, 224, 229, 230, 231,
                             232, 233, 234, 236, 237, 238, 239, 243, 244,
                             246, 249, 261, 262, 263]
    if zone_id not in manhattan_zone_id:
        return list()
    with open('./network_data/near_zone.pkl', 'rb') as file:
        near_zone_dict = pickle.load(file)
    return near_zone_dict[zone_id]


def compute_every_two_zone_distance():
    """
    计算任意两个区域之间的距离
    直接从历史信息中取平均
    """
    manhattan_zone_id_set = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75,
                             79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125,
                             127, 128, 137, 140, 141, 142, 143, 144, 148, 151,
                             152, 158, 161, 162, 163, 164, 166, 170, 186, 209,
                             211, 224, 229, 230, 231, 232, 233, 234, 236, 237,
                             238, 239, 243, 244, 246, 249, 261, 262, 263]
    weekend = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
    # print(manhattan_zone_id_set)
    # print(len(manhattan_zone_id_set))
    chunk_size = 100000
    distance_array = np.ones(shape=(264, 264), dtype=np.float) * -1
    # print(len(distance_array))
    distance_dict: Dict[Tuple:List] = dict()
    shortest_distance = np.ones(shape=(264, 264), dtype=np.float) * -1
    for i in range(1, 31):
        if i in weekend:
            continue
        for csv_iterator in pd.read_table("./raw_data/temp/Manhattan/order_data_{:03d}.csv".format(i), chunksize=chunk_size,
                                          iterator=True):
            for line in csv_iterator.values:
                s = line[0].split(',')
                zone_a = int(s[3])
                zone_b = int(s[4])
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
                distance_array[j][i] = np.mean(distance_dict[(i, j)])
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
    # with open('./network_data/shortest_distance.pkl', 'wb') as file:
    #     pickle.dump(shortest_distance, file)
    np.save("./network_data/shortest_distance", shortest_distance)


def compute_avg_vehicle_speed():
    """
    计算车辆的平均速度
    :return:
    """
    # 计算所有订单的总行程距离和总耗时
    weekend = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
    chunk_size = 100000
    total_distance = 0.0
    total_time = 0
    count = 0
    for i in range(1, 31):
        if i in weekend:
            continue
        for csv_iterator in pd.read_table("./raw_data/temp/Manhattan/order_data_{:03d}.csv".format(i), chunksize=chunk_size,
                                          iterator=True):
            for line in csv_iterator.values:
                s = line[0].split(',')
                distance = float(s[1])
                order_time = int(s[-1])
                total_distance += distance
                total_time += order_time
                count += 1
    avg_speed = total_distance / total_time
    avg_time = total_time / count
    print(avg_speed)
    print(avg_time)
    with open('./network_data/avg_speed.pkl', 'wb') as file:
        pickle.dump(avg_speed, file)

def compute_same_zone_distance():
    """
    计算起始点和终点在同一个区域订单的距离  作为车辆在同一个区域内移动距离的参考
    :return:
    """
    manhattan_zone_id_set = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75,
                             79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125,
                             127, 128, 137, 140, 141, 142, 143, 144, 148, 151,
                             152, 158, 161, 162, 163, 164, 166, 170, 186, 209,
                             211, 224, 229, 230, 231, 232, 233, 234, 236, 237,
                             238, 239, 243, 244, 246, 249, 261, 262, 263]
    weekend = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
    chunk_size = 100000
    distance_array = np.ones(shape=(264, 1), dtype=np.float) * -1
    distance_dict: Dict[int:List] = dict()
    shortest_distance = np.ones(shape=(264, 1), dtype=np.float) * -1
    for i in range(1, 31):
        if i in weekend:
            continue
        for csv_iterator in pd.read_table("./raw_data/temp/Manhattan/same_zone/order_data_{:03d}.csv".format(i),
                                          chunksize=chunk_size,
                                          iterator=True):
            for line in csv_iterator.values:
                s = line[0].split(',')
                zone_a = int(s[3])
                zone_b = int(s[4])
                if zone_a != zone_b:
                    continue
                distance = float(s[1])
                if zone_a not in distance_dict:
                    distance_dict[zone_a] = list()
                distance_dict[zone_a].append(distance)
    count = 0
    print(len(manhattan_zone_id_set))
    for i in manhattan_zone_id_set:
        if i in distance_dict.keys():
            count += 1
            distance_array[i] = np.mean(distance_dict[i])
        else:
            print(i, "not exist")
    # print(count)
    # print(distance_dict.keys())
    np.save("./network_data/same_zone_distance", distance_array)

if __name__ == "__main__":
    # compute_same_zone_distance()
# compute_avg_vehicle_speed()
    compute_every_two_zone_distance()
