# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 3:38 下午
import os

from algorithm.utility import MatchingMethod
from env.order import Order
from env.vehicle import Vehicle
from env.network import Network
from env.location import VehicleLocation
from setting import FLOAT_ZERO, VALUE_EPS, INT_ZERO, VEHICLE_SPEED, TIME_SLOT, GAMMA, LONG_TERM
from setting import LEARNING_RESULT_FILE
from utility import is_enough_small
from learning.run import LafVehicleState, VehicleState, LevelVehicleState
import time
import pickle
from collections import defaultdict, Counter
from queue import Queue
from typing import Set, List, Tuple, Dict, NoReturn
import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class BipartiteGraph:

    __slots__ = ["pair_social_welfare_matrix", "bids", "vehicle_link_orders", "order_link_vehicles",
                 "vehicle_number", "order_number", "index2order", "order2index", "index2vehicle",
                 "vehicle2index", "deny_vehicle_index", "deny_order_index", "backup_sw_line"]

    def __init__(self, feasible_vehicles: Set[Vehicle], feasible_orders: Set[Order]):
        # 内部与外面的对接
        self.index2vehicle = [vehicle for vehicle in feasible_vehicles]
        self.vehicle2index = {vehicle: i for i, vehicle in enumerate(self.index2vehicle)}
        self.index2order = [order for order in feasible_orders]
        self.order2index = {order: i for i, order in enumerate(self.index2order)}
        self.vehicle_link_orders = defaultdict(set)
        self.order_link_vehicles = defaultdict(set)
        self.vehicle_number = len(feasible_vehicles)
        self.order_number = len(feasible_orders)
        self.pair_social_welfare_matrix = np.zeros(shape=(self.vehicle_number, self.order_number))
        self.deny_vehicle_index = -2  # 初始化没有该禁止车辆
        self.deny_order_index = -2  # 初始化没有静止订单
        self.backup_sw_line = None
        self.bids = None

    def get_vehicle_order_pair_bid(self, vehicle: Vehicle, order: Order):
        return self.bids[vehicle][order]

    def temporarily_remove_vehicle(self, vehicle: Vehicle):
        """
        暂时的剔除车辆
        :param vehicle:
        :return:
        """
        vehicle_index = self.vehicle2index[vehicle]
        self.deny_vehicle_index = vehicle_index  # 不允许匹配的车辆
        self.backup_sw_line = self.pair_social_welfare_matrix[vehicle_index, :].copy()  # 先备份后删除
        self.pair_social_welfare_matrix[vehicle_index, :] = FLOAT_ZERO

    def temporarily_remove_order(self, order: Order):
        """
        暂时剔除订单  !!!和暂时剔除车辆不能同时使用!!!
        :param order:
        :return:
        """
        order_index = self.order2index[order]
        self.deny_order_index = order_index
        self.backup_sw_line = self.pair_social_welfare_matrix[:, order_index].copy()  # 先备份后删除
        self.pair_social_welfare_matrix[:, order_index] = FLOAT_ZERO

    def recovery_remove_order(self):
        """
        修复剔除订单带来的影响   !!!和恢复剔除车辆不能同时使用!!!
        :return:
        """
        self.pair_social_welfare_matrix[:, self.deny_order_index] = self.backup_sw_line
        self.backup_sw_line = None
        self.deny_order_index = -2

    def recovery_remove_vehicle(self):
        """
        修复剔除车辆带来的影响
        :return:
        """
        self.pair_social_welfare_matrix[self.deny_vehicle_index, :] = self.backup_sw_line
        self.backup_sw_line = None
        self.deny_vehicle_index = -2

    def add_edge(self, vehicle: Vehicle, order: Order, pair_social_welfare: float):
        self.vehicle_link_orders[vehicle].add(order)
        self.order_link_vehicles[order].add(vehicle)
        vehicle_index = self.vehicle2index[vehicle]
        order_index = self.order2index[order]
        self.pair_social_welfare_matrix[vehicle_index, order_index] = pair_social_welfare  # 在没有相应state时，此处就是order.fare - cost

    def get_sub_graph(self, st_order: Order, covered_vehicles: Set[Vehicle], covered_orders: Set[Order]):
        temp_vehicle_set = set()
        temp_order_set = set()
        bfs_queue = Queue()
        bfs_queue.put(st_order)
        temp_order_set.add(st_order)
        covered_orders.add(st_order)

        while not bfs_queue.empty():
            node = bfs_queue.get()
            if isinstance(node, Vehicle):
                for order in self.vehicle_link_orders[node]:
                    if order not in covered_orders:
                        covered_orders.add(order)
                        temp_order_set.add(order)
                        bfs_queue.put(order)
            else:
                for vehicle in self.order_link_vehicles[node]:
                    if vehicle not in covered_vehicles:
                        covered_vehicles.add(vehicle)
                        temp_vehicle_set.add(vehicle)
                        bfs_queue.put(vehicle)

        cls = type(self)
        sub_graph = cls(temp_vehicle_set, temp_order_set)
        for order in temp_order_set:
            for vehicle in self.order_link_vehicles[order]:
                if vehicle in temp_vehicle_set:
                    vehicle_index = self.vehicle2index[vehicle]
                    order_index = self.order2index[order]
                    sub_graph.add_edge(vehicle, order, self.pair_social_welfare_matrix[vehicle_index, order_index])
        sub_graph.bids = self.bids
        return sub_graph

    def get_sub_graphs(self):
        covered_vehicles = set()  # 已经覆盖了的车辆
        covered_orders = set()  # 已经覆盖了的订单

        for order in self.order2index:
            if order not in covered_orders:
                sub_graph = self.get_sub_graph(order, covered_vehicles, covered_orders)  # 构建子图
                yield sub_graph
                if sub_graph.order_number == self.order_number and sub_graph.vehicle_number == self.vehicle_number:  # 没有必要往后探索
                    break

    def maximal_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        """
        :param return_match: 如果返回匹配关系就要将订单和车辆的匹配对列表
        :return order_match: 订单和车辆的匹配对
        :return social_welfare: 社会福利
        """
        raise NotImplementedError


class MaximumWeightMatchingGraph(BipartiteGraph):
    __slots__ = []

    def __init__(self, feasible_vehicles: Set[Vehicle], feasible_orders: Set[Order]):
        super(MaximumWeightMatchingGraph, self).__init__(feasible_vehicles, feasible_orders)

    def maximal_weight_matching(self, return_match=False) -> Tuple[float, List[Tuple[Vehicle, Order]]]:
        row_index, col_index = linear_sum_assignment(-self.pair_social_welfare_matrix)  # linear_sum_assignment 只可以解决最小值问题，本问题是最大值问题所以这样处理
        social_welfare = self.pair_social_welfare_matrix[row_index, col_index].sum()
        match_pairs = []
        if return_match:
            for vehicle_index, order_index in zip(row_index, col_index):
                if vehicle_index == self.deny_vehicle_index or order_index == self.deny_order_index:
                    continue
                vehicle = self.index2vehicle[vehicle_index]
                order = self.index2order[order_index]
                if order in self.vehicle_link_orders[vehicle]:
                    match_pairs.append((vehicle, order))
        return social_welfare, match_pairs


class FairClusterLevelVehicleStateMatchingMethod(MatchingMethod):
    """
    使用二部图匹配决定分配，利用vcg价格进行支付，主要基vcg机制理论, 最大化社会福利 pair_social_welfare = sum{order.order_fare} - sum{bid.additional_cost}
    """

    __slots__ = ["graph"]

    def __init__(self):
        super(FairClusterLevelVehicleStateMatchingMethod, self).__init__()

    @staticmethod
    def _build_graph(vehicles: List[Vehicle], orders: Set[Order],
                     current_time: int, network: Network, graph_type, near_zones, states) -> BipartiteGraph:
        """
        :param vehicles:
        :param orders:
        :param current_time:
        :param network:
        :param graph_type:
        :param args: 是否为长期收益 长期收益时此处传入Dict[VehicleState]字典
        :return:
        """
        # print("len = ",len(states))
        feasible_vehicles = set()
        feasible_orders = set()
        bids = dict()
        for vehicle in vehicles:
            if vehicle.have_service_mission:
                continue
            order_bids = vehicle.get_costs(orders, current_time, network)
            order_bids = {order: cost for order, cost in order_bids.items() if is_enough_small(cost, order.order_fare)}  # 这里可以保证订单的合理性
            if len(order_bids) > 0:
                feasible_vehicles.add(vehicle)
                bids[vehicle] = order_bids
            for order in order_bids:
                feasible_orders.add(order)
        graph = graph_type(feasible_vehicles, feasible_orders)
        for vehicle, order_bids in bids.items():
            for order, cost in order_bids.items():
                if states:
                    # print("states")
                    # 二部图权重 长期收益情况下二部图权重改为r^t*V(s') - v(s) + R_r
                    r = order.order_fare - cost
                    delta_t = int(np.ceil(
                        (network.get_shortest_distance(vehicle.location, order.pick_location) + order.order_distance) /
                        VEHICLE_SPEED /
                        TIME_SLOT
                    ))   # 从当前位置 -> 接到乘客 -> 送达乘客 所需时间 单位为/time_slot
                    vehicle_state = LevelVehicleState(vehicle.location, vehicle.income_level)
                    vehicle_near = near_zones[vehicle.location.osm_index]
                    s0 = 0
                    value = 0
                    if vehicle_state in states:
                        s0 += states[vehicle_state]
                    for near in vehicle_near:
                        if LevelVehicleState(VehicleLocation(near), vehicle.income_level) in states:
                            value += states[LevelVehicleState(VehicleLocation(near), vehicle.income_level)]

                    v0 = (s0 + value) / (1 + len(vehicle_near))  # smoothed_value
                    # v0 = s0
                    _location = VehicleLocation(order.drop_location.osm_index)
                    vehicle_state_ = LevelVehicleState(_location, vehicle.income_level)
                    order_near = near_zones[order.drop_location.osm_index]
                    s1 = 0
                    value1 = 0
                    if vehicle_state_ in states:
                        s1 += states[vehicle_state_]
                    for near in order_near:
                        if LevelVehicleState(VehicleLocation(near), vehicle.income_level) in states:
                            value1 += states[LevelVehicleState(VehicleLocation(near), vehicle.income_level)]

                    v1 = (s1 + value1) / (1 + len(order_near))  # smoothed_value
                    # v1 = s1
                    # 如果状态没有在字典中记录，则相应值为0
                    a = r + np.power(GAMMA, delta_t) * v1 - v0

                    graph.add_edge(vehicle, order, a)
                else:
                    graph.add_edge(vehicle, order, order.order_fare - cost)
        graph.bids = bids

        return graph


    def result_saving(self, bipartite_graph: BipartiteGraph, match_pairs: List[Tuple[Vehicle, Order]], social_welfare: float):
        for winner_vehicle, corresponding_order in match_pairs:
            # 计算VCG价格
            cost = bipartite_graph.get_vehicle_order_pair_bid(winner_vehicle, corresponding_order)

            # cost 保证平台收益不为负数
            passenger_payment = corresponding_order.order_fare

            # print(passenger_payment)
            # 保存结果
            self._matched_vehicles.add(winner_vehicle)
            self._matched_orders.add(corresponding_order)
            self._matched_results[winner_vehicle].set_order(corresponding_order, passenger_payment, 0)
            self._matched_results[winner_vehicle].set_vehicle(cost)
            self._matched_results[winner_vehicle].set_income(passenger_payment-cost)
            self._matched_results[winner_vehicle].set_per_hour_income(passenger_payment - cost)  # 需要清零
            self._matched_results[winner_vehicle].set_route([corresponding_order.pick_location, corresponding_order.drop_location])
            self._social_cost += cost
            self._total_driver_costs += cost
            self._passenger_payment += passenger_payment
            self._passenger_utility += 0
            # self._total_driver_payoffs += driver_payoff
            self._platform_profit += (passenger_payment - cost)
            self._social_welfare += corresponding_order.order_fare - cost
        # self._social_welfare += social_welfare  # 这里不能直接写+= social_welfare long term的情况下 sw为长期的


    def run_cluster(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, near_zones, states,clusters) -> NoReturn:
        self.reset()  # 清空结果
        # 构建图
        t1 = time.time()
        # print(" \n clusters = ", clusters,"len(orders) = ",len(orders),"len(vehicles) = ",len(vehicles),"\n")

        for l in clusters:
            # print("l = ",l)
            round_vehicle = []
            for v in vehicles:
                if v.have_service_mission:
                    continue
                if v.clusters == l:
                    round_vehicle.append(v)
            main_graph = self._build_graph(round_vehicle, orders, current_time, network,
                                           MaximumWeightMatchingGraph,
                                           near_zones, states)
            matchedOrders = []
            matching = []
            for sub_graph in main_graph.get_sub_graphs():
                sub_social_welfare, sub_match_pairs = sub_graph.maximal_weight_matching(
                    return_match=True)  # 胜者决定
                # print(sub_match_pairs)
                self.result_saving(sub_graph, sub_match_pairs, sub_social_welfare)  # 统计结果
                # matching.append(sub_match_pairs)
                for ve, ord in sub_match_pairs:
                    matchedOrders.append(ord)
            matchedOrders = set(matchedOrders)
            orders = orders - matchedOrders
            # print("匹配后orders num = ", len(orders))

        self._running_time = (time.time() - t1)


fair_cluster_levelvehicle_state_matching_method = FairClusterLevelVehicleStateMatchingMethod()
