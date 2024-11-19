import time

import numpy as np
import pickle
import os
from typing import List, Dict, NoReturn, Set
import sys

from algorithm.LongTerm_ILP import longterm_ilp_matching_method

sys.path.append("/data/yueqi/Long_term_round_based_ridehailing_full")
from matplotlib import pyplot as plt

from agent.platform import Platform

from algorithm.ILP import ilp_matching_method
from algorithm.fair_cluster_levelvehicle_state_matching_method import \
    fair_cluster_levelvehicle_state_matching_method

from algorithm.laf_matching_method import laf_matching_method
from algorithm.reassign.reassign import reassign_method
from algorithm.worst_first import worst_first_method
from env.vehicle import Vehicle, generate_road_vehicles_data, generate_grid_vehicles_data
from env.location import *
from algorithm.nearest_matching_method import nearest_matching_method
from setting import GEO_DATA_FILE, GEO_NAME, SAVE_LEARNING_RESULT_FILES, TYPED_LEVEL_LEARNING_RESULT_FILE, \
    FAIR_DISPATCH, LafMatchingMethod, LAF_DISPATCH, FLOAT_ZERO, ReassignMethod, \
    FairClusterLevelVehicleStateMatchingMethod, \
    FAIR_LEVEL_DISPATCH, ILPMatchingMethod, WorstFirstMethod, FairNoClusterMatchingMethod, NEW_FAIR_MATCH, \
    LongTermILPMatchingMethod, JainILPMatchingMethod, LongTermMeanILPMatchingMethod, Hungarian_Algorithm
from setting import ROAD_MODE, GRID_MODE
from setting import MATCHING_METHOD, EXPERIMENTAL_MODE, NEAREST_MATCHING_METHOD
from setting import TIME_SLOT, VEHICLE_NUMBER, MIN_REQUEST_TIME, INT_ZERO, VEHICLE_SPEED, MAX_REQUEST_TIME, MAX_REPEATS
from setting import LONG_TERM, LEARNING, GAMMA, TYPED, TYPED_LEARNING_RESULT_FILE, \
    LEARNING_RESULT_FILE, EMPTY_VEHICLE_DISPATCH, DISPATCH_METHOD
from env.graph import BaseGraph, generate_grid_graph, generate_road_graph
from env.order import Order, generate_road_orders_data
from env.network import Network
from learning.run import TypedVehicleState, VehicleState, LevelVehicleState, LafVehicleState
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
# import matplotlib.pyplot as plt

gamma = 0.9
beta = 0.025

class Simulator:
    __slots__ = [
        "network", "vehicles", "orders", "platform", "time_slot",
        "current_time",
        "social_welfare_trend", "social_cost_trend", "total_passenger_payment_trend", "total_passenger_utility_trend", "platform_profit_trend",
        "accumulate_service_ratio_trend", "total_orders_number_trend", "serviced_orders_number_trend",
        "empty_vehicle_number_trend", "total_vehicle_number_trend", "empty_vehicle_ratio_trend",
        "accumulate_service_distance_trend", "accumulate_random_distance_trend",
        "each_orders_service_time_trend", "each_orders_wait_time_trend",
        "each_vehicles_reward", "each_vehicles_cost", "each_vehicles_finish_order_number", "each_vehicles_service_distance", "each_vehicles_random_distance",
        "bidding_time_trend", "running_time_trend",
        # learning过程需要的参数
        "state_transition_dict",
        # planning过程需要的参数
        "states",
        # 空闲车辆调度时需要的参数，一个区域的邻近区域
        "near_zone",
        # 保存状态转移的V值
        "state_dict",
        # 保存每个车的收入情况
        "each_vehicles_income",
        "state_laf_dict",
        "median",
        "hour",
        "clusters",
        "jain",
        "clustersList",
        "running_time",
        "median_75",
        "JAIN_trend"
    ]

    def __init__(self):
        if EXPERIMENTAL_MODE == ROAD_MODE:
            BaseGraph.set_generate_graph_function(generate_road_graph)
            Vehicle.set_generate_vehicles_function(generate_road_vehicles_data)
            Order.set_order_generator(generate_road_orders_data)
        # elif EXPERIMENTAL_MODE == GRID_MODE:
        #     BaseGraph.set_generate_graph_function(generate_grid_graph)
        #     Vehicle.set_generate_vehicles_function(generate_grid_vehicles_data)
        #     Order.set_order_generator(generate_grid_orders_data)
        else:
            raise Exception("目前还没有实现其实验模式")

        if MATCHING_METHOD == NEAREST_MATCHING_METHOD:
            mechanism = nearest_matching_method
        elif MATCHING_METHOD == LafMatchingMethod:
            mechanism = laf_matching_method
        elif MATCHING_METHOD == ILPMatchingMethod:
            mechanism = ilp_matching_method
        elif MATCHING_METHOD == ReassignMethod:
            mechanism = reassign_method
        elif MATCHING_METHOD == WorstFirstMethod:
            mechanism = worst_first_method
        elif MATCHING_METHOD == FairClusterLevelVehicleStateMatchingMethod:
            mechanism = fair_cluster_levelvehicle_state_matching_method
        elif MATCHING_METHOD == LongTermILPMatchingMethod:
            mechanism = longterm_ilp_matching_method


        else:
            raise Exception("目前还没有实现其他类型的订单分配机制")

        network = Network(BaseGraph.generate_graph())
        platform = Platform(mechanism)

        # 初始化模拟器的变量
        self.platform: Platform = platform
        self.vehicles: List[Vehicle] = list()
        self.orders = None  # 实则是一个生成器
        self.network: Network = network
        self.time_slot: int = TIME_SLOT
        self.current_time = MIN_REQUEST_TIME
        self.social_welfare_trend = list()
        self.social_cost_trend = list()
        self.total_passenger_payment_trend = list()
        self.total_passenger_utility_trend = list()
        self.platform_profit_trend = list()
        self.total_orders_number_trend = list()
        self.serviced_orders_number_trend = list()
        self.accumulate_service_ratio_trend = list()
        self.empty_vehicle_number_trend = list()
        self.total_vehicle_number_trend = list()
        self.empty_vehicle_ratio_trend = list()
        self.accumulate_service_distance_trend = list()
        self.accumulate_random_distance_trend = list()
        self.each_orders_service_time_trend = list()
        self.each_orders_wait_time_trend = list()
        self.each_vehicles_cost = list()
        self.each_vehicles_finish_order_number = list()
        self.each_vehicles_service_distance = list()
        self.each_vehicles_random_distance = list()
        self.bidding_time_trend = list()
        self.running_time_trend = list()
        self.state_transition_dict = dict()
        self.states: Dict[VehicleState] = dict()
        self.near_zone: Dict = dict()
        self.state_dict: Dict[LevelVehicleState] = dict()
        self.each_vehicles_income = list()
        self.state_laf_dict: Dict[LafVehicleState] = dict()
        self.JAIN_trend = list()

        self.median = 0.1
        self.median_75 = 0.1
        self.hour = 0

        self.clusters = 1
        self.jain = 1.0
        self.clustersList = [1]

        self.running_time = FLOAT_ZERO

        if EMPTY_VEHICLE_DISPATCH:  # 调度空闲车辆
            geo_data_base_folder = GEO_DATA_FILE["base_folder"]
            near_zone_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["near_zone_file"])
            with open(near_zone_file, "rb") as file:
                self.near_zone = pickle.load(file)

    def create_vehicle_env(self, vehicles_data_save_file, vehicle_number=VEHICLE_NUMBER):
        """
        用于创造车辆环境
        """
        Vehicle.generate_vehicles_data(vehicle_number, self.network, vehicles_data_save_file)

    def create_order_env(self, orders_data_save_file):
        """
        用于创造订单环境
        """
        Order.generate_orders_data(orders_data_save_file, self.network)

    def load_env(self, vehicles_data_file, orders_data_file):
        """
        首先加载环境，然后
        :return:
        """

        self.vehicles = Vehicle.load_vehicles_data(VEHICLE_SPEED, self.time_slot, vehicles_data_file)
        self.orders = Order.load_orders_data(MIN_REQUEST_TIME, self.time_slot, orders_data_file)

    def save_simulate_result(self, file_name):
        import pickle
        result = [
            self.social_welfare_trend,
            self.social_cost_trend,
            self.total_passenger_payment_trend,
            self.total_passenger_utility_trend,
            self.platform_profit_trend,
            self.total_orders_number_trend,
            self.serviced_orders_number_trend,
            self.accumulate_service_ratio_trend,
            self.empty_vehicle_ratio_trend,
            self.bidding_time_trend,
            self.running_time_trend,
            self.accumulate_service_distance_trend,
            self.accumulate_random_distance_trend,
            self.each_orders_wait_time_trend,
            self.each_orders_service_time_trend,
            self.each_vehicles_cost,
            self.each_vehicles_finish_order_number,
            self.each_vehicles_service_distance,
            self.each_vehicles_random_distance,
            self.each_vehicles_income,
            self.JAIN_trend
        ]
        with open(file_name, "wb") as file:
            pickle.dump(result, file)

    def save_simulate_learning_result(self, file_name):
        import pickle
        with open(file_name, "wb") as file:
            pickle.dump(self.state_transition_dict, file)

    def save_simulate_level_learning_result(self, file_name):
        import pickle
        with open(file_name, "wb") as file:
            pickle.dump(self.state_dict, file)

    def save_simulate_VS_learning_result(self, file_name):
        import pickle
        print("======================true======================")
        with open(file_name, "wb") as file:
            pickle.dump(self.states, file)

    def save_simulate_laf_learning_result(self, file_name):
        import pickle
        print(file_name)
        with open(file_name, "wb") as file:
            pickle.dump(self.state_laf_dict, file)

    def reset(self):
        """
        一个模拟之前的整理工作, 将上一步结果清空
        """
        self.platform.reset()
        self.orders = None
        self.vehicles = None
        self.median = 0.1
        self.median_75 = 0.1
        self.hour = 0
        self.clusters = 1
        self.clustersList = [1]
        self.jain = 1.0
        self.social_welfare_trend.clear()
        self.social_cost_trend.clear()
        self.total_passenger_payment_trend.clear()
        self.total_passenger_utility_trend.clear()
        self.platform_profit_trend.clear()
        self.total_orders_number_trend.clear()
        self.serviced_orders_number_trend.clear()
        self.accumulate_service_ratio_trend.clear()
        self.empty_vehicle_number_trend.clear()
        self.total_vehicle_number_trend.clear()
        self.empty_vehicle_ratio_trend.clear()
        self.accumulate_service_distance_trend.clear()
        self.accumulate_random_distance_trend.clear()
        self.each_orders_service_time_trend.clear()
        self.each_orders_wait_time_trend.clear()
        self.each_vehicles_cost.clear()
        self.each_vehicles_finish_order_number.clear()
        self.each_vehicles_service_distance.clear()
        self.each_vehicles_random_distance.clear()
        self.bidding_time_trend.clear()
        self.running_time_trend.clear()
        self.state_transition_dict.clear()
        self.each_vehicles_income.clear()
        self.JAIN_trend.clear()

    def laf_simulate(self):
        # 用来计算静止成本
        same_zone_distance = os.path.join(GEO_DATA_FILE["base_folder"], GEO_DATA_FILE["same_zone_distance_file"])

        with open("../result/TYPED_LEVEL_States_Values_EMPTY.pkl", "rb") as file:
            self.state_laf_dict = pickle.load(file)
        if EMPTY_VEHICLE_DISPATCH:  # 调度空闲车辆
            geo_data_base_folder = GEO_DATA_FILE["base_folder"]
            near_zone_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["near_zone_file"])
            with open(near_zone_file, "rb") as file:
                self.near_zone = pickle.load(file)

        for current_time, new_orders in self.orders:  # orders 是一个生成器
            self.current_time = current_time
            t1 = time.time()
            self.platform.laf_round_based_process(self.vehicles, new_orders, self.current_time, self.network,
                                                  self.near_zone, self.state_laf_dict,self.median)

            self.summary_laf_state_transition_info()  # 保存状态转移并且进行值迭代更新
            if EMPTY_VEHICLE_DISPATCH:
                self.dispatch_empty_vehicle()

            self.trace_vehicles_info()  # 车辆更新信息
            self.summary_each_round_result(new_orders, same_zone_distance)  # 统计匹配结果

            self.running_time = (time.time() - t1)
            print(
                "at {0} social welfare {1:.2f} passenger payment {2:.2f} vehicle cost {3:.2f} platform profit {4:.2f} empty vehicle ratio {5:.4f} service ratio {6:.4f} bidding time {7:.4f} running time {8:.4f}".format(
                    current_time,
                    self.social_welfare_trend[-1],
                    self.total_passenger_payment_trend[-1],
                    self.social_cost_trend[-1],
                    self.platform_profit_trend[-1],
                    self.empty_vehicle_ratio_trend[-1],
                    self.accumulate_service_ratio_trend[-1],
                    self.bidding_time_trend[-1],
                    self.running_time_trend[-1],
                )
            )

        # 等待所有车辆完成订单之后结束
        self.current_time += self.time_slot
        self.finish_all_orders()

    def fair_simulate(self):
        # 用来计算静止成本
        same_zone_distance = os.path.join(GEO_DATA_FILE["base_folder"], GEO_DATA_FILE["same_zone_distance_file"])

        with open("../result/TYPED_LEVEL_States_Values_EMPTY.pkl", "rb") as file:
            self.state_dict = pickle.load(file)


        geo_data_base_folder = GEO_DATA_FILE["base_folder"]
        near_zone_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["near_zone_file"])
        with open(near_zone_file, "rb") as file:
            self.near_zone = pickle.load(file)
        cnt = 0
        for current_time, new_orders in self.orders:  # orders 是一个生成器
            cnt += 1
            self.current_time = current_time
            t1 = time.time()
            self.platform.round_cluster_based_process(self.vehicles, new_orders, self.current_time,  self.network,self.near_zone,
                                                      self.state_dict,self.clustersList)
            self.summary_online_level_state_transition_info() # 保存状态转移并且进行值迭代更新
            if EMPTY_VEHICLE_DISPATCH:
                self.dispatch_empty_vehicle()
            self.trace_vehicles_info()  # 车辆更新信息
            self.summary_each_round_result(new_orders, same_zone_distance)  # 统计匹配结果
            if cnt % 5 == 0:
                self.set_the_best_clustes_level()
                # self.clustering_drivers(2)
            self.running_time = (time.time() - t1)

            print(
                "at {0} social welfare {1:.2f} passenger payment {2:.2f} vehicle cost {3:.2f} platform profit {4:.2f} empty vehicle ratio {5:.4f} service ratio {6:.4f} bidding time {7:.4f} running time {8:.4f}".format(
                    current_time,
                    self.social_welfare_trend[-1],
                    self.total_passenger_payment_trend[-1],
                    self.social_cost_trend[-1],
                    self.platform_profit_trend[-1],
                    self.empty_vehicle_ratio_trend[-1],
                    self.accumulate_service_ratio_trend[-1],
                    self.bidding_time_trend[-1],
                    self.running_time_trend[-1],
                )
            )

        # 等待所有车辆完成订单之后结束
        self.current_time += self.time_slot
        self.finish_all_orders()

    def simulate(self):
        # 用来计算静止成本
        same_zone_distance = os.path.join(GEO_DATA_FILE["base_folder"], GEO_DATA_FILE["same_zone_distance_file"])
        if EMPTY_VEHICLE_DISPATCH:  # 调度空闲车辆
            geo_data_base_folder = GEO_DATA_FILE["base_folder"]
            near_zone_file = os.path.join(geo_data_base_folder, GEO_DATA_FILE["near_zone_file"])
            with open(near_zone_file, "rb") as file:
                self.near_zone = pickle.load(file)
        cnt = 0
        for current_time, new_orders in self.orders:  # orders 是一个生成器
            cnt += 1
            self.current_time = current_time

            t1 = time.time()
            self.platform.round_based_process(self.vehicles, new_orders, self.current_time, self.network)  # 订单分发和司机定价

            if EMPTY_VEHICLE_DISPATCH:
                self.dispatch_empty_vehicle()
            self.trace_vehicles_info()  # 车辆更新信息
            self.summary_each_round_result(new_orders, same_zone_distance)  # 统计匹配结果
            self.running_time = (time.time() - t1)
            print(
                "at {0} social welfare {1:.2f} passenger payment {2:.2f} vehicle cost {3:.2f} platform profit {4:.2f} empty vehicle ratio {5:.4f} service ratio {6:.4f} bidding time {7:.4f} running time {8:.4f}".format(
                    current_time,
                    self.social_welfare_trend[-1],
                    self.total_passenger_payment_trend[-1],
                    self.social_cost_trend[-1],
                    self.platform_profit_trend[-1],
                    self.empty_vehicle_ratio_trend[-1],
                    self.accumulate_service_ratio_trend[-1],
                    self.bidding_time_trend[-1],
                    self.running_time_trend[-1],
                )
            )

        # 等待所有车辆完成订单之后结束
        self.current_time += self.time_slot
        self.finish_all_orders()


    def New_Fair_dispatch(self):
        """
                对空闲车辆进行调度
                这里的调度成本需不需要计算
                :return:
                """
        # print("dispatch true")
        mechanism = self.platform.matching_method
        for vehicle in self.vehicles:
            # 车上有单 / 车正在进行调度 / 这里要排除上一次匹配获得订单的车
            if vehicle.have_service_mission or vehicle in mechanism.matched_vehicles:
                continue
            # 车辆长期处于静止状态 > 5个时间槽？？
            if vehicle.vehicle_type.idle_time >= 5:
                near_index_list = self.near_zone[vehicle.location.osm_index]
                # 计算到达相邻区域的时间
                near_index_lot = set()
                for ne in near_index_list:
                    near_index_lot.add(ne)
                    temp = self.near_zone[ne]
                    for t in temp:
                        near_index_lot.add(t)

                a_dict: Dict = dict()  # 邻近区域的权重字典
                for near_index in near_index_lot:
                    drop_location = DropLocation(near_index)
                    distance = self.network.get_shortest_distance(vehicle.location, drop_location)
                    cost = distance * vehicle.unit_cost
                    delta_t = int(np.ceil(
                        distance /
                        VEHICLE_SPEED /
                        TIME_SLOT
                    ))  # 从当前位置 -> 调度位置 所需时间 单位为/time_slot
                    vehicle_state = LevelVehicleState(vehicle.location, vehicle.income_level)
                    _location = VehicleLocation(drop_location.osm_index)
                    # 注意这里delta_t 要✖乘 time_Slot
                    vehicle_state_ = LevelVehicleState(_location, vehicle.income_level)
                    # 状态没有在字典中记录,则相应位置为0
                    a = 0
                    if vehicle_state in self.state_dict:
                        a -= self.state_dict[vehicle_state]
                    if vehicle_state_ in self.state_dict:
                        a += np.power(0.999, delta_t) * self.state_dict[vehicle_state_]
                    if a <= 0:
                        continue
                    # a_dict[near_index] = a / (distance + 1)
                    a_dict[near_index] = a
                if len(a_dict) == 0:
                    continue
                max_a_index = max(a_dict, key=a_dict.get)  # 获得最大长期收益对应的邻近区域序号
                if max_a_index == vehicle.location.osm_index:
                    continue
                drop_location = DropLocation(max_a_index)
                vehicle.set_route([drop_location])
                cost_time = np.ceil(self.network.get_shortest_distance(vehicle.location, drop_location) / VEHICLE_SPEED)
                vehicle.vehicle_type.available_time = self.current_time + cost_time
                vehicle.vehicle_type.idle_time = 0



    def LAF_dispatch(self):
        """
                对空闲车辆进行调度
                这里的调度成本需不需要计算
                :return:
                """
        mechanism = self.platform.matching_method
        drivers = self.vehicles
        drivers.sort(key=lambda x: x.income[-1])
        median = drivers[int(len(drivers) * 0.50)].income[-1]
        for vehicle in drivers:
            # 车上有单 / 车正在进行调度 / 这里要排除上一次匹配获得订单的车
            if vehicle.have_service_mission or vehicle in mechanism.matched_vehicles:
                continue
            # 车辆长期处于静止状态 > 5个时间槽？？
            if vehicle.vehicle_type.idle_time >= 5:
                olt = self.current_time - MIN_REQUEST_TIME
                ratio = vehicle.income[-1] / (olt + 0.1)
                if 6 <= ratio <= 12:
                    continue
                else:
                    best_grid_id, best_value = vehicle.location.osm_index, -100
                    current_state = LafVehicleState(vehicle.location)
                    current_value = 0
                    if current_state in self.state_laf_dict:
                        current_value = self.state_laf_dict[current_state]

                    near_index_list = self.near_zone[vehicle.location.osm_index]
                    # 计算到达相邻区域的时间
                    near_index_lot = set()
                    for ne in near_index_list:
                        near_index_lot.add(ne)
                        temp = self.near_zone[ne]
                        for t in temp:
                            near_index_lot.add(t)
                    for near_index in near_index_lot:
                        drop_location = DropLocation(near_index)
                        distance = self.network.get_shortest_distance(vehicle.location, drop_location)
                        delta_t = int(np.ceil(
                            distance /
                            VEHICLE_SPEED /
                            TIME_SLOT
                        ))  # 从当前位置 -> 调度位置 所需时间 单位为/time_slot
                        _location = VehicleLocation(drop_location.osm_index)
                        # 注意这里delta_t 要✖乘 time_Slot
                        vehicle_state_ = LafVehicleState(_location)
                        proposed_value = 0
                        if vehicle_state_ in self.state_laf_dict:
                            proposed_value = self.state_laf_dict[vehicle_state_]
                        # 状态没有在字典中记录,则相应位置为0
                        if ratio < 6:
                            incremental_value = np.power(0.999, delta_t) * proposed_value - current_value
                        else:
                            incremental_value = -abs(median - vehicle.income[-1] / (olt + 0.1) / 3600)
                        if incremental_value > best_value:
                            best_grid_id, best_value = near_index, incremental_value
                    drop_location = DropLocation(best_grid_id)
                    vehicle.set_route([drop_location])
                    cost_time = np.ceil(self.network.get_shortest_distance(vehicle.location, drop_location) / VEHICLE_SPEED)
                    vehicle.vehicle_type.available_time = self.current_time + cost_time
                    vehicle.vehicle_type.idle_time = 0

    def dispatch_empty_vehicle(self):
        if DISPATCH_METHOD == NEW_FAIR_MATCH:
            self.New_Fair_dispatch()
        elif DISPATCH_METHOD == LAF_DISPATCH:
            self.LAF_dispatch()


    def summary_state_transition_info(self):
        """
        large-scale中的方法 不考虑车辆的类型 奖励值直接为订单的价格
        :return:
        """
        mechanism = self.platform.matching_method
        state_list = list()
        # 静止的状态也要记录
        for vehicle in self.vehicles:
            s = VehicleState(self.current_time, vehicle.location)
            if vehicle in mechanism.matched_vehicles:
                order = mechanism.matched_results[vehicle].order
                cost = mechanism.matched_results[vehicle].driver_cost
                distance = cost / vehicle.unit_cost
                # 注意这个time不是以时间槽为单位的，但是必须是时间槽的整数倍
                time = int(np.ceil(distance / VEHICLE_SPEED / TIME_SLOT)) * TIME_SLOT
                _location = VehicleLocation(order.drop_location.osm_index)
                s_ = VehicleState(self.current_time + time, _location)
                r = order.order_fare
            else:
                s_ = VehicleState(self.current_time + TIME_SLOT, vehicle.location)
                r = 0
            state_list.append((s, r, s_))
        self.state_transition_dict[self.current_time] = state_list

    def summary_laf_state_transition_info(self):
        mechanism = self.platform.matching_method
        state_list = list()
        # 静止的状态也要记录
        for vehicle in self.vehicles:
            time = TIME_SLOT
            s = LafVehicleState(vehicle.location)
            if vehicle in mechanism.matched_vehicles:
                order = mechanism.matched_results[vehicle].order
                cost = mechanism.matched_results[vehicle].driver_cost
                distance = cost / vehicle.unit_cost
                # 注意这个time不是以时间槽为单位的，但是必须是时间槽的整数倍
                time = int(np.ceil(distance / VEHICLE_SPEED / TIME_SLOT)) * TIME_SLOT
                _location = VehicleLocation(order.drop_location.osm_index)
                s_ = LafVehicleState(_location)
                r = order.order_fare - cost
            else:
                s_ = LafVehicleState(vehicle.location)
                r = 0
            state_list.append((s, r, s_))
            if s not in self.state_laf_dict:
                self.state_laf_dict[s] = 0
            if s_ not in self.state_laf_dict:
                self.state_laf_dict[s_] = 0
            self.state_laf_dict[s] +=  beta * (np.power(gamma, time) * self.state_laf_dict[s_] + r - self.state_laf_dict[s])
        self.state_transition_dict[self.current_time] = state_list


    def summary_online_level_state_transition_info(self):
        mechanism = self.platform.matching_method
        state_list = list()
        # 静止的状态也要记录
        for vehicle in self.vehicles:
            time = TIME_SLOT
            s = LevelVehicleState(vehicle.location, vehicle.income_level)
            if vehicle in mechanism.matched_vehicles:
                order = mechanism.matched_results[vehicle].order
                cost = mechanism.matched_results[vehicle].driver_cost
                distance = cost / vehicle.unit_cost
                # 注意这个time不是以时间槽为单位的，但是必须是时间槽的整数倍
                time = int(np.ceil(distance / VEHICLE_SPEED / TIME_SLOT)) * TIME_SLOT
                _location = VehicleLocation(order.drop_location.osm_index)
                s_ = LevelVehicleState(_location, vehicle.income_level)
                r = order.order_fare - cost
            else:
                s_ = LevelVehicleState(vehicle.location, vehicle.income_level)
                r = 0
            state_list.append((s, r, s_))
            if s not in self.state_dict:
                self.state_dict[s] = 0
            if s_ not in self.state_dict:
                self.state_dict[s_] = 0
            # self.state_dict[s] += beta * (np.power(gamma, time) * self.state_dict[s_] + r - self.state_dict[s]) # 0.025   137060.6761476253  0.1  136185.40901313187
            self.state_dict[s] += 0.007 * (np.power(gamma, time) * self.state_dict[s_] + r - self.state_dict[s]) # 0.02   137739.12367626396  0.01   138052.20578717737  0.15  135106.03242812358
        self.state_transition_dict[self.current_time] = state_list
#     0.015 GAMMA= 0.9  137938.12437439276
#     0.015 GAMMA= 0.8  137939.78607082093
#     0.005 GAMMA = 0.99  138314.0857286135
#     0.005 GAMMA= 0.9 138740.3415098511
#     0.001 GAMMA = 0.9 138031.05347316182
#     0.001 GAMMA= 0.85 137339.17010000226
#     0.006 GAMMA= 0.9   138711.18073042063
#     0.007 GAMMA= 0.8   138660.61162063942
#     0.007 GAMMA= 0.9   139477.27422719361
#     0.007 GAMMA= 0.99   139459.14512892166
#     0.0075 GAMMA = 0.9  138734.03754076586
#     0.008 GAMMA= 0.9   138869.46970589214
#     0.003 GAMMA= 0.9   138207.41217349895
#     0.009 GAMMA= 0.9   139208.71207209118



    def trace_vehicles_info(self, print_vehicle=False) -> NoReturn:
        """
        更新车辆信息 由于中途不会接到订单，可以直接等到到达目的地时一次性完成更新 每个时间槽更新时判断车辆原有订单是否完成
        :return:
        """
        mechanism = self.platform.matching_method
        empty_vehicle_number = INT_ZERO
        total_vehicle_number = INT_ZERO
        if self.current_time % 3600 == 0 and self.current_time != MAX_REQUEST_TIME:
            print("*"*10+" new hour "+"*"*10)
            print("current_time = ",self.current_time)
            self.hour += 1
            for vehicle in self.vehicles: #记录下一个小时的收入
                vehicle.income.append(0)

        income = []
        sumIncome = 0

        inchour = []
        roundInc = []

        for vehicle in self.vehicles:
            if not vehicle.is_activated:
                # vehicle.enter_platform()  # TODO 日后去解决这个问题
                continue

            total_vehicle_number += 1
            # 如果车辆有订单，判断这个时间段能否到达
            # print(len(vehicle.route))
            if vehicle.have_service_mission:  # 上一轮次被匹配车辆的更新
                # print("f")
                # 在下一个时间段已经到达终点
                if self.current_time + TIME_SLOT >= vehicle.vehicle_type.available_time:
                    # 更新位置
                    # print(vehicle.location)
                    # print(vehicle.route)
                    vehicle.location.set_location(vehicle.route[-1].osm_index)
                    # print(vehicle.location)
                    vehicle.set_route([])
                    vehicle.vehicle_type.available_time = -1
            else:
                # 没有接到订单的静止
                if vehicle not in mechanism.matched_vehicles:
                    empty_vehicle_number += 1
                    if not vehicle.have_service_mission:
                        vehicle.increase_idle_time(1)
                else:
                    matched_result = mechanism.matched_results[vehicle]
                    matched_result.order.set_belong_vehicle(vehicle)
                    # 设置车辆路线；车辆成本自增；车辆分配订单数量自增
                    vehicle.set_route(matched_result.driver_route)
                    vehicle.set_vehicle_income(matched_result.driver_income) # 该小时收入
                    vehicle.set_vehicle_incomeacc(matched_result.driver_income)
                    vehicle.set_vehicle_roundIncome(matched_result.driver_income)

                    vehicle.set_vehicle_per_hour_income(matched_result.driver_income)

                    vehicle.increase_accumulated_cost(matched_result.driver_cost)
                    vehicle.increase_assigned_order_number(1)
                    vehicle.vehicle_type.idle_time = 0
                    # 计算订单等待时间和完成的时间
                    pick_up_time = np.round(
                        self.network.get_shortest_distance(vehicle.location, matched_result.order.pick_location)
                        / vehicle.vehicle_speed)
                    matched_result.order.real_pick_up_time = self.current_time + pick_up_time
                    matched_result.order.real_wait_time = self.current_time + pick_up_time - matched_result.order.request_time
                    service_time = np.round(matched_result.order.order_distance / vehicle.vehicle_speed)
                    vehicle.vehicle_type.available_time = self.current_time + pick_up_time + service_time
                    # 当前刚接到订单，但是在下一个时刻已经能够到达终点
                    if self.current_time + TIME_SLOT >= vehicle.vehicle_type.available_time:
                        # 更新位置
                        vehicle.location.set_location(vehicle.route[-1].osm_index)
                        vehicle.set_route([])
                        vehicle.vehicle_type.available_time = -1
            inchour.append(vehicle.income[-1])
            roundInc.append(vehicle.roundIncomeGet)
            income.append(vehicle.incomeacc)
            sumIncome += (vehicle.incomeacc) * (vehicle.incomeacc)

        self.jain = round(((np.sum(income)*np.sum(income))/(len(self.vehicles) * (sumIncome+1))), 2)
        print("总体标准差: ", np.std(income))
        print("jain = ", self.jain,"sumIncome = ", sumIncome,"np.sum(income) = ", np.sum(income))

        inchour.sort()
        medianhour = inchour[int(len(self.vehicles) / 2)]
        income.sort()
        self.median_75 = income[int(len(self.vehicles) * 0.5)]
        if medianhour == 0:
            medianhour = 0.1

        self.median = medianhour

        roundMedian = roundInc[int(len(self.vehicles) / 2)]
        if roundMedian == 0:
            roundMedian = 0.1


        self.empty_vehicle_number_trend.append(empty_vehicle_number)
        self.total_vehicle_number_trend.append(total_vehicle_number)
        self.empty_vehicle_ratio_trend.append(empty_vehicle_number / total_vehicle_number)
        self.JAIN_trend.append(self.jain)


    def set_the_best_clustes_level(self):
        print("ELBOW")
        vehicle_List = []
        x = []
        cnt = 0
        y = []
        income = []

        for vehicle in self.vehicles:
            x.append(cnt)
            y.append(vehicle.income[-1])
            temp = [vehicle.income[-1]]
            income.append(temp)
            cnt += 1
            vehicle_List.append(vehicle)

        # print(income)
        featureList = ['income']
        mdl = pd.DataFrame.from_records(income, columns=featureList)
        mdlNew = np.array(mdl[['income']])

        K = [2, 10]
        sse_result = []
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(mdlNew)
            sse_result.append(sum(np.min(cdist(mdlNew, kmeans.cluster_centers_, 'euclidean'), axis=1)) / mdlNew.shape[0])


        kl = KneeLocator(K, sse_result, curve="convex", direction="decreasing")
        # kl.plot_knee()
        # print(kl.knee)
        self.clusters = int(kl.knee)
        self.clustering_drivers(int(kl.knee))


    def clustering_drivers(self,cluters):
        self.clusters = cluters
        print("收入状态更新 cluters = ",cluters)
        if cluters == 1:
            self.clustersList = [1]
            for vehicle in self.vehicles:
                vehicle.set_clusters(1)
        else:
            vehicle_List = []
            x = []
            cnt = 0
            y = []
            income = []
            for vehicle in self.vehicles:
                x.append(cnt)
                y.append(vehicle.income[-1])
                temp = [vehicle.income[-1]]
                income.append(temp)
                cnt += 1
                vehicle_List.append(vehicle)

            # print(income)
            featureList = ['income']
            mdl = pd.DataFrame.from_records(income, columns=featureList)
            mdlNew = np.array(mdl[['income']])


            clf = KMeans(n_clusters=self.clusters)
            clf.fit(mdlNew)
            mdl['label'] = clf.labels_

            # print("labels_ = ", clf.labels_)
            y_ = clf.predict(mdlNew)
            tempClu = []
            indexList = []
            for i in range(self.clusters):
                index = np.argwhere(y_ == i).reshape(-1)
                indexList.append(index)
                # print(income[index[-1]])
                tem = [i, income[index[-1]]]
                tempClu.append(tem)

            tempClu.sort(key=lambda x: x[1])
            # print("tempClu = ", tempClu)
            tempList = []
            for item in tempClu:
                round = indexList[item[0]]
                tempList.append(item[0])
                for i in round:
                    vehicle_List[i].set_clusters(item[0])
            # print("tempList = ", tempList)
            self.clustersList = tempList




    # 最后一个时间槽 然后所有有订单的车结束所有订单
    def finish_all_orders(self):
        # idIncome = []
        for vehicle in self.vehicles:
            if not vehicle.is_activated:
                continue
            # 还有订单，直接结束
            # if vehicle.have_service_mission:
            self.each_vehicles_cost.append(vehicle.vehicle_type.accumulated_cost)
            self.each_vehicles_finish_order_number.append(vehicle.assigned_order_number)
            self.each_vehicles_service_distance.append(vehicle.service_driven_distance)
            self.each_vehicles_random_distance.append(vehicle.random_driven_distance)
            self.each_vehicles_income.append([vehicle.vehicle_id, vehicle.unit_cost, vehicle.incomeacc, vehicle.income_level, vehicle.income])
            # idIncome.append([vehicle.vehicle_id, vehicle.unit_cost,vehicle.income])
        self.accumulate_service_distance_trend.append(
            sum([vehicle.service_driven_distance for vehicle in self.vehicles if vehicle.is_activated]))
        self.accumulate_random_distance_trend.append(
            sum([vehicle.random_driven_distance for vehicle in self.vehicles if vehicle.is_activated]))



    def summary_each_round_result(self, new_orders: Set[Order], same_zone_distance) -> NoReturn:
        """
        总结这次分配的结果
        :param new_orders: 新的订单
        :return:
        """
        mechanism = self.platform.matching_method
        # print(mechanism.passenger_payment)
        # print(mechanism.social_welfare)
        # print(len(mechanism.matched_results))
        res = []
        for vehicle in self.vehicles:
            res.append(vehicle.incomeacc)
            if vehicle in mechanism.matched_vehicles:
                continue
        # print("current time",self.current_time)
        self.social_welfare_trend.append(mechanism.social_welfare)
        self.social_cost_trend.append(mechanism.social_cost)
        self.total_passenger_payment_trend.append(mechanism.passenger_payment)
        self.total_passenger_utility_trend.append(mechanism.passenger_utility)
        self.platform_profit_trend.append(mechanism.platform_profit)
        self.serviced_orders_number_trend.append(len(mechanism.matched_orders))
        self.total_orders_number_trend.append(len(new_orders))
        if sum(self.total_orders_number_trend) != 0:
            self.accumulate_service_ratio_trend.append(sum(self.serviced_orders_number_trend) / sum(self.total_orders_number_trend))
        self.bidding_time_trend.append(mechanism.bidding_time)
        # self.running_time_trend.append(mechanism.running_time)
        self.running_time_trend.append(self.running_time)
        self.accumulate_service_distance_trend.append(sum([vehicle.service_driven_distance for vehicle in self.vehicles if vehicle.is_activated]))
        self.accumulate_random_distance_trend.append(sum([vehicle.random_driven_distance for vehicle in self.vehicles if vehicle.is_activated]))
        # 记录状态（s, a, r, s'）