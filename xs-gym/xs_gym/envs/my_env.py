import os
import pickle
import random
import time
from typing import NoReturn, List

import gym
from gym import spaces

from env.order import Order
from env.vehicle import Vehicle
from runner.simulator import Simulator
import numpy as np

from setting import MIN_REQUEST_TIME, EXPERIMENTAL_MODE, MATCHING_METHOD, MAX_REPEATS, MAX_REQUEST_TIME, TIME_SLOT, \
    GEO_DATA_FILE, INT_ZERO, VEHICLE_SPEED, PKL_PATH

"""
s  上一轮订单服务率、空闲司机率、平均数、jain、当前轮次T  [](20,50,20,10,90)
"""


class MyEnv(gym.Env):
    def __init__(self):
        self.simulator = Simulator()
        # 动作空间维度
        self.action_space_shape = 10
        self.action_space = spaces.Discrete(self.action_space_shape)
        # 状态空间维度
        self.observation_space_shape = 5
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.observation_space_shape,), dtype=np.float32)
        same_zone_distance = os.path.join(GEO_DATA_FILE["base_folder"], GEO_DATA_FILE["same_zone_distance_file"])

        self.current_time = 0

        self.vehicle_num = 1500
        # print("self.vehicle_num = ",self.vehicle_num)
        self._INPUT_VEHICLES_DATA_FILES = [
            "../data/input/vehicles_data/{0}_{1}_{2}_{3}_{4}.csv".format(
                EXPERIMENTAL_MODE, i, self.vehicle_num,
                MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in
            range(MAX_REPEATS)]

        self._INPUT_ORDERS_DATA_FILES = [
            "../data/input/orders_data/{0}_{1}_{2}_{3}.csv".format(
                EXPERIMENTAL_MODE, i, MIN_REQUEST_TIME,
                MAX_REQUEST_TIME)
            for i in range(MAX_REPEATS)]

        # pkl_path = ""
        # for str in PKL_PATH.split("_")[1:-1]:
        #     pkl_path += str + "_"
        # print("pkl_path = ",pkl_path)
        self._SAVE_RESULT_FILES = [
                        "../result/{0}/near_lot/{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.pkl".format(MATCHING_METHOD,
                        # "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD,
                        #                                                        "LONG_TERM_WITH_NEW_FAIR_EMPTY",pkl_path.strip("_"),
                                                                               "LONG_TERM_WITH_NEW_FAIR_EMPTY",PKL_PATH,
                                                                               # "LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5",
                                                                               EXPERIMENTAL_MODE, i, self.vehicle_num,
                                                                               TIME_SLOT, MIN_REQUEST_TIME,
                                                                               MAX_REQUEST_TIME) for i
                        in range(MAX_REPEATS)]

        with open(
                "../result/TYPED_LEVEL_States_Values_EMPTY.pkl",
                "rb") as file:
            self.state_dict = pickle.load(file)
        self.step_total_income = 0
        self.step_sum_last_10_percent_income = []
        self.step_sum_last_30_percent_income = 0
        self.cur_round_incomes = []
        self.lamuda = 0.9
        print("self.lamuda = ",self.lamuda)
        np.random.seed(0)
        self.percent = 0.1
        print("后百分之 self.percent = ", self.percent)

        self.epoch = 0

        self.IS_train = False



    def step(self, action):
        obs = []
        reward = None
        done = False
        info = None
        self.step_total_income = 0
        self.step_sum_last_30_percent_income = 0
        cnt = 0
        action += 1
        self.simulator.clustering_drivers(action)
        for current_time, new_orders in self.simulator.orders:
            # print("second current_time = ", current_time)
            self.current_time = current_time
            t1 = time.time()
            self.simulator.platform.round_cluster_based_process(self.simulator.vehicles, new_orders, current_time,
                                                                self.simulator.network,
                                                                self.simulator.near_zone, self.state_dict,
                                                                self.simulator.clustersList)
            self.simulator.summary_online_level_state_transition_info()  # 保存状态转移并且进行值迭代更新
            self.simulator.dispatch_empty_vehicle()
            self.trace_vehicles_info_step()  # 车辆更新信息
            self.simulator.summary_each_round_result(new_orders, 1)  # 统计匹配结果
            self.simulator.running_time = (time.time() - t1)
            cnt += 1
            if cnt == 5:
                break
        if self.current_time == MAX_REQUEST_TIME:
            self.simulator.finish_all_orders()
            if not self.IS_train:
                print("save result")
                self.simulator.save_simulate_result(self._SAVE_RESULT_FILES[self.epoch])
            done = True

        # s  上一轮订单服务率、空闲司机率、平均数、jain、当前轮次T
        obs.append(self.simulator.accumulate_service_ratio_trend[-1])
        obs.append(self.simulator.empty_vehicle_ratio_trend[-1])
        obs.append(np.mean(self.cur_round_incomes))
        obs.append(self.jain)
        obs.append((self.current_time-MIN_REQUEST_TIME)/60)

        reward = (self.lamuda * self.step_sum_last_30_percent_income + (1-self.lamuda) * self.step_total_income) / 100
        print("self.step_sum_last_30_percent_income = ", self.step_sum_last_30_percent_income,
              "self.step_total_income = ", self.step_total_income,"reward = ",reward)
        return obs, reward, done, info

    def reset(self):
        # print("reset")
        # 该轮中订单数、空闲司机数、司机之间的收入方差、平均数、jain、当前轮次T
        obs = []

        # p = np.array([0.5,0.5])
        # self.epoch = np.random.choice([0,1],p=p.ravel())
        # v_nums = np.array([0.25,0.25,0.25,0.25])
        # self.vehicle_num = np.random.choice([1500,2000,2500,3000],p=v_nums.ravel())
        self.simulator.reset()
        print("self.epoch= ",self.epoch,"self.vehicle_num = ",self.vehicle_num)
        if not self.IS_train:
            print("存储数据 self._SAVE_RESULT_FILES = ",self._SAVE_RESULT_FILES[self.epoch])
        self.simulator.load_env(self._INPUT_VEHICLES_DATA_FILES[self.epoch], self._INPUT_ORDERS_DATA_FILES[self.epoch])
        cnt = 0
        for current_time, new_orders in self.simulator.orders:
            # print("first current_time = ",current_time)
            self.current_time = current_time
            self.simulator.platform.round_cluster_based_process(self.simulator.vehicles, new_orders, current_time, self.simulator.network,
                                                  self.simulator.near_zone, self.state_dict, self.simulator.clustersList)
            self.simulator.summary_online_level_state_transition_info()  # 保存状态转移并且进行值迭代更新
            self.simulator.dispatch_empty_vehicle()
            self.trace_vehicles_info()  # 车辆更新信息
            self.simulator.summary_each_round_result(new_orders, 1)  # 统计匹配结果
            cnt += 1
            if cnt == 5:
                break

        # s  上一轮订单服务率、空闲司机率、平均数、jain、当前轮次T
        obs.append(self.simulator.accumulate_service_ratio_trend[-1])
        obs.append(self.simulator.empty_vehicle_ratio_trend[-1])
        obs.append(np.mean(self.cur_round_incomes))
        obs.append(self.jain)
        obs.append((self.current_time - MIN_REQUEST_TIME) / 60)
        return obs

    def render(self):
        print("render")

    def trace_vehicles_info(self, print_vehicle=False) -> NoReturn:
        """
        更新车辆信息 由于中途不会接到订单，可以直接等到到达目的地时一次性完成更新 每个时间槽更新时判断车辆原有订单是否完成
        :return:
        """
        mechanism = self.simulator.platform.matching_method
        empty_vehicle_number = INT_ZERO
        total_vehicle_number = INT_ZERO
        if self.current_time % 3600 == 0 and self.current_time != MAX_REQUEST_TIME:
            # print("*"*10+" new hour "+"*"*10)
            # print("current_time = ",self.current_time)
            self.simulator.hour += 1
            for vehicle in self.simulator.vehicles: #记录下一个小时的收入
                vehicle.income.append(0)

        income = []
        sumIncome = 0

        inchour = []
        roundInc = []

        for vehicle in self.simulator.vehicles:
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

                    self.step_total_income += matched_result.driver_income

                    vehicle.set_vehicle_per_hour_income(matched_result.driver_income)

                    vehicle.increase_accumulated_cost(matched_result.driver_cost)
                    vehicle.increase_assigned_order_number(1)
                    vehicle.vehicle_type.idle_time = 0
                    # 计算订单等待时间和完成的时间
                    pick_up_time = np.round(
                        self.simulator.network.get_shortest_distance(vehicle.location, matched_result.order.pick_location)
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
            self.cur_round_incomes.append(vehicle.incomeacc)
            roundInc.append(vehicle.roundIncomeGet)
            income.append(vehicle.incomeacc)
            sumIncome += (vehicle.incomeacc) * (vehicle.incomeacc)

        self.jain = round(((np.sum(income)*np.sum(income))/(len(self.simulator.vehicles) * (sumIncome+1))), 2)
        self.simulator.empty_vehicle_number_trend.append(empty_vehicle_number)
        self.simulator.total_vehicle_number_trend.append(total_vehicle_number)
        self.simulator.empty_vehicle_ratio_trend.append(empty_vehicle_number / total_vehicle_number)

    def trace_vehicles_info_step(self, print_vehicle=False) -> NoReturn:
        """
        更新车辆信息 由于中途不会接到订单，可以直接等到到达目的地时一次性完成更新 每个时间槽更新时判断车辆原有订单是否完成
        :return:
        """
        mechanism = self.simulator.platform.matching_method
        empty_vehicle_number = INT_ZERO
        total_vehicle_number = INT_ZERO
        if self.current_time % 3600 == 0 and self.current_time != MAX_REQUEST_TIME:
            # print("*"*10+" new hour "+"*"*10)
            # print("current_time = ",self.current_time)
            self.simulator.hour += 1
            for vehicle in self.simulator.vehicles: #记录下一个小时的收入
                vehicle.income.append(0)

        income = []
        sumIncome = 0

        inchour = []
        roundInc = []
        self.cur_round_incomes.sort()
        last_percent_30_income = self.cur_round_incomes[int(len(self.simulator.vehicles)*self.percent)]
        # last_percent_30_income = self.cur_round_incomes[int(len(self.simulator.vehicles)*0.2)]
        # print("last_percent_30_income = ",last_percent_30_income)
        self.cur_round_incomes = []
        for vehicle in self.simulator.vehicles:
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

                    self.step_total_income += matched_result.driver_income
                    if vehicle.incomeacc < last_percent_30_income:
                        self.step_sum_last_30_percent_income += matched_result.driver_income

                    vehicle.set_vehicle_per_hour_income(matched_result.driver_income)

                    vehicle.increase_accumulated_cost(matched_result.driver_cost)
                    vehicle.increase_assigned_order_number(1)
                    vehicle.vehicle_type.idle_time = 0
                    # 计算订单等待时间和完成的时间
                    pick_up_time = np.round(
                        self.simulator.network.get_shortest_distance(vehicle.location, matched_result.order.pick_location)
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
            self.cur_round_incomes.append(vehicle.incomeacc)
            roundInc.append(vehicle.roundIncomeGet)
            income.append(vehicle.incomeacc)
            sumIncome += (vehicle.incomeacc) * (vehicle.incomeacc)

        self.jain = round(((np.sum(income)*np.sum(income))/(len(self.simulator.vehicles) * (sumIncome+1))), 2)
        # print("总体标准差: ", np.std(income))
        # print("jain = ", self.jain,"sumIncome = ", sumIncome,"np.sum(income) = ", np.sum(income))


        self.simulator.empty_vehicle_number_trend.append(empty_vehicle_number)
        self.simulator.total_vehicle_number_trend.append(total_vehicle_number)
        self.simulator.empty_vehicle_ratio_trend.append(empty_vehicle_number / total_vehicle_number)


