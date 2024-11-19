# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 9:09 上午
import os
import random

import numpy as np
import pandas as pd
from typing import List, NoReturn, Dict, Set
from setting import INT_ZERO, FLOAT_ZERO, FIRST_INDEX
from agent.utility import VehicleType
from env.location import VehicleLocation, OrderLocation, PickLocation, DropLocation
from env.network import Network
from env.order import Order


class Vehicle:
    """
    车辆
    vehicle_id: 车辆id
    location: 车辆当前位置
    available_seats:  车辆剩下的位置数目
    unit_cost: 车俩的单位行驶成本
    route: 车辆的自身的行驶路线 实质上就是包含一系列订单的起始位置的序列
    service_driven_distance: 车辆为了服务行驶的距离
    is_activated: 车辆是否已经激活


    """
    __slots__ = ["_vehicle_id", "_is_activated", "_route", "_roundIncomeGet","_incomeacc", "_vehicle_type", "_accumulated_cost",
                 "_income", "_distance", "_income_level","_clusters", "_per_hour_income"]
    generate_vehicles_function = None  # 车辆生成函数

    # 这里的location指的是区域的位置 GeoLocation中的osm_index现在指区域id
    def __init__(self, vehicle_id: int, location: VehicleLocation, unit_cost: float):
        self._vehicle_id: int = vehicle_id  # 车辆唯一标识
        self._is_activated: bool = True  # 车辆是否处于激活状态，默认车是激活状态
        self._route = list()  # 车辆的行驶路线
        self._income: [float] = [FLOAT_ZERO]  # 纯利润
        self._incomeacc: float = FLOAT_ZERO
        self._roundIncomeGet: float = FLOAT_ZERO
        self._per_hour_income: float = FLOAT_ZERO
        # self._income_level: int = random.randint(1,4) # 1 2 3 分别表示相对高收入、中等、低收入状态
        self._income_level: int = 1
        self._clusters: int = 1
        self._distance: float = FLOAT_ZERO
        self._vehicle_type = VehicleType(
            location=location,  # 车辆当前的位置
            unit_cost=unit_cost,  # 单位行驶成本
            service_driven_distance=FLOAT_ZERO,  # 车俩为了服务行驶的距离
            random_driven_distance=FLOAT_ZERO,  #
            assigned_order_number=INT_ZERO,     # 车辆已完成订单数量
            accumulated_cost=FLOAT_ZERO
        )


    @staticmethod
    def set_vehicle_speed(vehicle_speed: float) -> NoReturn:
        VehicleType.set_vehicle_speed(vehicle_speed)

    @staticmethod
    def set_could_drive_distance(could_drive_distance: float) -> NoReturn:
        VehicleType.set_could_drive_distance(could_drive_distance)

    @classmethod
    def set_generate_vehicles_function(cls, function) -> NoReturn:
        cls.generate_vehicles_function = function

    @classmethod
    def generate_vehicles_data(cls, vehicle_number: int, network: Network, output_file: str):
        """
        用于生成用于模拟的文件，用于
        :param vehicle_number:
        :param network:
        :param output_file:
        :return:
        """
        locations = network.generate_random_locations(vehicle_number, VehicleLocation)  # 得到车辆位置
        with open(output_file, "w") as file:
            file.write("location_index,seats,unit_cost\n")
            for line in cls.generate_vehicles_function(locations, vehicle_number):  # 得到很多的车辆类
                file.write(",".join(map(str, line)) + "\n")

    @classmethod
    def load_vehicles_data(cls, vehicle_speed: float, time_slot: int, input_file) -> List:
        """
        用于导入已经生成的车辆数据，并加工用于模拟
        :param vehicle_speed: 车辆速度
        :param time_slot: 表示
        :param proxy_bidder: 代理投标者  不需要投标了
        :param route_planner: 路线规划器 不需要规划路径 直接去就完事了
        :param input_file: 路网
        :return:
        """
        cls.set_vehicle_speed(vehicle_speed)  # 初初始化车辆速度
        cls.set_could_drive_distance(vehicle_speed * time_slot)  # 初始化车辆的行驶
        # cls.set_proxy_bidder(proxy_bidder)
        # cls.set_route_planner(route_planner)
        vehicle_raw_data = pd.read_csv(input_file)
        vehicle_number = vehicle_raw_data.shape[0]
        vehicles = []
        for vehicle_id in range(vehicle_number):
            each_vehicle_data = vehicle_raw_data.iloc[vehicle_id, :]
            vehicles.append(cls(vehicle_id, VehicleLocation(int(each_vehicle_data["location_index"])), each_vehicle_data["unit_cost"]))
        return vehicles

    @property
    def assigned_order_number(self) -> int:
        return self.vehicle_type.assigned_order_number

    @property
    def income(self) -> list:
        return self._income

    @property
    def incomeacc(self) -> float:
        return self._incomeacc

    @property
    def roundIncomeGet(self) -> float:
        return self._roundIncomeGet

    @property
    def per_hour_income(self) -> float:
        return self._per_hour_income

    @property
    def income_level(self) -> int:
        return self._income_level

    @property
    def clusters(self) -> int:
        return self._clusters

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @property
    def is_activated(self) -> bool:
        """
        返回当前车俩是否存活
        :return:
        """
        return self._is_activated

    @property
    def vehicle_type(self) -> VehicleType:
        """
        返回车辆类型 （包括车辆的位置，单位成本，可用座位，服务行驶距离，随机行驶距离）
        :return:
        """
        return self._vehicle_type

    @property
    def available_time(self) -> int:
        return self._vehicle_type.available_time

    @property
    def unit_cost(self) -> float:
        """
        返回单位成本
        :return:
        """
        return self._vehicle_type.unit_cost

    def set_unit_cost(self, new_unit_cost: float):
        self.vehicle_type._unit_cost = new_unit_cost

    @property
    def route(self) -> List[OrderLocation]:
        """
        返回车俩行驶路线
        :return:
        """
        return self._route

    @property
    def location(self) -> VehicleLocation:
        """
        返回车辆的当前的位置
        :return:
        """
        return self._vehicle_type.location

    @property
    def service_driven_distance(self) -> float:
        """
        返回车俩为了服务行驶的距离
        :return:
        """
        return self._vehicle_type.service_driven_distance


    @property
    def random_driven_distance(self) -> float:
        """
        返回车辆随机行驶的距离
        :return:
        """
        return self._vehicle_type.random_driven_distance

    @property
    def could_drive_distance(self) -> float:
        """
        返回车辆在一个时刻可以移动的距离，这个距离是最小值其实车辆可能行驶更多距离
        :return:
        """
        return VehicleType.could_drive_distance

    @property
    def vehicle_speed(self) -> float:
        """
        返回车辆速度
        :return:
        """
        return VehicleType.vehicle_speed

    @property
    def have_service_mission(self) -> bool:
        """
        返回当前车辆是否有服务订单的任务在身
        :return:
        """
        return len(self.route) != INT_ZERO

    def get_cost(self, order: Order, current_time: int, network: Network) -> float:
        distance_to_pickup = network.get_shortest_distance(self.location, order.pick_location) # -1 表示在同一个区域
        max_wait_distance = np.round((order.request_time + order.wait_time - current_time) * self.vehicle_speed)
        # print("distance_to_pickup = ", distance_to_pickup, "max_wait_distance = ", max_wait_distance)
        # 能够按时接到乘客 cost = (pickup_distance + order_distance) * unit_cost
        if network.is_smaller_bound_distance(distance_to_pickup, max_wait_distance):
            # print("is small distance_to_pickup = ", distance_to_pickup, "max_wait_distance = ", max_wait_distance)
            # os.system("pause")
            total_distance = network.get_shortest_distance(self.location, order.pick_location) + order.order_distance
            cost = self.unit_cost * total_distance
        else:
            cost = np.inf
        return cost

    def get_costs(self, orders: Set[Order], current_time: int, network: Network):
        costs = dict()
        for order in orders:
            cost = self.get_cost(order, current_time, network)
            if cost != np.inf:
                costs[order] = cost
        return costs
    # def route_planning(self, order: Order, current_time: int, network: Network) -> PlanningResult:
    #     return self.route_planner.planning(self._vehicle_type, self.route, order, current_time, network)

    def drive_on_random(self, network: Network) -> NoReturn:
        """
        车辆在路上随机行驶
        :param network: 路网
        :return:
        ------
        注意：
        不要那些只可以进去，不可以出来的节点
        如果车辆就正好在一个节点之上，那么随机选择一个节点到达，如果不是这些情况就在原地保持不动
        """
        self.increase_random_distance(network.drive_on_random(self.location, self.could_drive_distance))

    def drive_on_route(self, current_time: int, network: Network) -> List[Order]:
        """
        车辆自己按照自己的行驶路线
        :param current_time: 当前时间
        :param network: 路网
        """
        un_covered_location_index = FIRST_INDEX  # 未完成订单坐标的最小索引
        g = network.drive_on_route(self.location, self.route, self.could_drive_distance)  # 开启车辆位置更新的生成器
        now_time: float = current_time
        _finish_orders: List[Order] = list()  # 这一轮完成的订单
        for is_access, covered_index, order_location, vehicle_to_order_distance in g:
            # is_access 表示是否可以到达 order_location
            # covered_index 表示车辆当前覆盖的路线规划列表索引
            # order_location 表示当前可以访问的订单位置
            # vehicle_to_order_distance 表示车辆到 order_location 可以行驶的距离

            self.increase_service_distance(vehicle_to_order_distance)  # 更新车辆服务行驶的距离
            now_time = now_time + vehicle_to_order_distance / self.vehicle_speed
            un_covered_location_index = covered_index + 1  # 更新未完成订单的情况
            if is_access:  # 如果当前订单是可以到达的情况
                belong_order: Order = order_location.belong_order
                if isinstance(order_location, PickLocation):
                    belong_order.set_pick_status(self.service_driven_distance, int(now_time))  # 更新当前订单的接送行驶距离
                if isinstance(order_location, DropLocation):
                    belong_order.set_drop_status(self.service_driven_distance, int(now_time))
                    self._finish_orders_number += 1
                    _finish_orders.append(belong_order)
        if un_covered_location_index != FIRST_INDEX:  # 只有有变化才更新路径
            self.set_route(self.route[un_covered_location_index:])
        return _finish_orders

    def set_route(self, route: List[OrderLocation]) -> NoReturn:
        self._route = route

    def set_vehicle_income(self, net_profit: float) -> NoReturn:
        self._income[-1] += net_profit

    def set_vehicle_incomeacc(self, net_profit: float) -> NoReturn:
        self._incomeacc += net_profit

    def set_vehicle_per_hour_income(self, net_profit: float) -> NoReturn:
        self._per_hour_income += net_profit

    def set_vehicle_roundIncome(self, net_profit: float) -> NoReturn:
        self._roundIncomeGet = net_profit

    def zero_vehicle_per_hour_income(self) -> NoReturn:
        self._per_hour_income = FLOAT_ZERO

    def set_income_level(self, level: int) -> NoReturn:
        self._income_level = level

    def set_clusters(self, clusters: int) -> NoReturn:
        self._clusters = clusters

    # def decrease_available_seats(self, n_riders: int) -> NoReturn:
    #     self._vehicle_type.available_seats -= n_riders

    # def increase_available_seats(self, n_riders: int) -> NoReturn:
    #     self._vehicle_type.available_seats += n_riders

    # 累积成本自增
    def increase_accumulated_cost(self, cost: float) -> NoReturn:
        self._vehicle_type.accumulated_cost += cost

    # 累积完成订单数量自增
    def increase_assigned_order_number(self, number: int) -> NoReturn:
        self.vehicle_type.assigned_order_number += number

    # 累积闲置时间增加
    def increase_idle_time(self, time: int) -> NoReturn:
        self._vehicle_type.idle_time += time

    # def increase_earn_profit(self, profit: float):
    #     self._earn_payoff += profit

    def increase_service_distance(self, additional_distance: float) -> NoReturn:
        self._vehicle_type.service_driven_distance += additional_distance

    def increase_random_distance(self, additional_distance: float) -> NoReturn:
        self._vehicle_type.random_driven_distance += additional_distance

    def enter_platform(self) -> NoReturn:
        """
        当车辆是没有激活的状态的时候，按照一定概率进入平台
        :return:
        """
        # TODO 日后补这个函数，车辆可能还会休息一段时间之后进入平台
        pass

    def leave_platform(self) -> NoReturn:
        """
        按照一定概率离开平台
        :return:
        """
        # TODO 日后补这个函数, 车俩可能只是在某一个时间端进行工作
        pass

    def __repr__(self):
        return "id: {0} incomeacc:{1} income_level: {2} location: {3} unit_cost: {4} route: {5}". \
            format(self.vehicle_id,self.incomeacc, self.income_level, self.location, self.unit_cost, self.route)

    def __hash__(self):
        return hash(self.vehicle_id)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
        return isinstance(other, self.__class__) and self.vehicle_id == other.vehicle_id


def generate_road_vehicles_data(locations: List[VehicleLocation], vehicle_number: int) -> List:
    from setting import FUEL_CONSUMPTION_DATA_FILE
    from setting import N_SEATS
    from setting import VEHICLE_FUEL_COST_RATIO
    car_fuel_consumption_info = pd.read_csv(FUEL_CONSUMPTION_DATA_FILE)
    cars_info = car_fuel_consumption_info.sample(n=vehicle_number)
    unit_cost_info = cars_info["fuel_consumption"].values.astype(np.float) * VEHICLE_FUEL_COST_RATIO
    """
        cost 取 6， 8， 10
    """
    seats = 4
    unit_costs = [6, 8, 10]
    # return [(locations[vehicle_id].osm_index, seats, unit_cost_info[vehicle_id]) for vehicle_id in range(vehicle_number)]
    return [(locations[vehicle_id].osm_index, seats, np.random.choice(unit_costs) * VEHICLE_FUEL_COST_RATIO) for vehicle_id in range(vehicle_number)]


def generate_grid_vehicles_data(locations: List[VehicleLocation], vehicle_number: int) -> List:
    # 车辆成本单位成本可选范围 每米司机要花费的钱
    UNIT_COSTS = [1.2, 1.3, 1.4, 1.5]
    from setting import N_SEATS
    unit_costs = np.random.choice(UNIT_COSTS, size=(vehicle_number,))
    seats = np.random.choice(N_SEATS)
    return [(locations[vehicle_id].osm_index, seats, unit_costs[vehicle_id]) for vehicle_id in range(vehicle_number)]
