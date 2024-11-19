import collections
import os
from collections import defaultdict
from typing import Set, List, Tuple, Dict, NoReturn, Any

from algorithm.KM import find_max_match
from algorithm.utility import MatchingMethod
from env.location import VehicleLocation
from env.network import Network
from env.order import Order
from env.pair import Pair
from env.vehicle import Vehicle
from learning.run import LevelVehicleState
from setting import VEHICLE_SPEED, TIME_SLOT, GAMMA, MIN_REQUEST_TIME, FLOAT_ZERO
import numpy as np
import time
from utility import is_enough_small
from learning.run import LafVehicleState


class LafMatchingMethod(MatchingMethod):

    __slots__ = ["bids","id2vehicle","id2order"]

    def __init__(self):
        super(LafMatchingMethod, self).__init__()
        self.id2vehicle= None
        self.id2order= None
        self.bids= None

    def _dispatch(self, vehicles: List[Vehicle], orders: Set[Order],
                     current_time: int, network: Network, near_zones, states,median) ->  List[Tuple[Vehicle, Order]]:
        """
        :param vehicles:
        :param orders:
        :param current_time:
        :param network:
        :param args: 是否为长期收益 长期收益时此处传入Dict[VehicleState]字典
        :return:
        """
        feasible_vehicles = set()
        feasible_orders = set()
        bids = dict()
        idtovehicle = dict()
        idtoorder = dict()
        for vehicle in vehicles:
            if vehicle.have_service_mission:
                continue
            order_bids = vehicle.get_costs(orders, current_time, network)
            order_bids = {order: cost for order, cost in order_bids.items() if
                          is_enough_small(cost, order.order_fare)}  # 这里可以保证订单的合理性
            idtovehicle[vehicle.vehicle_id] = vehicle
            if len(order_bids) > 0:
                feasible_vehicles.add(vehicle)
                bids[vehicle] = order_bids
            for order in order_bids:
                feasible_orders.add(order)
                idtoorder[order.order_id] = order

        self.bids = bids
        self.id2vehicle = idtovehicle
        self.id2order = idtoorder
        edges = []  # type: List[Pair]
        for vehicle, order_bids in bids.items():
            for order, cost in order_bids.items():
                if states:
                    # 二部图权重 长期收益情况下二部图权重改为r^t*V(s') - v(s) + R_r
                    r = order.order_fare - cost
                    delta_t = int(np.ceil(
                        (network.get_shortest_distance(vehicle.location, order.pick_location) + order.order_distance) /
                        VEHICLE_SPEED /
                        TIME_SLOT
                    ))  # 从当前位置 -> 接到乘客 -> 送达乘客 所需时间 单位为/time_slot
                    vehicle_state = LafVehicleState(vehicle.location)
                    vehicle_near = near_zones[vehicle.location.osm_index]
                    s0 = 0
                    value = 0
                    if vehicle_state in states:
                        s0 += states[vehicle_state]
                    for near in vehicle_near:
                        if LafVehicleState(VehicleLocation(near)) in states:
                            value += states[LafVehicleState(VehicleLocation(near))]
                    v0 = (s0+value) / (1 + len(vehicle_near)) # smoothed_value
                    _location = VehicleLocation(order.drop_location.osm_index)
                    vehicle_state_ = LafVehicleState(_location)
                    # 注意这里delta_t 要✖乘 time_Slot
                    order_near = near_zones[order.drop_location.osm_index]
                    s1 = 0
                    value1 = 0
                    if vehicle_state_ in states:
                        s1 += states[vehicle_state_]
                    for near in order_near:
                        if LafVehicleState(VehicleLocation(near)) in states:
                            value1 += states[LafVehicleState(VehicleLocation(near))]
                    v1 = (s1 + value1) / (1 + len(order_near))  # smoothed_value
                    # 如果状态没有在字典中记录，则相应值为0
                    a = r + np.power(GAMMA, delta_t) * v1 - v0
                    pair = Pair(vehicle.vehicle_id, order.order_id, a)
                    edges.append(pair)
                else:
                    pair = Pair(vehicle.vehicle_id, order.order_id, order.order_fare - cost)
                    edges.append(pair)

        values = [(each.vehicle_id, each.order_id, each.weight) for each in edges]
        olt = {vehicle.vehicle_id: current_time - MIN_REQUEST_TIME for vehicle in vehicles}
        inc = {vehicle.vehicle_id: vehicle.income[-1] for vehicle in vehicles}

        ratios = [(inc[vehicle.vehicle_id] / (olt[vehicle.vehicle_id]+0.1)) / 3600 for vehicle in vehicles]
        ratios.sort()
        num = len(ratios)
        interval = [ratios[0], ratios[int(num * 0.25)], ratios[num // 2], ratios[int(num * 0.75)], ratios[-1]]

        order_price_dur = {order.order_id: (order.order_fare, order.order_distance / VEHICLE_SPEED /
                                            TIME_SLOT) for order in feasible_orders}

        val, dispatch_tuple = find_max_match(x_y_values=values, online_time=olt, income=inc,
                                             interval=interval, order_price_dur=order_price_dur,
                                             split=True, mult_process=False)
        match_tuple = [(each[0], each[1]) for each in dispatch_tuple]
        res = []
        for each in match_tuple:
            res.append((self.id2vehicle[each[0]], self.id2order[each[1]]))

        return res



    def result_saving(self,match_pairs: List[Tuple[Vehicle, Order]]):
        for winner_vehicle, corresponding_order in match_pairs:
            # 计算VCG价格
            if corresponding_order in self.bids[winner_vehicle].keys():
                cost = self.bids[winner_vehicle][corresponding_order]
            else:
                continue
            # cost 保证平台收益不为负数
            passenger_payment = corresponding_order.order_fare

            # print(passenger_payment)
            # 保存结果
            self._matched_vehicles.add(winner_vehicle)
            self._matched_orders.add(corresponding_order)
            self._matched_results[winner_vehicle].set_order(corresponding_order, passenger_payment, 0)
            self._matched_results[winner_vehicle].set_vehicle(cost)
            self._matched_results[winner_vehicle].set_income(passenger_payment-cost)
            # self._matched_results[winner_vehicle].set_per_hour_income(passenger_payment-cost)# 需要清零
            self._matched_results[winner_vehicle].set_route([corresponding_order.pick_location, corresponding_order.drop_location])
            self._social_cost += cost
            self._total_driver_costs += cost
            self._passenger_payment += passenger_payment
            self._passenger_utility += 0
            # self._total_driver_payoffs += driver_payoff
            self._platform_profit += (passenger_payment - cost)
            self._social_welfare += corresponding_order.order_fare - cost



    def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, near_zone, states, median) -> NoReturn:
        self.reset()  # 清空结果
        # 构建图
        t1 = time.time()
        # states = dict()

        self._bidding_time = (time.time() - t1)
        self.result_saving(self._dispatch(vehicles, orders, current_time, network, near_zone, states,median))
        self._running_time = (time.time() - t1)


laf_matching_method = LafMatchingMethod()
