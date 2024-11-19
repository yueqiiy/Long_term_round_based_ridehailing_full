# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/5 11:05 上午
from typing import List, Set, Tuple, NoReturn
import numpy as np
import time
from algorithm.utility import MatchingMethod
from env.network import Network
from env.order import Order
from env.vehicle import Vehicle
from utility import is_enough_small


class NearestMatchingMethod(MatchingMethod):

    def __init__(self):
        super(NearestMatchingMethod, self).__init__()

    def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, *args) -> NoReturn:
        self.reset()
        t1 = time.time()
        sort_by_time_orders = sorted(orders, key=lambda _order: _order.request_time)  # 按时间排序
        for order in sort_by_time_orders:
            t2 = time.time()
            vehicle_distances: List[Tuple[Vehicle, float]] = list()
            for vehicle in vehicles:
                # 如果车辆已经得到订单或者上一轮订单还没完成，剔除
                if vehicle in self._matched_vehicles or vehicle.have_service_mission:
                    continue
                # 可达性判断
                pick_up_distance = network.get_shortest_distance(vehicle.location, order.pick_location)
                max_time = order.request_time + order.wait_time - current_time
                if is_enough_small(pick_up_distance, max_time * vehicle.vehicle_speed):
                    vehicle_distances.append((vehicle,
                                              network.compute_vehicle_to_order_distance(vehicle.location,
                                                                                        order.pick_location)))
            self._bidding_time += (time.time() - t2)
            vehicle_distances.sort(key=lambda x: x[1])
            for vehicle, distance in vehicle_distances:
                cost = vehicle.get_cost(order, current_time, network)
                if is_enough_small(cost, order.order_fare):
                    self._matched_vehicles.add(vehicle)
                    self._matched_orders.add(order)
                    self._matched_results[vehicle].set_order(order, order.order_fare, 0)
                    self._matched_results[vehicle].set_vehicle(cost)
                    # self._matched_results[vehicle].set_income(order.order_fare)
                    self._matched_results[vehicle].set_income(order.order_fare - cost)
                    self._matched_results[vehicle].set_per_hour_income(order.order_fare - cost)
                    self._matched_results[vehicle].set_route([order.pick_location, order.drop_location])
                    self._social_cost += cost
                    self._total_driver_costs += cost
                    self._passenger_payment += order.order_fare
                    self._passenger_utility += 0
                    self._platform_profit += (order.order_fare - cost)
                    self._social_welfare += order.order_fare - cost
                    # 接到订单的车要剔除掉
                    break
        self._running_time += (time.time() - t1)


nearest_matching_method = NearestMatchingMethod()
