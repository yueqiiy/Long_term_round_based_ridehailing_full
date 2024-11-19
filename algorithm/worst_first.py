# from collections import defaultdict
# import random
# from typing import Dict, List, Any, Set, Tuple, NoReturn
# import time
#
# from algorithm.utility import MatchingMethod
# from env.network import Network
# from env.order import Order
# from env.vehicle import Vehicle
# from utility import is_enough_small
#
#
# class WorstfirstMatching(MatchingMethod):
#     """ Agent for dispatching and reposition """
#
#     def __init__(self):
#         super(WorstfirstMatching,self).__init__()
#         """ Load your trained model and initialize the parameters """
#         # this utility is different from the one of Recorder, this one is simply a distance substraction
#         self.driver_to_utility = defaultdict(float)
#         self.id2vehicle = None
#         self.id2order = None
#         self.bids = None
#
#     def _dispatch(self, vehicles: List[Vehicle], orders: Set[Order],
#                   current_time: int, network: Network) -> List[Tuple[Vehicle, Order]]:
#         """ Compute the assignment between drivers and passengers at each time step
#         :param dispatch_observ: a list of dict, the key in the dict includes:
#                 order_id, int
#                 driver_id, int
#                 order_driver_distance, float
#                 order_start_location, a list as [lng, lat], float
#                 order_finish_location, a list as [lng, lat], float
#                 driver_location, a list as [lng, lat], float
#                 timestamp, int
#                 order_finish_timestamp, int
#                 day_of_week, int
#                 reward_units, float
#                 pick_up_eta, float
#         :param index2hash: driver_id to driver_hash
#         :return: a list of dict, the key in the dict includes:
#                 order_id and driver_id, the pair indicating the assignment
#         """
#         # record orders of each driver, add driver into utility recorder
#         if len(orders) == 0:
#             return []
#
#         cur_sec = current_time * 60
#         feasible_vehicles = set()
#         feasible_orders = set()
#         idtovehicle = dict()
#         idtoorder = dict()
#         bids = dict()
#         dispatch_observ = []
#
#         driver_to_orders = defaultdict(list)
#         for vehicle in vehicles:
#             if vehicle.have_service_mission:
#                 continue
#             order_bids = vehicle.get_costs(orders, current_time, network)
#             order_bids = {order: cost for order, cost in order_bids.items() if
#                           is_enough_small(cost, order.order_fare)}  # 这里可以保证订单的合理性
#
#             idtovehicle[vehicle.vehicle_id] = vehicle
#             if len(order_bids) > 0:
#                 feasible_vehicles.add(vehicle)
#                 bids[vehicle] = order_bids
#             for order in order_bids:
#                 feasible_orders.add(order)
#                 # v_o_distance = network.get_shortest_distance(vehicle.location, order.pick_location)
#                 # dispatch_observ.append((vehicle.vehicle_id,order.order_id,v_o_distance))
#                 idtoorder[order.order_id] = order
#                 driver_to_orders[vehicle.vehicle_id].append((order.order_fare - order_bids[order], order.order_id))
#
#
#         self.id2vehicle = idtovehicle
#         self.id2order = idtoorder
#         self.bids = bids
#
#
#         # for vehicle, order_bids in bids.items():
#         #     for order, cost in order_bids.items():
#         #         driver_to_orders[vehicle.vehicle_id].append((order.order_fare - cost, order.order_id))
#
#         # make the right order of drivers based on utility
#         # caution: only consider drivers in this round
#         utility_driver_worst_first = [(self.id2vehicle[driver_id].income[-1], driver_id)
#                                       for driver_id in driver_to_orders]
#         utility_driver_worst_first.sort()
#         # same utility need shuffling
#         i = 0
#         while i < len(utility_driver_worst_first) - 1:
#             j = i + 1
#             while j < len(utility_driver_worst_first) and \
#                     abs(utility_driver_worst_first[j][0] - utility_driver_worst_first[i][0]) < 0.000005:
#                 j += 1
#             if j - i > 1:
#                 copy = utility_driver_worst_first[i:j]
#                 random.shuffle(copy)
#                 utility_driver_worst_first[i:j] = copy
#             i = j
#
#         # worst first matching
#         assigned_orders = set()
#         dispatch_action = []
#         for utility, driver_id in utility_driver_worst_first:
#             # sort based on pref from high to low
#             driver_to_orders[driver_id].sort(reverse=True)
#             for pref, order_id in driver_to_orders[driver_id]:
#                 if order_id in assigned_orders:
#                     continue
#                 assigned_orders.add(order_id)
#                 dispatch_action.append((self.id2vehicle[driver_id], self.id2order[order_id]))
#                 self.driver_to_utility[driver_id] += pref
#                 break
#
#
#         return dispatch_action
#
#     def result_saving(self,match_pairs: List[Tuple[Vehicle, Order]]):
#         for winner_vehicle, corresponding_order in match_pairs:
#             # 计算VCG价格
#             if corresponding_order in self.bids[winner_vehicle].keys():
#                 cost = self.bids[winner_vehicle][corresponding_order]
#             else:
#                 continue
#             # cost 保证平台收益不为负数
#             passenger_payment = corresponding_order.order_fare
#
#             # print(passenger_payment)
#             # 保存结果
#             self._matched_vehicles.add(winner_vehicle)
#             self._matched_orders.add(corresponding_order)
#             self._matched_results[winner_vehicle].set_order(corresponding_order, passenger_payment, 0)
#             self._matched_results[winner_vehicle].set_vehicle(cost)
#             self._matched_results[winner_vehicle].set_income(passenger_payment-cost)
#             self._matched_results[winner_vehicle].set_per_hour_income(passenger_payment-cost)# 需要清零
#             self._matched_results[winner_vehicle].set_route([corresponding_order.pick_location, corresponding_order.drop_location])
#             self._social_cost += cost
#             self._total_driver_costs += cost
#             self._passenger_payment += passenger_payment
#             self._passenger_utility += 0
#             # self._total_driver_payoffs += driver_payoff
#             self._platform_profit += (passenger_payment - cost)
#             self._social_welfare += corresponding_order.order_fare - cost
#
#     def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, *args) -> NoReturn:
#         self.reset()  # 清空结果
#         # 构建图
#         # t1 = time.time()
#
#         # self._bidding_time = (time.time() - t1)
#         self.result_saving(self._dispatch(vehicles, orders, current_time, network))
#         # self._running_time = (time.time() - t1)
#
# worst_first_method = WorstfirstMatching()
import os
from collections import defaultdict
import random
from typing import Dict, List, Any, Set, Tuple, NoReturn
import time

from algorithm.utility import MatchingMethod
from env.network import Network
from env.order import Order
from env.vehicle import Vehicle
from utility import is_enough_small


class WorstfirstMatching(MatchingMethod):
    """ Agent for dispatching and reposition """

    def __init__(self):
        super(WorstfirstMatching,self).__init__()
        """ Load your trained model and initialize the parameters """
        # this utility is different from the one of Recorder, this one is simply a distance substraction
        self.driver_to_utility = defaultdict(float)
        self.id2vehicle = None
        self.id2order = None
        self.bids = None

    def _dispatch(self, vehicles: List[Vehicle], orders: Set[Order],
                  current_time: int, network: Network) -> List[Tuple[Vehicle, Order]]:
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
                order_id, int
                driver_id, int
                order_driver_distance, float
                order_start_location, a list as [lng, lat], float
                order_finish_location, a list as [lng, lat], float
                driver_location, a list as [lng, lat], float
                timestamp, int
                order_finish_timestamp, int
                day_of_week, int
                reward_units, float
                pick_up_eta, float
        :param index2hash: driver_id to driver_hash
        :return: a list of dict, the key in the dict includes:
                order_id and driver_id, the pair indicating the assignment
        """
        # record orders of each driver, add driver into utility recorder
        if len(orders) == 0:
            return []

        utility_driver_worst_first = [(driver.income[-1], driver)
                                      for driver in vehicles]
        utility_driver_worst_first.sort(key=lambda x: x[0])
        # same utility need shuffling
        i = 0
        while i < len(utility_driver_worst_first) - 1:
            j = i + 1
            while j < len(utility_driver_worst_first) and \
                    abs(utility_driver_worst_first[j][0] - utility_driver_worst_first[i][0]) < 0.000005:
                j += 1
            if j - i > 1:
                copy = utility_driver_worst_first[i:j]
                random.shuffle(copy)
                utility_driver_worst_first[i:j] = copy
            i = j

        # worst first matching
        assigned_orders = set()
        dispatch_action = []
        for utility, driver in utility_driver_worst_first:
            if driver in self._matched_vehicles or driver.have_service_mission or len(orders) == 0:
                continue
            # sort based on pref from high to low
            order_bids = driver.get_costs(orders, current_time, network)
            order_bids = [(order.order_fare - cost, order,cost) for order, cost in order_bids.items() if
                          is_enough_small(cost, order.order_fare)]
            order_bids.sort(key=lambda x: x[0],reverse=True)
            if len(order_bids) > 0:
                cost = order_bids[0][2]
                assigned_orders.add(order_bids[0][1])
                dispatch_action.append((driver, order_bids[0][1]))
                self.driver_to_utility[driver.vehicle_id] += order_bids[0][1].order_fare - order_bids[0][0]
                orders.remove(order_bids[0][1])
                self._matched_vehicles.add(driver)
                self._matched_orders.add(order_bids[0][1])
                self._matched_results[driver].set_order(order_bids[0][1], order_bids[0][1].order_fare, 0)
                self._matched_results[driver].set_vehicle(cost)
                self._matched_results[driver].set_income(order_bids[0][1].order_fare - cost)
                self._matched_results[driver].set_per_hour_income(order_bids[0][1].order_fare - cost)  # 需要清零
                self._matched_results[driver].set_route(
                    [order_bids[0][1].pick_location, order_bids[0][1].drop_location])
                self._social_cost += cost
                self._total_driver_costs += cost
                self._passenger_payment += order_bids[0][1].order_fare
                self._passenger_utility += 0
                # self._total_driver_payoffs += driver_payoff
                self._platform_profit += (order_bids[0][1].order_fare - cost)
                self._social_welfare += order_bids[0][1].order_fare - cost


        return dispatch_action



    def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, *args) -> NoReturn:
        self.reset()  # 清空结果
        # 构建图
        # t1 = time.time()

        # self._bidding_time = (time.time() - t1)
        self._dispatch(vehicles, orders, current_time, network)
        # self._running_time = (time.time() - t1)

worst_first_method = WorstfirstMatching()