import os
from collections import defaultdict
from typing import Set, List, Tuple, NoReturn

from pulp import LpProblem, LpMinimize, LpVariable, PULP_CBC_CMD, LpBinary

from algorithm.utility import MatchingMethod
from env.location import VehicleLocation
from env.network import Network
from env.order import Order
from env.vehicle import Vehicle
from learning.run import LevelVehicleState
from setting import MIN_REQUEST_TIME, TIME_SLOT, VEHICLE_SPEED, GAMMA
from utility import is_enough_small
import time
import numpy as np

topK = 50000
start_time = MIN_REQUEST_TIME

class LongTermILPMatchingMethod(MatchingMethod):
    __slots__ = ["drivers_utility","driver_online_rounds","driver_max_last_round","bids"]

    def __init__(self):
        super(LongTermILPMatchingMethod, self).__init__()
        self.drivers_utility = defaultdict(float)  # type: Dict[int, float]
        self.driver_online_rounds = defaultdict(int)  # type: Dict[int, int]
        self.driver_max_last_round = 2
        self.bids = None


    def _dispatch(self, vehicles: List[Vehicle], orders: Set[Order],
                  current_time: int, network: Network, near_zones, states) -> List[Tuple[Vehicle, Order]]:
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

        if len(orders) == 0:
            return []
        cur_local = current_time
        od_decision = dict()  # type: Dict[str, Any]
        if_match_utility = dict()  # type: Dict[str, float]
        drivers_cur_round = set()  # type: Set[int]
        orders_cur_round = set()  # type: Set[int]
        order_driver_cand = defaultdict(int)

        dispatch_observ = []
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
                # feasible_vehicles.add(vehicle)
                bids[vehicle] = order_bids
            for order in order_bids:
                # feasible_orders.add(order)
                v_o_distance = network.get_shortest_distance(vehicle.location, order.pick_location)
                dispatch_observ.append((vehicle, order, v_o_distance))
                idtoorder[order.order_id] = order

        self.bids = bids
        dispatch_observ.sort(key=lambda x: x[2])
        for od in dispatch_observ:
            driver_id = od[0].vehicle_id  # type: int
            order_id = od[1].order_id  # type: int
            orders_cur_round.add(order_id)
            if order_driver_cand[order_id] < topK:
                order_driver_cand[order_id] += 1
                drivers_cur_round.add(driver_id)
                driver_id_order_id = str(driver_id) + "_" + str(order_id)  # type: str
                od_decision[driver_id_order_id] = LpVariable(cat=LpBinary, name=driver_id_order_id)

                r = od[1].order_fare - self.bids[od[0]][od[1]]
                delta_t = int(np.ceil(
                    (network.get_shortest_distance(od[0].location, od[1].pick_location) + od[1].order_distance) /
                    VEHICLE_SPEED /
                    TIME_SLOT
                ))  # 从当前位置 -> 接到乘客 -> 送达乘客 所需时间 单位为/time_slot
                vehicle_state = LevelVehicleState(od[0].location, od[0].income_level)
                vehicle_near = near_zones[od[0].location.osm_index]
                s0 = 0
                value = 0
                if vehicle_state in states:
                    s0 += states[vehicle_state]
                for near in vehicle_near:
                    if LevelVehicleState(VehicleLocation(near), od[0].income_level) in states:
                        value += states[LevelVehicleState(VehicleLocation(near), od[0].income_level)]

                v0 = (s0 + value) / (1 + len(vehicle_near))  # smoothed_value
                # v0 = s0
                _location = VehicleLocation(od[1].drop_location.osm_index)
                vehicle_state_ = LevelVehicleState(_location, od[0].income_level)
                order_near = near_zones[od[1].drop_location.osm_index]
                s1 = 0
                value1 = 0
                if vehicle_state_ in states:
                    s1 += states[vehicle_state_]
                for near in order_near:
                    if LevelVehicleState(VehicleLocation(near), od[0].income_level) in states:
                        value1 += states[LevelVehicleState(VehicleLocation(near), od[0].income_level)]

                v1 = (s1 + value1) / (1 + len(order_near))  # smoothed_value
                # v1 = s1
                # 如果状态没有在字典中记录，则相应值为0
                a = r + 5*(np.power(GAMMA, delta_t) * v1 - v0)
                if_match_utility[driver_id_order_id] = a


        # Create a new model
        m = LpProblem(name="ILP_Model", sense=LpMinimize)
        # each driver should only have at most one order
        # each order should only have at most one driver
        driver_constrains = defaultdict(int)
        order_constrains = defaultdict(int)
        goal = 0
        for driver_id_order_id in od_decision:
            driver_id, order_id = driver_id_order_id.split('_')
            driver_constrains[driver_id] += od_decision[driver_id_order_id]
            order_constrains[order_id] += od_decision[driver_id_order_id]
            one_driver = self.driver_max_last_round - \
                         (self.drivers_utility[driver_id] / (self.driver_online_rounds[driver_id] + 1)
                          + od_decision[driver_id_order_id] * if_match_utility[driver_id_order_id] /
                          (self.driver_online_rounds[driver_id] + 1))
            # one_driver = self.driver_max_last_round - \
            #              (self.drivers_utility[driver_id] + od_decision[driver_id_order_id] * if_match_utility[driver_id_order_id])
            one_driver_abs = LpVariable(name="abs_driver" + driver_id_order_id)
            m += (one_driver_abs >= one_driver)
            m += (one_driver_abs >= -one_driver)
            goal += one_driver_abs

        cnt = 0
        for driver_id in driver_constrains:
            m += (driver_constrains[driver_id] <= 1)
        for order_id in order_constrains:
            m += (order_constrains[order_id] <= 1)
        # os.system("pause")
        m += goal
        m.solve(PULP_CBC_CMD(msg=False))
        # update the online rounds
        # if cur_local.tm_min == 0 and cur_local.tm_min == 0:

        if cur_local == start_time + TIME_SLOT:
            # print("cur_local = ",cur_local,"startTime = ",start_time)
            for driver_id in drivers_cur_round:
                self.driver_online_rounds[driver_id] = 1
            for driver_id in self.drivers_utility:
                self.drivers_utility[driver_id] = 0

        for driver_id in drivers_cur_round:
            self.driver_online_rounds[driver_id] += 1

        # dispatch_action = []  type: # List[Dict[str, int]]
        dispatch_action = []
        temp = 0
        drivers_match = defaultdict(int)
        orders_match = defaultdict(int)
        drivers_set = set()
        res = []

        for v in m.variables():
            if v.varValue == 1:
                if v.name.startswith('abs_driver'):
                    break
                driver_id_str, order_id_str = v.name.split('_')
                drivers_match[int(driver_id_str)] += 1
                orders_match[int(driver_id_str)] += 1

                res.append(v.name)
                dispatch_action.append((idtovehicle[int(driver_id_str)], idtoorder[int(order_id_str)]))
                # update the utility
                # self.drivers_utility[int(driver_id_str)] += if_match_utility[driver_id_str + '_' + order_id_str]
                self.drivers_utility[int(driver_id_str)] += idtoorder[int(order_id_str)].order_fare - \
                                                            self.bids[idtovehicle[int(driver_id_str)]][
                                                                idtoorder[int(order_id_str)]]

                self.driver_max_last_round = max(self.driver_max_last_round,
                                                 self.drivers_utility[int(driver_id_str)] / self.driver_online_rounds[
                                                     int(driver_id_str)])

        print("driver_max_last_round = ", self.driver_max_last_round)

        return dispatch_action

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



    def run_cluster(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network,
                        near_zones, states, clusters) -> NoReturn:
        self.reset()  # 清空结果
        # 构建图
        t1 = time.time()
        # states = dict()

        self._bidding_time = (time.time() - t1)
        self.result_saving(self._dispatch(vehicles, orders, current_time, network, near_zones, states))
        self._running_time = (time.time() - t1)


longterm_ilp_matching_method = LongTermILPMatchingMethod()