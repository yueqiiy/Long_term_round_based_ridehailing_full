import os
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple, NoReturn

from algorithm.reassign.KM import find_max_match
from algorithm.reassign.utils import cal_online_seconds
from algorithm.utility import MatchingMethod
from env.network import Network
from env.order import Order
from env.vehicle import Vehicle
from setting import MIN_REQUEST_TIME
from utility import is_enough_small
import time

IS_PER_HOUR = False
VEHICLE_SPEED = 4.0

class Reassign(MatchingMethod):
    __slots__ = ["bids", "id2vehicle", "id2order"]

    def __init__(self):
        super(Reassign, self).__init__()
        self.id2vehicle= None
        self.id2order= None
        self.bids= None

    def _dispatch(self, vehicles: List[Vehicle], orders: Set[Order],
                     current_time: int, network: Network) ->  List[Tuple[Vehicle, Order]]:
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


        cur_sec = current_time
        feasible_vehicles = set()
        feasible_orders = set()
        idtovehicle = dict()
        idtoorder = dict()
        bids = dict()
        dispatch_observ = []
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
                v_o_distance = network.get_shortest_distance(vehicle.location, order.pick_location)
                dispatch_observ.append((vehicle.vehicle_id,order.order_id,v_o_distance))
                idtoorder[order.order_id] = order


        drivers = set(v.vehicle_id for v in feasible_vehicles)
        orders = set(o.order_id for o in feasible_orders)
        self.id2vehicle = idtovehicle
        self.id2order = idtoorder

        order_to_dur = defaultdict(int)
        for each in orders:
            order_to_dur[each] = self.id2order[each].order_distance / VEHICLE_SPEED
        order_to_pri = {each: self.id2order[each].order_fare for each in orders}
        fake_edges = [(driver, int("888888" + str(driver)), 0) for driver in drivers]
        self.bids = bids
        edges = []
        for vehicle, order_bids in bids.items():
            for order, cost in order_bids.items():
                edges.append((vehicle.vehicle_id, order.order_id, order.order_fare-cost))

        edge_plus = edges + fake_edges

        # get M_old
        v, match_old = find_max_match(edges)
        match_old_dic = {each[0]: each[1] for each in match_old}
        # get M_fair  bi search for edge weights
        match_fair = match_old
        lo, hi = 0, 50
        print("len(match_old_dic) = ", len(match_old_dic))
        while abs(lo - hi) > 0.001:
            f = (lo + hi) / 2
            # edge_f = [edge for edge in edge_plus if (self.id2vehicle[edge[0]].incomeacc    # F =  3993.7773927114076  平均总收益 105534.0308817124
            #                                          + edge[2]) /
            #           (cal_online_seconds(MIN_REQUEST_TIME, cur_sec, order_to_dur[edge[1]], IS_PER_HOUR) + 0.1) > f]
            edge_f = [edge for edge in edge_plus if (self.id2vehicle[edge[0]].income[-1]  # F =  3942.4990993294887  平均总收益 105441.2788496093
                                                     + edge[2]) > f]
            v_f, match_fair = find_max_match(edge_f)
            perfect_match = True
            if len(match_fair) < min(len(order_to_dur), len(drivers)):
                perfect_match = False
            if perfect_match:
                lo = f
            else:
                hi = f

        # while abs(lo - hi) > 0.001:
        #     f = (lo + hi) / 2
        #     # edge_f = [edge for edge in edge_plus if (self.id2vehicle[edge[0]].incomeacc    # F =  3993.7773927114076  平均总收益 105534.0308817124
        #     #                                          + edge[2]) /
        #     #           (cal_online_seconds(MIN_REQUEST_TIME, cur_sec, order_to_dur[edge[1]], IS_PER_HOUR) + 0.1) > f]
        #     edge_f = [edge for edge in edge_plus if (self.id2vehicle[edge[0]].income[-1]  # F =  3942.4990993294887  平均总收益 105441.2788496093
        #                                              + edge[2]) < f]
        #     v_f, match_fair = find_max_match(edge_f)
        #     perfect_match = True
        #     if len(match_fair) > min(len(order_to_dur), len(drivers)):
        #         perfect_match = False
        #     if perfect_match:
        #         lo = f
        #     else:
        #         hi = f

        match_fair_dic = {each[0]: each[1] for each in match_fair if each[2] > 0.000001}
        # print("len(match_fair) = ",len(match_fair),"len(match_fair_dic) = ",len(match_fair_dic))
        f_opt = lo

        # get f_threshold
        driver_incomes = [self.id2vehicle[driver].income[-1] for driver in drivers]
        driver_incomes.sort()
        f_thresh = driver_incomes[int(len(driver_incomes) * 0.1)]
        # print("lo = ", lo, "hi = ", hi,"f_thresh = ",f_thresh,"f_opt = ",f_opt)
        if f_opt <= f_thresh:
            match_new_dic = match_old_dic
        elif f_thresh > f_opt:
            match_new_dic = match_fair_dic
        else:
            # reassign
            print("reassign\n")
            match_new_dic = match_old_dic
            break_loop = 0
            while True:
                break_loop += 1
                for driver in match_new_dic:
                    order = match_new_dic[driver]
                    # print("order = ",order,"self.id2order[order] = ",self.id2order[order])
                    # price = order_to_pri[order]
                    if self.id2order[order] not in self.bids[self.id2vehicle[driver]]:
                        continue
                    price = self.bids[self.id2vehicle[driver]][self.id2order[order]]
                    # if (self.id2vehicle[driver].incomeacc + price) / \
                    #         (cal_online_seconds(MIN_REQUEST_TIME, cur_sec, order_to_dur[order], IS_PER_HOUR) + 0.1) < f_thresh:
                    if (self.id2vehicle[driver].income[-1] + price) < f_thresh:
                        v = driver
                        break
                else:
                    break
                match_new_dic.pop(v)
                if v not in match_fair_dic:
                    continue
                while True:
                    break_loop += 1
                    for driver in match_new_dic:
                        if match_new_dic[driver] == match_fair_dic[v]:
                            vp = driver
                            break
                    else:
                        break
                    match_new_dic.pop(vp)
                    match_new_dic[v] = match_fair_dic[v]
                    v = vp
                    if v not in match_fair_dic:
                        break
                    if break_loop > 1000000:
                        print("may cause dead loop")
                        break
                match_new_dic[v] = match_fair_dic[v]
                if break_loop > 1000000:
                    print("may cause dead loop")
                    break
        res = []
        assigned_orders = set()
        assigned_drivers = set()
        # print( "len(match_new_dic) = ", len(match_new_dic))
        # print("*"*50)
        for driver in drivers:
            # if driver in match_new_dic and (len(str(match_new_dic[driver])) < 4 or
            #                                 str(match_new_dic[driver])[0:4] != "8888"):
            if driver in match_new_dic and str(match_new_dic[driver])[0:6] != "888888":
                res.append((self.id2vehicle[driver], self.id2order[match_new_dic[driver]]))
                assigned_drivers.add(driver)
                assigned_orders.add(match_new_dic[driver])

        dispatch_observ.sort(key=lambda x: x[2])
        for each in dispatch_observ:
            if each[0] not in assigned_drivers and each[1] not in assigned_orders:
                res.append((self.id2vehicle[each[0]], self.id2order[each[1]]))
                assigned_orders.add(each[1])
                assigned_drivers.add(each[0])

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
            self._matched_results[winner_vehicle].set_per_hour_income(passenger_payment-cost)# 需要清零
            self._matched_results[winner_vehicle].set_route([corresponding_order.pick_location, corresponding_order.drop_location])
            self._social_cost += cost
            self._total_driver_costs += cost
            self._passenger_payment += passenger_payment
            self._passenger_utility += 0
            # self._total_driver_payoffs += driver_payoff
            self._platform_profit += (passenger_payment - cost)
            self._social_welfare += corresponding_order.order_fare - cost



    def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, *args) -> NoReturn:
        self.reset()  # 清空结果
        # 构建图
        t1 = time.time()

        self._bidding_time = (time.time() - t1)
        self.result_saving(self._dispatch(vehicles, orders, current_time, network))
        self._running_time = (time.time() - t1)


reassign_method = Reassign()


