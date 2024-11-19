# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/14 9:15 上午

from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME, TIME_SLOT, MAX_REPEATS
from setting import TYPED, TYPED_LEARNING_RESULT_FILE, LEARNING_RESULT_FILE, GAMMA
from setting import WEEKEND
from typing import Dict, Set, Tuple, List
import numpy as np
import pickle

gamma = 0.9


class VehicleState:
    """
        车辆状态
    """
    __slots__ = ["_time", "_location"]

    def __init__(self, time, location):
        self._time = time
        self._location = location

    @property
    def time(self):
        return self._time

    @property
    def location(self):
        return self._location

    @time.setter
    def time(self, time):
        self._time = time

    @location.setter
    def location(self, location):
        self._location = location

    def __repr__(self):
        return "time : {0}, location : {1}".format(self._time, self._location)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(self.location)

# class LevelVehicleState(VehicleState):
#
#     __slots__ = ["_level"]
#
#     def __init__(self, time, location, level):
#         super().__init__(time, location)
#         self._level = level
#
#     def __repr__(self):
#         return "time : {0}, location : {1}, level: {2}".format(self._time, self._location, self._level)
#
#     def __eq__(self, other):
#         if not isinstance(other, self.__class__):
#             raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
#         return self.time == other.time and self.location == other.location and self.level == other.level
#
#     def __hash__(self):
#         return hash(self.location)
#
#     @property
#     def level(self):
#         return self._level
#
#     @level.setter
#     def level(self, level):
#         self._level = level

class LevelVehicleState():  # online

    __slots__ = ["_location","_level"]

    def __init__(self,  location, level):
        self._location = location
        self._level = level

    def __repr__(self):
        return " location : {0}, level: {1}".format(self._location, self._level)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
        return self._location == other._location and self.level == other.level

    def __hash__(self):
        return hash(self._location)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level

class LafVehicleState:
    """
            车辆状态
        """
    __slots__ = ["_location"]

    def __init__(self, location):
        self._location = location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._location = location

    def __repr__(self):
        return "location : {0}".format(self._location)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
        return self.location == other.location

    def __hash__(self):
        return hash(self.location)


class TypedVehicleState(VehicleState):

    __slots__ = ["_type"]

    def __init__(self, time, location, type):
        super().__init__(time, location)
        self._type = type

    def __repr__(self):
        return "time : {0}, location : {1}, type: {2}".format(self._time, self._location, self._type)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("{0} is not {1}".format(other.__class__, self.__class__))
        return self.time == other.time and self.location == other.location and self.type == other.type

    def __hash__(self):
        return hash(self.location)
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        self._type = type


def policy_valuation_without_vehicle_type(d: Dict):
    """
        根据历史记录计算每个时间点每个区域内车的价值函数
        :param d: 历史记录
    """
    v: Dict[VehicleState] = dict()    # 各个状态s的价值 s = (t, g_t)
    n: Dict[VehicleState] = dict()
    for t in range(MAX_REQUEST_TIME, MIN_REQUEST_TIME, -TIME_SLOT):
        """
            d_t中存放的为s_i状态下时间为t的历史数据(s_i, a_i, r_i, s_i') s_i = (t, g_t)
            这里没有考虑车的类型
            a_i不需要
        """
        d_t: List[Tuple] = d[t]
        for (s_i, r_i, s_i_) in d_t:
            if s_i not in n:
                n[s_i] = 0
            n[s_i] += 1
            if s_i not in v:
                v[s_i] = 0
            if s_i_ not in v:
                v[s_i_] = 0
            delta_t = int(np.ceil(
                (s_i_.time - s_i.time) /
                TIME_SLOT
            ))
            r_gamma = 0
            for j in range(delta_t):
                r_gamma += np.power(gamma, j) * (r_i / delta_t)
            v[s_i] += float(1/n[s_i]) * (np.power(gamma, delta_t)*v[s_i_] + r_gamma - v[s_i])
    with open(LEARNING_RESULT_FILE, 'wb') as file:
        pickle.dump(v, file)


def policy_valuation_with_vehicle_type(d: Dict):
    v: Dict[TypedVehicleState] = dict()  # 各个状态s的价值 s = (t, g_t, type)
    n: Dict[TypedVehicleState] = dict()
    for t in range(MAX_REQUEST_TIME, MIN_REQUEST_TIME, -TIME_SLOT):
        """
            d_t中存放的为s_i状态下时间为t的历史数据(s_i, a_i, r_i, s_i') s_i = (t, g_t, type)
            a_i 不需要
            !!!末尾的状态价值为0怎么处理!!!
        """
        # print(t)
        d_t: List[Tuple] = d[t]
        for (s_i, r_i, s_i_) in d_t:
            if s_i not in n:
                n[s_i] = 0
            n[s_i] += 1
            if s_i not in v:
                v[s_i] = 0
            if s_i_ not in v:
                v[s_i_] = 0
            delta_t = int(np.ceil(
                (s_i_.time - s_i.time) /
                TIME_SLOT
            ))
            r_gamma = 0
            for j in range(delta_t):
                r_gamma += np.power(gamma, j) * (r_i / delta_t)
            v[s_i] += float(1 / n[s_i]) * (np.power(gamma, delta_t) * v[s_i_] + r_gamma - v[s_i])
    return v


if __name__ == "__main__":
    # 求工作日平均的v
    v_all: Dict = dict()
    v_mean: Dict = dict()
    for day in range(1, 31):
        if day in WEEKEND:
            continue
        print(day)
        with open("../result/learning/Nearest_Matching_{0}_0_2500_60_68400_75600.pkl".format(day), "rb") as file:
            d: Dict = pickle.load(file)
            v = policy_valuation_without_vehicle_type(d)
            print(len(v.keys()))
            for key in v.keys():
                if key not in v_all:
                    v_all[key] = list()
                v_all[key].append(v[key])
    for key in v_all.keys():
        v_mean[key] = np.mean(v_all[key])
    with open(LEARNING_RESULT_FILE, "wb") as file:
        pickle.dump(v_mean, file)
    # v_all: Dict = dict()
    # v_mean: Dict = dict()
    # for day in range(1, 31):
    #     if day in WEEKEND:
    #         continue
    #     print(day)
    #     with open("../result/learning/TYPED_Nearest_Matching_{0}_0_2500_60_68400_75600.pkl".format(day), "rb") as file:
    #         d: Dict = pickle.load(file)
    #         v = policy_valuation_with_vehicle_type(d)
    #         print(len(v.keys()))
    #         for key in v.keys():
    #             if key not in v_all:
    #                 v_all[key] = list()
    #             v_all[key].append(v[key])
    # for key in v_all.keys():
    #     v_mean[key] = np.mean(v_all[key])
    # with open(TYPED_LEARNING_RESULT_FILE, "wb") as file:
    #     pickle.dump(v_mean, file)

#     # 求平均的v
#     v_all: Dict = dict()
#     v_mean: Dict = dict()
#     for repeat_time in range(MAX_REPEATS):
#         print(repeat_time)
#         with open("../result/learning/Nearest_Matching_{0}_1500_60_61200_72000.pkl".format(repeat_time), "rb") as file:
#             d: Dict = pickle.load(file)
#             v = policy_valuation_with_vehicle_type(d)
#             print(len(v.keys()))
#             for key in v.keys():
#                 if key not in v_all:
#                     v_all[key] = list()
#                 v_all[key].append(v[key])
#     for key in v_all.keys():
#         v_mean[key] = np.mean(v_all[key])
#     with open(LEARNING_RESULT_FILE, "wb") as file:
#         pickle.dump(v_mean, file)
