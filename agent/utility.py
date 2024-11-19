# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 9:34 上午

from typing import List
import numpy as np
from env.location import VehicleLocation


class VehicleType:
    """
    vehicle 用这个类与外界进行交互
    _location 当前位置,
    _available_time 可用的时间: -1代表可用 >0则表示接到了订单到达目的地的时间
    车辆接到订单修改_assigned_order_number _route _available_time
    当前时间等于_available_time时完成订单 修改_service_driven_distance _location _available_time
    """
    __slots__ = ["_location", "_available_time", "_unit_cost", "_service_driven_distance",
                 "_random_driven_distance", "_accumulated_cost", "_assigned_order_number", "_idle_time"]

    vehicle_speed: float = None  # 车辆的平均速度
    could_drive_distance: float = None  # 一个分配时间可以行驶的距离

    def __init__(self, location: VehicleLocation, unit_cost: float, service_driven_distance: float,
                 assigned_order_number: int, random_driven_distance: float, accumulated_cost: float):
        self._location = location
        self._available_time = -1
        self._idle_time = 0
        self._unit_cost = unit_cost
        self._service_driven_distance = service_driven_distance
        self._random_driven_distance = random_driven_distance
        self._accumulated_cost = accumulated_cost
        self._assigned_order_number = assigned_order_number  # 已经分配的订单数目

    @property
    def assigned_order_number(self) -> int:
        return self._assigned_order_number

    @assigned_order_number.setter
    def assigned_order_number(self, value: int):
        self._assigned_order_number = value

    @property
    def accumulated_cost(self) -> float:
        return self._accumulated_cost

    @accumulated_cost.setter
    def accumulated_cost(self, cost):
        self._accumulated_cost = cost

    @property
    def idle_time(self) -> int:
        return self._idle_time

    @idle_time.setter
    def idle_time(self, time):
        self._idle_time = time

    @property
    def available_time(self) -> int:
        return self._available_time

    @available_time.setter
    def available_time(self, time):
        self._available_time = time

    @property
    def location(self) -> VehicleLocation:
        return self._location

    @location.setter
    def location(self, location: VehicleLocation):
        self._location = location

    @property
    def unit_cost(self) -> float:
        return self._unit_cost

    @property
    def service_driven_distance(self) -> float:
        return self._service_driven_distance

    @service_driven_distance.setter
    def service_driven_distance(self, distance: float):
        self._service_driven_distance = distance

    @property
    def random_driven_distance(self) -> float:
        return self._random_driven_distance

    @random_driven_distance.setter
    def random_driven_distance(self, distance: float):
        self._random_driven_distance = distance

    @classmethod
    def set_vehicle_speed(cls, vehicle_speed: float):
        cls.vehicle_speed = np.round(vehicle_speed)

    @classmethod
    def set_could_drive_distance(cls, could_drive_distance: float):
        could_drive_distance = np.round(could_drive_distance)
        cls.could_drive_distance = could_drive_distance

