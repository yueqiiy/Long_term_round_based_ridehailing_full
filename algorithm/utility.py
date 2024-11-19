# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 11:14 上午
from collections import defaultdict
from typing import List, Set, NoReturn

from env.vehicle import Vehicle
from env.location import OrderLocation
from env.network import Network
from env.order import Order
from setting import FLOAT_ZERO


class MatchingResult:
    __slots__ = ["_order", "_driver_route", "_passenger_payment", "_passenger_utility", "_driver_cost", "_driver_income", "_per_hour_driver_income"]

    def __init__(self):
        self._order: Order = None
        self._driver_route: List[OrderLocation] = None
        self._passenger_payment: float = FLOAT_ZERO
        self._passenger_utility: float = FLOAT_ZERO
        self._driver_cost: float = FLOAT_ZERO
        self._driver_income: float = FLOAT_ZERO
        self._per_hour_driver_income: float = FLOAT_ZERO

    @property
    def driver_income(self) -> float:
        return self._driver_income

    @property
    def per_hour_driver_income(self) -> float:
        return self._per_hour_driver_income

    @property
    def order(self) -> Order:
        return self._order

    @property
    def driver_route(self) -> List[OrderLocation]:
        return self._driver_route

    @property
    def passenger_payment(self) -> float:
        return self._passenger_payment

    @property
    def passenger_utility(self) -> float:
        return self._passenger_utility

    @property
    def driver_cost(self) -> float:
        return self._driver_cost


    def set_order(self, order: Order, payment: float, utility: float):
        self._order = order
        self._passenger_payment = payment
        self._passenger_utility = utility

    def set_vehicle(self, cost: float):
        self._driver_cost = cost

    def set_income(self, _passenger_payment: float):
        self._driver_income = _passenger_payment  # 实际上是 P - C

    def set_per_hour_income(self, _passenger_payment: float):
        self._per_hour_driver_income = _passenger_payment  # 实际上是 P - C

    def set_route(self, route: List[OrderLocation]):
        self._driver_route = route


class MatchingMethod:
    """
    分配方法类
    供需比 feasible_vehicle_number / feasible_order_number
    matched_orders: 已经得到分配的订单
    matched_vehicles: 订单分发中获得订单的车辆集合
    matched_result: 分发结果, 包括车辆获得哪些订单和回报
    bidding_time: 投标时间
    running_time: 算法分配运行的时间
    social_welfare：此轮分配的社会福利
    social_cost: 分配订单的车辆的运行成本
    total_driver_rewards: 分配订单车辆的总体支付
    total_driver_payoffs: 分配订单车辆的总效用和
    platform_profit: 平台在此轮运行中的收益
    """
    __slots__ = [
        "_matched_orders",
        "_matched_vehicles",
        "_matched_results",
        "_social_welfare",
        "_social_cost",
        "_total_driver_costs",
        "_platform_profit",
        "_passenger_payment",
        "_passenger_utility",
        "_bidding_time",
        "_running_time"]

    def __init__(self):
        self._matched_vehicles: Set[Vehicle] = set()
        self._matched_orders: Set[Order] = set()
        self._matched_results: defaultdict = defaultdict(MatchingResult)
        self._social_welfare: float = FLOAT_ZERO
        self._social_cost: float = FLOAT_ZERO
        self._total_driver_costs: float = FLOAT_ZERO
        self._passenger_payment: float = FLOAT_ZERO
        self._passenger_utility: float = FLOAT_ZERO
        self._platform_profit: float = FLOAT_ZERO
        self._bidding_time: float = FLOAT_ZERO  # 这个是比较细化的时间
        self._running_time: float = FLOAT_ZERO  # 这个是比较细化的时间

    def reset(self):
        self._matched_vehicles.clear()
        self._matched_orders.clear()
        self._matched_results.clear()
        self._social_welfare = FLOAT_ZERO
        self._social_cost = FLOAT_ZERO
        self._total_driver_costs = FLOAT_ZERO
        self._passenger_payment = FLOAT_ZERO
        self._passenger_utility = FLOAT_ZERO
        self._platform_profit = FLOAT_ZERO
        self._bidding_time = FLOAT_ZERO
        self._running_time = FLOAT_ZERO

    def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, *args) -> NoReturn:
        raise NotImplementedError  # 后续代码还需要自己实现
    # def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, near_zones, states,median) -> NoReturn:
    #     raise NotImplementedError  # 后续代码还需要自己实现
    # def run(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, near_zones, states) -> NoReturn:
    #     raise NotImplementedError  # 后续代码还需要自己实现

    def run_cluster(self, vehicles: List[Vehicle], orders: Set[Order], current_time: int, network: Network, near_zones, states, cluster) -> NoReturn:
        raise NotImplementedError  # 后续代码还需要自己实现

    @property
    def matched_vehicles(self) -> Set[Vehicle]:
        return self._matched_vehicles

    @property
    def matched_orders(self) -> Set[Order]:
        return self._matched_orders

    @property
    def matched_results(self) -> defaultdict:
        return self._matched_results

    @property
    def social_welfare(self) -> float:
        return self._social_welfare

    @social_welfare.setter
    def social_welfare(self, value: float) -> NoReturn:
        self._social_welfare = value

    @property
    def social_cost(self) -> float:
        return self._social_cost

    @social_cost.setter
    def social_cost(self, value: float) -> NoReturn:
        self._social_cost = value

    @property
    def total_driver_costs(self) -> float:
        return self._total_driver_costs

    @property
    def passenger_payment(self) -> float:
        return self._passenger_payment

    @property
    def passenger_utility(self) -> float:
        return self._passenger_utility

    @property
    def platform_profit(self) -> float:
        return self._platform_profit

    @platform_profit.setter
    def platform_profit(self, value: float) -> NoReturn:
        self._platform_profit = value

    @property
    def bidding_time(self) -> float:
        return self._bidding_time

    @property
    def running_time(self) -> float:
        return self._running_time
