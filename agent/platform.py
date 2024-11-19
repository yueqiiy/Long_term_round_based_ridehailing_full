# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 10:35 上午
from utility import singleton
from typing import Set, NoReturn, List
from env.vehicle import Vehicle
from env.order import Order
from env.network import Network
from algorithm.utility import MatchingMethod


@singleton
class Platform:
    """
    平台
    dispatching_mechanism: 平台的运行的机制
    """
    __slots__ = ["_matching_method", "_order_pool"]

    def __init__(self, matching_method: MatchingMethod):
        self._order_pool: Set = set()
        self._matching_method: MatchingMethod = matching_method

    def collect_orders(self,  new_orders: Set[Order], current_time: int) -> NoReturn:
        """
        收集这一轮的新订单同时剔除一些已经过期的订单
        :param new_orders: 新的订单集合
        :param current_time: 当前时间
        :return:
        """
        unused_orders = set([order for order in self._order_pool if order.request_time + order.wait_time < current_time])  # 找到已经过期的订单
        self._order_pool -= unused_orders
        self._order_pool |= new_orders


    def remove_dispatched_orders(self) -> NoReturn:
        """
        从订单池子中移除已经得到分发的订单
        :return:
        """

        self._order_pool -= self._matching_method.matched_orders

    def round_based_process(self, vehicles: List[Vehicle], new_orders: Set[Order],
                            current_time: int, network: Network, *args) -> NoReturn:
    # def round_based_process(self, vehicles: List[Vehicle], new_orders: Set[Order],
    #                             current_time: int, network: Network, near_zones, states) -> NoReturn:
        """
        一轮运行过程
        :param vehicles: 车辆
        :param new_orders: 新产生的订单
        :param current_time:  当前时间
        :param network:  环境
        :return:
        """
        #  收集订单
        self.collect_orders(new_orders, current_time)

        # # 匹配定价
        # if args:    # 把传入的参数取出来
        #     args = args[0]
        self._matching_method.run(vehicles, self._order_pool, current_time, network, args)
        # self._matching_method.run(vehicles, self._order_pool, current_time, network, near_zones, states)

        # 移除已经分配的订单
        self.remove_dispatched_orders()

    def round_cluster_based_process(self, vehicles: List[Vehicle], new_orders: Set[Order],
                                current_time: int, network: Network, near_zones, states,clusters) -> NoReturn:
        """
        一轮运行过程
        :param vehicles: 车辆
        :param new_orders: 新产生的订单
        :param current_time:  当前时间
        :param network:  环境
        :return:
        """
        #  收集订单
        self.collect_orders(new_orders, current_time)

        self._matching_method.run_cluster(vehicles, self._order_pool, current_time, network, near_zones, states,clusters)

        # 移除已经分配的订单
        self.remove_dispatched_orders()


    def laf_round_based_process(self, vehicles: List[Vehicle], new_orders: Set[Order],
                            current_time: int, network: Network, near_zone, states, median) -> NoReturn:
        """
        一轮运行过程
        :param vehicles: 车辆
        :param new_orders: 新产生的订单
        :param current_time:  当前时间
        :param network:  环境
        :return:
        """
        #  收集订单
        self.collect_orders(new_orders, current_time)

        self._matching_method.run(vehicles, self._order_pool, current_time, network, near_zone, states, median)

        # 移除已经分配的订单
        self.remove_dispatched_orders()

    @property
    def order_pool(self) -> Set[Order]:
        return self._order_pool

    @property
    def matching_method(self) -> MatchingMethod:
        return self._matching_method

    def reset(self):
        """
        平台重置
        """
        self._order_pool.clear()  # 清空订单
        self._matching_method.reset()  # 机制自动清空
