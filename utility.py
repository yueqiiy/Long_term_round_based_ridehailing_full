# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 10:37 上午

from functools import wraps
from setting import  VALUE_EPS

def singleton(cls):
    _instance = {}

    @wraps(cls)
    def _func(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _func


def equal(float_value1: float, float_value2: float):
    """
    用于判断两个浮点数相等
    :param float_value1: 第一个浮点数
    :param float_value2: 第二个浮点数
    :return:
    """
    return abs(float_value1 - float_value2) <= VALUE_EPS


def is_enough_small(value: float, eps: float) -> bool:
    """
    如果值过小(小于等于eps)就返回True
    :param value: 距离
    :param eps: 评判足够小的刻度，小于等于这个值就是足够小了
    :return:
    """
    return value < eps or equal(value, eps)  # 由于浮点数有精度误差必须这么写

class Day:
    __instance = None
    __slots__ = ["day"]

    def __new__(cls, *args, **kwargs):
        if cls.__instance==None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance