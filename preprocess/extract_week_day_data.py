# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/12/10 上午9:48
import pandas as pd
from setting import MILE_TO_M
from setting import POINT_LENGTH
import numpy as np

"""
导出工作日订单
"""
weekend = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
manhattan_order_data = pd.read_csv("./raw_data/temp/Manhattan/manhattan_temp.csv")
manhattan_order_data["order_distance"] = np.round(manhattan_order_data["order_distance"] * MILE_TO_M, POINT_LENGTH)
# isin 判断对应列的值是不是在weekend中 ～去翻
manhattan_order_data = manhattan_order_data[~manhattan_order_data['day'].isin(weekend)]
manhattan_order_data = manhattan_order_data.sort_values(by="request_time", axis=0, ascending=True)
manhattan_order_data.to_csv("./raw_data/temp/Manhattan/week_day.csv", index=False)