# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/9 10:25 上午

import pandas as pd
# from setting import MILE_TO_M
# from setting import POINT_LENGTH
import numpy as np

manhattan_order_data = pd.read_csv("./raw_data/temp/Manhattan/manhattan_temp.csv")
# manhattan_order_data["order_distance"] = np.round(manhattan_order_data["order_distance"] * MILE_TO_M, POINT_LENGTH)
# 保存订单数据
for day, df_day in manhattan_order_data.groupby("day"):
    day = int(day)
    df_day = df_day.sort_values(by="request_time", axis=0, ascending=True)
    df_day.drop(columns=["day"], axis=1, inplace=True)
    df_day.to_csv("./raw_data/temp/Manhattan/order_data_{:03d}.csv".format(day), index=False)
