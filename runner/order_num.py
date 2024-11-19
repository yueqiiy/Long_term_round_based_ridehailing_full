"""
从工作日中随机选取平均数量订单
:return:
"""
# 这个是由真实的订单数据的生成需要的结果
from setting import MIN_REQUEST_TIME, MAX_REQUEST_TIME
import numpy as np
import pandas as pd
# shortest_distance = np.load("../data/Manhattan/network_data/shortest_distance.npy")
order_data = pd.read_csv("../preprocess/raw_data/temp/Manhattan/week_day.csv")
order_data = order_data[(MIN_REQUEST_TIME <= order_data.request_time) & (order_data.request_time < MAX_REQUEST_TIME)]
# order_data["wait_time"] = np.random.choice(WAIT_TIMES, size=order_data.shape[0])
# order_data["n_riders"] = 1  # TODO 这一步是为了能保证2的上限, 以后可能需要修改
# order_data["order_fare"] = np.round(order_data["order_fare"].values, POINT_LENGTH)
# 计算平均订单数量
avg_num = 0
count = 0
for day, df_day in order_data.groupby("day"):
    print("day",day)
    print("df_day",df_day)
    avg_num = avg_num + len(df_day)
    count = count + 1
avg_num = int(np.ceil(avg_num / count))
print(avg_num)
# 从order_data中随机去avg_num个数据
order_data = order_data.sample(avg_num)
# order_data = order_data[
#     ["request_time",  "pick_index", "drop_index", "order_distance", "order_fare", "n_riders"]]
# order_data = order_data.sort_values(by="request_time", axis=0, ascending=True)
order_data.to_csv("../temp.csv", index=False)