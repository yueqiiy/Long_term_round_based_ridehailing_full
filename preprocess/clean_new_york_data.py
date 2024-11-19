# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/8 2:53 下午

"""
清理纽约市出租车数据
"""
import os
import pandas as pd
import numpy as np
import pickle

def time2int(time, gap=0):
    _s = time.split(":")
    return str(int(_s[0]) * 60 * 60 + int(_s[1]) * 60 + int(_s[2]) + gap)

result_dir = "raw_data/temp/Manhattan/"
green, yellow = "green", "yellow"
chunk_size = 100000
file_name = "./yellow_tripdata_2019-06.csv"
# new_york_data = pd.read_csv(file_name)
# new_york_data.drop(columns=["VendorID", "RatecodeID", "passenger_count", "extra", "mta_tax", "tip_amount", "tolls_amount", "ehail_fee", "improvement_surcharge", "total_amount", "payment_type", "trip_type", "congestion_surcharge"], axis=1, inplace=True)
# new_york_data["order_time"] = new_york_data["lpep_dropoff_datetime"] - new_york_data["lpep_pickup_datetime"]
new_york_temp_result_file = os.path.join(result_dir, "{0}_temp.csv".format("manhattan"))
temp_file = open(new_york_temp_result_file, 'w')
manhattan_zone_id_set = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75,
                         79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125,
                         127, 128, 137, 140, 141, 142, 143, 144, 148, 151,
                         152, 158, 161, 162, 163, 164, 166, 170, 186, 209,
                         211, 224, 229, 230, 231, 232, 233, 234, 236, 237,
                         238, 239, 243, 244, 246, 249, 261, 262, 263]
temp_file.write(",".join(["day", "request_time", "order_distance", "order_fare", "pick_index", "drop_index", "order_time"]) + "\n")
# print(new_york_data.columns)
cnt = 0
total = 0
for color in [green, yellow]:
    for csv_iterator in pd.read_table("./{0}_tripdata_2019-06.csv".format(color), chunksize=chunk_size, iterator=True):
        # print(csv_iterator)
        for line in csv_iterator.values:
            total += 1
            s = line[0].split(',')
            # print(s)
            # print(s)
            month = s[1].split(' ')[0].split('-')[1]
            date = s[1].split(' ')[0].split('-')[-1]
            # print(s[1].split(' ')[1])
            request_time = time2int(s[1].split(' ')[1])
            order_time = str(int(time2int(s[2].split(' ')[1])) - int(request_time))
            # 剔除订单时间过长的异常订单
            if int(order_time) > 3600:
                continue
            s[0] = date
            # 剔除日期不在六月的订单
            if int(month) != 6:
                continue
            s[1] = request_time
            if color == green:
                # 剔除乘客数量为0的订单
                if int(s[7]) == 0:
                    continue
                s[2] = s[8]
                s[3] = s[9]
                s[4] = s[5]
                s[5] = s[6]
                s[6] = order_time
                s = s[:7]
                # 剔除不在曼哈顿地区的订单；剔除起始点终点为同一个区域的订单；剔除行程距离为0的订单; 剔除行程时间过短的订单
                if int(s[4]) == int(s[5]) or \
                        int(s[4]) not in manhattan_zone_id_set or \
                        int(s[5]) not in manhattan_zone_id_set or float(s[2]) == 0 or int(s[6]) < 60:
                    continue
                temp_file.write(','.join(s) + "\n")
                cnt += 1
            else:
                # 剔除乘客数量为0的订单
                if int(s[3]) == 0:
                    continue
                s[2] = s[4]
                s[3] = s[10]
                s[4] = s[7]
                s[5] = s[8]
                s[6] = order_time
                s = s[:7]
                if int(s[4]) == int(s[5]) or \
                        int(s[4]) not in manhattan_zone_id_set or \
                        int(s[5]) not in manhattan_zone_id_set or float(s[2]) == 0 or int(s[6]) < 60:
                    continue
                # print(s)
                temp_file.write(','.join(s) + "\n")
                cnt += 1
            # print(s)
            # print(request_time)
            # print(order_time)
            # print(date)
print(cnt)
print(total)
temp_file.close()
