import csv
import math
import os

import numpy as np
import pickle
from matplotlib import pyplot as plt
# result = [
#     0 self.social_welfare_trend,
#     1 self.social_cost_trend,
#     2 self.total_passenger_payment_trend,
#     3 self.total_passenger_utility_trend,
#     4 self.platform_profit_trend,
#     5 self.total_orders_number_trend,
#     6 self.serviced_orders_number_trend,
#     7 self.accumulate_service_ratio_trend,
#     8 self.empty_vehicle_ratio_trend,
#     9 self.bidding_time_trend,
#     10 self.running_time_trend,
#     11 self.accumulate_service_distance_trend,
#     12 self.accumulate_random_distance_trend,
#     13 self.each_orders_wait_time_trend,
#     14 self.each_orders_service_time_trend,
#     15 self.each_vehicles_cost,
#     16 self.each_vehicles_finish_order_number,
#     17 self.each_vehicles_service_distance,
#     18 self.each_vehicles_random_distance,
#     19 self.each_vehicles_income   [vehicle.vehicle_id, vehicle.unit_cost, vehicle.incomeacc, vehicle.income_level, vehicle.income]
# ]
from setting import MAX_REQUEST_TIME, MIN_REQUEST_TIME, TIME_SLOT


def print_result(num):
    income = []
    resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tmp = []
    inList = []
    F = []
    running_time = 0
    hours = int((MAX_REQUEST_TIME - MIN_REQUEST_TIME) / (TIME_SLOT * TIME_SLOT))
    repeats = 5
    for i in range(0, repeats):
        fairList = []
        j = 0
        while j < hours:
            fairList.append([])
            j += 1

        with open("LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_57600_79200.pkl"
        # with open("LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_28800_72000.pkl"
        # with open("LONG_TERM_WITH_NO_FAIRNESS_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                .format(i, num),
                "rb") as file:
            result = pickle.load(file)
            running_time += np.sum(result[10])
            for item in result[19]:
                income.append(float(item[2]))
                tmp.append(float(item[2]))
                t = 0
                for temp in item[4]:
                    fairList[t].append(float(temp))
                    t += 1
                if float(item[2]) < 10:
                    resList[0] += 1
                    continue

                if float(item[2]) < 20:
                    resList[1] += 1
                    continue

                if float(item[2]) < 30:
                    resList[2] += 1
                    continue

                if float(item[2]) < 40:
                    resList[3] += 1
                    continue

                if float(item[2]) < 50:
                    resList[4] += 1
                    continue

                if float(item[2]) < 60:
                    resList[5] += 1
                    continue

                if float(item[2]) < 70:
                    resList[6] += 1
                    continue

                if float(item[2]) < 80:
                    resList[7] += 1
                    continue

                if float(item[2]) < 90:
                    resList[8] += 1
                    continue

                if float(item[2]) < 100:
                    resList[9] += 1
                    continue
                resList[10] += 1
        inList.append(np.sum(tmp))
        tmp = []
        j = 0
        income_median = []# 收入中位数
        while j < hours:
            fairList[j].sort()
            mid = int(len(fairList[0])/2)
            income_median.append(fairList[j][mid])
            j += 1
        fw = []
        for te in range(num):
            temp = 0
            for t in range(hours):
                temp += (fairList[t][te] / (income_median[t]))
                t += 1
            fw.append(temp/hours)
        max_fw = max(fw)
        temp_F = 0
        for fwi in fw:
            if fwi == 0:
                fwi += 0.0001
            temp_F -= math.log((fwi/max_fw))
        F.append(temp_F)
    # print(income)
    print("方差: ", np.var(income))
    print("均值等于: ", np.mean(income))
    print("总体标准差: ", np.std(income))
    print("总和: ", np.sum(income))
    print("平均总收益", np.mean(inList))
    print("最大值: ", np.max(income))
    print("最小值: ", np.min(income))
    print("F = ",np.mean(F))
    print("running_time = ", running_time / repeats)
    s = 0
    for v in income:
        s += v * v
    print("JainIndex = ", (np.sum(income) * np.sum(income))/(num * repeats * s))
    x = range(10, 111, 10)
    plt.title("LAF"+str(num))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, resList)
    plt.show()



