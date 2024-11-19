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
    print("vehicle num = ",num)
    tmp = []
    inList = []
    F = []
    income_mean = []
    income_sum = []
    income_max = []
    income_min = []
    JainIndex = []
    running_time = []
    hours = int((MAX_REQUEST_TIME - MIN_REQUEST_TIME) / (TIME_SLOT * TIME_SLOT))
    repeats = 1
    for i in range(0, repeats):
        epoch = 4
        print("epoch = ",epoch)
        income = []
        fairList = []
        j = 0
        while j < hours:
            fairList.append([])
            j += 1

        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_1500_result_09_time_700_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_1500_result_09_time_700_final_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_085_700_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_088_700_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_2000_result_09_time_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_result_09_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_1500_09_700_300_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_085_700_500_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_1500_09_1000_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_2000_09_1000_final_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                        .format(epoch, num),
                "rb") as file:
            result = pickle.load(file)
            running_time.append(np.sum(result[10]))
            for item in result[19]:
                income.append(float(item[2]))
                tmp.append(float(item[2]))
                t = 0
                for temp in item[4]:
                    fairList[t].append(float(temp))
                    t += 1

        tmp = []
        j = 0
        income_median = []  # 收入中位数
        while j < hours:
            fairList[j].sort()
            mid = int(len(fairList[0]) / 2)
            income_median.append(fairList[j][mid])
            j += 1
        fw = []
        for te in range(num):
            temp = 0
            for t in range(hours):
                temp += (fairList[t][te] / income_median[t])
                t += 1
            fw.append(temp / hours)
        max_fw = max(fw)
        temp_F = 0
        for fwi in fw:
            if fwi == 0:
                fwi += 0.0001
            temp_F -= math.log((fwi / max_fw))
        F.append(temp_F)
        income_sum.append(np.sum(income))
        income_mean.append(np.mean(income))
        income_max.append(np.max(income))
        income_min.append(np.min(income))
        s = 0
        for v in income:
            s += v * v
        JainIndex.append((np.sum(income) * np.sum(income)) / (num * repeats * s))

    print("F = ",F)
    print("均值等于: ", np.mean(income_mean))
    print("均值 std: ", np.std(income_mean))
    print("平均总收益", np.mean(income_sum))
    print("平均总收益 std: ", np.std(income_sum))
    print("最大值: ", np.max(income_max))
    print("最小值: ", np.min(income_min))
    print("F = ", np.mean(F))
    print("F std = ", np.std(F))
    print("running_time = ", np.mean(running_time))
    print("running_time std = ", np.std(running_time))
    print("JainIndex = ", np.mean(JainIndex))



# 均值等于:  68.3149118016684
# 均值 std:  1.434537658407163
# 平均总收益 102472.3677025026
# 平均总收益 std:  2151.8064876107464
# 最大值:  142.25118572244287
# 最小值:  0.0
# F =  2971.8191089850948
# F std =  147.6742755171341
# running_time =  634.1310980319977
# running_time std =  26.541465823724614
# JainIndex =  0.16914386563770858
# print_result(1500)

# 均值等于:  63.80068554443699
# 均值 std:  0.6326766274886104
# 平均总收益 127601.37108887397
# 平均总收益 std:  1265.3532549772217
# 最大值:  142.7126452107956
# 最小值:  0.0
# F =  4372.273503678578
# F std =  119.62333581747508
# running_time =  829.0204803466797
# running_time std =  19.578960258507355
# JainIndex =  0.166210110690559
# print_result(2000)

# 均值等于:  63.80068554443699
# 均值 std:  0.6326766274886104
# 平均总收益 127601.37108887397
# 平均总收益 std:  1265.3532549772217
# 最大值:  142.7126452107956
# 最小值:  0.0
# F =  4372.273503678578
# F std =  119.62333581747508
# running_time =  829.0204803466797
# running_time std =  19.578960258507355
# JainIndex =  0.166210110690559
# print_result(2500)

# 均值等于:  52.12546427906276
# 均值 std:  0.2599906447824289
# 平均总收益 156376.39283718826
# 平均总收益 std:  779.9719343472852
# 最大值:  141.75664019187082
# 最小值:  0.0
# F =  9197.339284719481
# F std =  208.66825886830284
# running_time =  1365.210368680954
# running_time std =  17.129409934656923
# JainIndex =  0.14861648656406728
# print_result(3000)

# 均值等于:  47.594060332391265
# 均值 std:  0.3923832739119788
# 平均总收益 166579.21116336944
# 平均总收益 std:  1373.3414586919218
# 最大值:  137.3813995384632
# 最小值:  0.0
# F =  12961.00256858137
# F std =  369.7517314082395
# running_time =  1687.5083512306214
# running_time std =  22.297785088549052
# JainIndex =  0.13680064999412261
# print_result(3500)


