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

def jain_plot(jain):
    fig,ax = plt.subplots()
    x = range(0, len(jain[0]))
    legends = []
    # plt.suptitle("FAIR" + str(num))
    for i in range(len(jain)):
        y = jain[i]
        temp = "Line_{0}".format(i)
        legends.append(temp)
        ax.plot(x,y)

    ax.legend(legends)
    ax.set_xlabel("rounds")
    ax.set_ylabel("jain")
    plt.show()

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
    resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    hours = int((MAX_REQUEST_TIME - MIN_REQUEST_TIME) / (TIME_SLOT * TIME_SLOT))
    repeats = 5
    the_worst =[]
    percent = 0.1
    jain_trend = []
    empty = []
    for i in range(0, repeats):
        # i = 1
        # print("i = ",i)
        income = []
        fairList = []
        j = 0
        while j < hours:
            fairList.append([])
            j += 1

        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_1500_09_700_300_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_linux_2000_result_09_time_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_1000_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_2500_1000_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_2000v_1000ep_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_1500_09_1000_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_3000_1000_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_all_2400_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_2500_1000ep_10per_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_02_2000_1000_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_linux_2000_result_09_time_max_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
        # with open("LONG_TERM_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                  .format(i, num),
                "rb") as file:
            result = pickle.load(file)
            running_time.append(np.sum(result[10]))
            empty.append(result[8])
            jain_trend.append(result[20])
            for item in result[19]:
                income.append(float(item[2]))
                tmp.append(float(item[2]))
                t = 0
                for temp in item[4]:  # income中记录了每小时的司机收入，fairList每一行对应一个司机的所有的小时收入
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

                if float(item[2]) < 110:
                    resList[10] += 1
                    continue

                if float(item[2]) < 120:
                    resList[11] += 1
                    continue

                if float(item[2]) < 130:
                    resList[12] += 1
                    continue

                if float(item[2]) < 140:
                    resList[13] += 1
                    continue

                resList[14] += 1

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
        JainIndex.append((np.sum(income) * np.sum(income)) / (num * s))
        income.sort()
        worst = 0
        for i in range(int(percent * num)):
            worst += income[i]
        the_worst.append(worst/(percent * num))

    jain_plot(empty)
    print("F = ",F)
    print("income_sum = ",income_sum)
    # print("均值等于: ", np.mean(income_mean))
    # print("均值 std: ", np.std(income_mean))
    print("平均总收益", np.mean(income_sum))
    print("平均总收益 std: ", np.std(income_sum))
    # print("最大值: ", np.max(income_max))
    # print("最小值: ", np.min(income_min))
    print("F = ", np.mean(F))
    print("F std = ", np.std(F))
    print("running_time = ", np.mean(running_time))
    print("running_time std = ", np.std(running_time))
    print("JainIndex = ", np.mean(JainIndex))
    print("the worst = ",the_worst)
    print("最差的",percent*100,"%司机平均收入: ",np.mean(the_worst))
    x = range(10, 161, 10)
    # plt.suptitle("FAIR" + str(num))
    plt.title("FAIR" + str(num))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, resList)
    plt.show()
    # jain_plot(jain_trend)

# def summary_result(low,high):
#     total_income = []
#     total_income_std = []
#     fairness = []
#     fairness_std = []
#     run_time = []
#     run_time_std = []
#     the_worst_sum = []
#     the_worst_sum_std = []
#     service_ratio_sum = []
#     service_ratio_sum_std = []
#     empty_vehicle_ratio_sum = []
#     empty_vehicle_ratio_sum_std = []
#     for num in range(low,high+1,500):
#         tmp = []
#         inList = []
#         F = []
#         income_mean = []
#         income_sum = []
#         income_max = []
#         income_min = []
#         JainIndex = []
#         running_time = []
#         service_ratio = []
#         resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         hours = int((MAX_REQUEST_TIME - MIN_REQUEST_TIME) / (TIME_SLOT * TIME_SLOT))
#         repeats = 1
#         the_worst =[]
#         percent = 0.1
#         jain_trend = []
#         empty_vehicle_ratio = []
#         for i in range(0, repeats):
#             # i = 1
#             # print("i = ",i)
#             income = []
#             fairList = []
#
#             j = 0
#             while j < hours:
#                 fairList.append([])
#                 j += 1
#
#             # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_2500_1000ep_10per_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
#             # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_all_2400_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
#             # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_ROAD_MODE_{0}_{1}_60_64800_79200_elbow.pkl"
#             # with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_ROAD_MODE_{0}_{1}_60_64800_79200_elbow_lot.pkl"
#             # with open("elbow/LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
#             with open("near_lot/LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_all_2400_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
#                       .format(i, num),
#                     "rb") as file:
#                 result = pickle.load(file)
#                 running_time.append(np.sum(result[10]))
#                 # jain_trend.append(result[20])
#                 empty_vehicle_ratio.append(np.mean(result[8]) * 100)
#                 service_ratio.append(result[7][-1]*100)
#
#                 for item in result[19]:
#                     income.append(float(item[2]))
#                     tmp.append(float(item[2]))
#                     t = 0
#                     for temp in item[4]:  # income中记录了每小时的司机收入，fairList每一行对应一个司机的所有的小时收入
#                         fairList[t].append(float(temp))
#                         t += 1
#
#
#             tmp = []
#             j = 0
#             income_median = []  # 收入中位数
#             while j < hours:
#                 fairList[j].sort()
#                 mid = int(len(fairList[0]) / 2)
#                 income_median.append(fairList[j][mid])
#                 j += 1
#             fw = []
#             for te in range(num):
#                 temp = 0
#                 for t in range(hours):
#                     temp += (fairList[t][te] / income_median[t])
#                     t += 1
#                 fw.append(temp / hours)
#             max_fw = max(fw)
#             temp_F = 0
#             for fwi in fw:
#                 if fwi == 0:
#                     fwi += 0.001
#                 temp_F -= math.log((fwi / max_fw))
#             F.append(temp_F)
#             income_sum.append(np.sum(income))
#             income_mean.append(np.mean(income))
#             income_max.append(np.max(income))
#             income_min.append(np.min(income))
#             s = 0
#             for v in income:
#                 s += v * v
#             JainIndex.append((np.sum(income) * np.sum(income)) / (num * s))
#             income.sort()
#             worst = 0
#             for i in range(int(percent * num)):
#                 worst += income[i]
#             the_worst.append(worst/(percent * num))
#
#         total_income.append(np.mean(income_sum))
#         total_income_std.append(np.std(income_sum))
#         fairness.append(np.mean(F))
#         fairness_std.append(np.std(F))
#         run_time.append(np.mean(running_time))
#         run_time_std.append(np.std(running_time))
#         the_worst_sum.append(np.mean(the_worst))
#         the_worst_sum_std.append(np.std(the_worst))
#         service_ratio_sum.append(np.mean(service_ratio))
#         service_ratio_sum_std.append(np.std(service_ratio))
#         empty_vehicle_ratio_sum.append(np.mean(empty_vehicle_ratio))
#         empty_vehicle_ratio_sum_std.append(np.std(empty_vehicle_ratio))
#     print("total_income = ", total_income)
#     print("total_income_std = ", total_income_std)
#     print("fairness = ", fairness)
#     print("fairness_std = ", fairness_std)
#     print("run_time = ", run_time)
#     print("run_time_std = ", run_time_std)
#     print("the_worst_sum = ", the_worst_sum)
#     print("the_worst_sum_std = ", the_worst_sum_std)
#     print("service_ratio = ", service_ratio_sum)
#     print("service_ratio_std = ", service_ratio_sum_std)
#     print("empty_vehicle_ratio_sum = ", empty_vehicle_ratio_sum)
#     print("empty_vehicle_ratio_sum_std = ", empty_vehicle_ratio_sum_std)


def summary_result(low,high):
    total_income = []
    total_income_std = []
    fairness = []
    fairness_std = []
    run_time = []
    run_time_std = []
    the_worst_sum = []
    the_worst_sum_std = []
    service_ratio_sum = []
    service_ratio_sum_std = []
    for num in range(low,high+1,500):
        tmp = []
        inList = []
        F = []
        income_mean = []
        income_sum = []
        income_max = []
        income_min = []
        JainIndex = []
        running_time = []
        service_ratio = []
        resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        hours = int((MAX_REQUEST_TIME - MIN_REQUEST_TIME) / (TIME_SLOT * TIME_SLOT))
        repeats = 5
        the_worst =[]
        percent = 0.1
        jain_trend = []
        for i in range(0, repeats):
            # i = 1
            # print("i = ",i)
            income = []
            fairList = []

            j = 0
            while j < hours:
                fairList.append([])
                j += 1

            # with open("near_lot_1/LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_all_2400_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
            with open("LONG_TERM_WITH_NEW_FAIR_EMPTY_train_09_all_2400_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                      .format(i, num),
                    "rb") as file:
                result = pickle.load(file)
                running_time.append(np.sum(result[10]))
                # jain_trend.append(result[20])
                service_ratio.append(((np.sum(result[6])/np.sum(result[5]))))
                for item in result[19]:
                    income.append(float(item[2]))
                    tmp.append(float(item[2]))
                    t = 0
                    for temp in item[4]:  # income中记录了每小时的司机收入，fairList每一行对应一个司机的所有的小时收入
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
                    fwi += 0.001
                temp_F -= math.log((fwi / max_fw))
            F.append(temp_F)
            income_sum.append(np.sum(income))
            income_mean.append(np.mean(income))
            income_max.append(np.max(income))
            income_min.append(np.min(income))
            s = 0
            for v in income:
                s += v * v
            JainIndex.append((np.sum(income) * np.sum(income)) / (num * s))
            income.sort()
            worst = 0
            for i in range(int(percent * num)):
                worst += income[i]
            the_worst.append(worst/(percent * num))

        total_income.append(np.mean(income_sum))
        total_income_std.append(np.std(income_sum))
        fairness.append(np.mean(F))
        fairness_std.append(np.std(F))
        run_time.append(np.mean(running_time))
        run_time_std.append(np.std(running_time))
        the_worst_sum.append(np.mean(the_worst))
        the_worst_sum_std.append(np.std(the_worst))
        service_ratio_sum.append(np.mean(service_ratio))
        service_ratio_sum_std.append(np.std(service_ratio))
    print("total_income = ", total_income)
    print("total_income_std = ", total_income_std)
    print("fairness = ", fairness)
    print("fairness_std = ", fairness_std)
    print("run_time = ", run_time)
    print("run_time_std = ", run_time_std)
    print("the_worst_sum = ", the_worst_sum)
    print("the_worst_sum_std = ", the_worst_sum_std)
    print("service_ratio = ", service_ratio_sum)
    print("service_ratio_std = ", service_ratio_sum_std)


# print_result(1500)
#
# print_result(2000)
#
# print_result(2500)
#
# print_result(3000)

# print_result(3500)

summary_result(1500,3500)


