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


def plot(num):
    running_time = 0
    hours = int((MAX_REQUEST_TIME - MIN_REQUEST_TIME) / (TIME_SLOT * TIME_SLOT))
    repeats = 1
    for i in range(0, repeats):
        fairList = []
        j = 0
        while j < hours:
            fairList.append([])
            j += 1

        with open(
                "LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                        .format(i, num),
                "rb") as file:
            result = pickle.load(file)
            y = result[10]
            x = range(0, len(result[10]), 1)
            plt.title("LAF"+str(num))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(x, y)
            plt.show()

# plot(2500)
# plot(3500)
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
    for i in range(0, repeats):
        # i = 1
        # print("i = ",i)
        income = []
        fairList = []

        j = 0
        while j < hours:
            fairList.append([])
            j += 1

        with open("LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                  .format(i, num),
                "rb") as file:
            result = pickle.load(file)
            running_time.append(np.sum(result[10]))
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


    print("F = ",F)
    print("income_sum = ",income_sum)
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
    print("the worst = ",the_worst)
    print("最差的",percent*100,"%司机平均收入: ",np.mean(the_worst))
    x = range(10, 161, 10)
    # plt.suptitle("FAIR" + str(num))
    plt.title("FAIR" + str(num))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, resList)
    plt.show()

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

            with open("LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
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


# print_result(2000)

# print_result(2500)



# print_result(3000)

# print_result(3500)
summary_result(1500,3500)

# total_income =  [118904.65821113523, 145439.5566172377, 164494.42659573973, 181194.96346197036, 193890.12890803005]
# total_income_std =  [803.9096291045975, 503.28532302180685, 477.0076173162045, 618.4548542387005, 1669.6230559865473]
# fairness =  [2418.7363817196956, 3449.497570472039, 4686.215219538962, 6021.83912798745, 7882.565264041899]
# fairness_std =  [204.14680287876843, 295.5121882397559, 274.50584101757687, 494.96081110753653, 501.0652171855428]
# run_time =  [721.2577828407287, 926.6045374393464, 1177.9186757087707, 1591.2399263858795, 1756.7065794467926]
# run_time_std =  [15.229399477523462, 23.553059681288087, 58.577658180044104, 38.158645986568324, 32.47678853685314]
# the_worst_sum =  [48.50678323531751, 42.968845632243, 35.199652655667364, 27.77985628897826, 20.41451929596702]
# the_worst_sum_std =  [1.3379060961472478, 1.2421054681507586, 1.053632389055269, 1.6171869798504275, 0.8520262139706701]
# service_ratio =  [0.46898143291723493, 0.6211517885240743, 0.7341655302632959, 0.8229854190705967, 0.8858197839085282]
# service_ratio_std =  [0.0041421708191213685, 0.00368554997025486, 0.0027913955645601663, 0.006164630184116948, 0.002169253864736271]

