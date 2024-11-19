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
    print(legends)
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
    for i in range(0, repeats):
        # i = 1
        # print("i = ",i)
        income = []
        fairList = []

        j = 0
        while j < hours:
            fairList.append([])
            j += 1

        with open("LONG_TERM_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
                  .format(i, num),
                "rb") as file:
            result = pickle.load(file)
            running_time.append(np.sum(result[10]))
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
    jain_plot(jain_trend)

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

            with open("old_linux/LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_ROAD_MODE_{0}_{1}_60_64800_79200.pkl"
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
    print("total_income = ",total_income)
    print("total_income_std = ",total_income_std)
    print("fairness = ",fairness)
    print("fairness_std = ",fairness_std)
    print("run_time = ",run_time)
    print("run_time_std = ",run_time_std)
    print("the_worst_sum = ",the_worst_sum)
    print("the_worst_sum_std = ",the_worst_sum_std)
    print("service_ratio = ",service_ratio_sum)
    print("service_ratio_std = ",service_ratio_sum_std)




# print_result(1500)

# print_result(2000)
#
# print_result(2500)
#
# print_result(3000)
summary_result(1500,3500)

# total_income =  [117397.60797048181, 145931.935983094, 164943.2920046926, 181625.16072784935, 194753.9358061275]
# total_income_std =  [2517.1259900945893, 960.083097426436, 891.2999505271822, 1053.4110438919934, 1454.5803240035189]
# fairness =  [1726.6014720358246, 2497.0245695833796, 3720.0084808517618, 4896.906330434375, 6508.478734286869]
# fairness_std =  [110.37854367551628, 92.05591808927316, 271.32833895752424, 113.00015002700167, 137.90296771437528]
# run_time =  [1248.1520812034607, 1490.1954468250274, 1742.1630858898163, 2005.6096941947937, 2374.6919572353363]
# run_time_std =  [22.88997613776237, 42.14160466226687, 18.641831436511776, 13.544583853649335, 28.30579855420808]
# the_worst_sum =  [40.96888060199027, 37.64307419946921, 27.078951773641926, 22.45655519579382, 15.671394744858096]
# the_worst_sum_std =  [5.326863723664788, 1.1782671672118998, 4.841349009357844, 1.431767363585901, 2.247586523379214]
# service_ratio =  [0.478464281967901, 0.6289059058009021, 0.7359236336934858, 0.8236441833630547, 0.8786195321514738]
# service_ratio_std =  [0.005767608878037903, 0.0034989903540055727, 0.00560151431339455, 0.002408782822064721, 0.0035363328872388966]


