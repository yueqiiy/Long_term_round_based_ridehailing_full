import csv
import math
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt


# 1500
# {'0.0018275623301098054': 499, '0.002284452912637257': 510, '0.0013706717475823545': 491}
# {'0.0018275623301098054': 40579.82431771224, '0.002284452912637257': 36857.52878819186, '0.0013706717475823545': 40647.5005162973}

# 2000
# {'0.0018275623301098054': 635, '0.002284452912637257': 689, '0.0013706717475823545': 676}
# {'0.0018275623301098054': 45589.137081967936, '0.002284452912637257': 34742.79054718573, '0.0013706717475823545': 52908.03481102077}

# {'0.0018275623301098054': 797, '0.002284452912637257': 876, '0.0013706717475823545': 827}
# {'0.0018275623301098054': 51102.579799444036, '0.002284452912637257': 28780.39609962383, '0.0013706717475823545': 63070.44872585114}



# csvFile = open("MDP_2500_id_income.csv", "r")
# reader = csv.reader(csvFile)
# head = next(reader)
# numList = []
# income = []
# resList = [0, 0, 0]
# costMap = {}
# costMapList = {}
# for item in reader:
#     if item[1] not in costMap:
#         costMap[item[1]] = 1
#         costMapList[item[1]] = float(item[2])
#     else:
#         costMap[item[1]] += 1
#         costMapList[item[1]] += float(item[2])
#
# print(costMap)
# print(costMapList)
# {'0.0018275623301098054': 797, '0.002284452912637257': 876, '0.0013706717475823545': 827}

def statistic(fileneme: str):
    csvFile = open(fileneme, "r")
    reader = csv.reader(csvFile)
    head = next(reader)
    income = []
    resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for item in reader:
        income.append(float(item[2]))
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
    print(income)
    print("方差: ", np.var(income))
    print("均值等于: ",np.mean(income))
    print("总体标准差: ", np.std(income))
    print("总和: ", np.sum(income))
    print(np.max(income))
    print(np.min(income))
    x = range(10, 111, 10)
    plt.title(fileneme+'chart')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, resList)
    plt.show()

# Bipartite_Matching
# vehicle number:  1500
# total passenger payment =  118084.85362220144
# total cost =  65685.09938722567
# vehicle number:  2000
# total passenger payment =  133239.96244017442
# total cost =  75252.66231970002
# vehicle number:  2500
# total passenger payment =  142953.42462491902
# total cost =  80805.1493981033
# vehicle number:  3000
# total passenger payment =  149574.5928186262
# total cost =  84465.0622716278
# vehicle number:  3500
# total passenger payment =  154080.92717652398
# total cost =  87284.9210427281

# 方差:  564.8031025430762
# 均值等于:  78.7232357481343
# 总体标准差:  23.765586517969133
# 总和:  118084.85362220145
# 121.43206959912331
# 0.0

# statistic("MDP_1500_id_income.csv")

# 方差:  1019.4996723295995
# 均值等于:  57.18136984996761
# 总体标准差:  31.929604951041902
# 总和:  142953.42462491902
# 110.02778165913135
# 0.0
# statistic("MDP_2500_id_income.csv")

# 方差:  1093.8636767356886
# 均值等于:  49.85819760620873
# 总体标准差:  33.07360997435401
# 总和:  149574.5928186262
# 110.48247217233106
# 0.0
# statistic("MDP_3000_id_income.csv")

# 方差:  452.597053765391
# 均值等于:  50.26763470152785
# 总体标准差:  21.274328515029353
# 总和:  150802.90410458355
# 106.32554525488129
# 0.0
# statistic("recost_MDP_3000_id_income.csv")

# 方差:  409.0173382916684
# 均值等于:  57.822205218277404
# 总体标准差:  20.224177073287024
# 总和:  144555.51304569351
# 111.19469101682981
# 0.0
# statistic("recost_MDP_2500_id_income.csv")

# 方差:  425.9364884415151
# 均值等于:  79.52555478569364
# 总体标准差:  20.63822881066869
# 总和:  119288.33217854047
# 127.4614964600428
# statistic("recost_MDP_1500_id_income.csv")

def print_result(num):
    income = []
    resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tmp = []
    inList = []
    for i in range(0, 5):
        with open("../result/Bipartite_Matching/LONG_TERM_WITH_MDP_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl".format(
                        i, num),
                "rb") as file:
            result = pickle.load(file)
            for item in result[19]:
                income.append(float(item[2]))
                tmp.append(float(item[2]))
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
    # print(income)
    print("方差: ", np.var(income))
    print("均值等于: ", np.mean(income))
    print("总体标准差: ", np.std(income))
    print("总和: ", np.sum(income))
    print("平均总收益", np.mean(inList))
    print("max = ", np.max(income))
    print("min = ", np.min(income))
    s = 0
    max_in = float(np.max(income))
    for v in income:
        s += v * v
    print("JainIndex = ", (np.sum(income) * np.sum(income)) / (num * 5 * s))
    x = range(10, 111, 10)
    plt.title("Bipartite_Matching"+str(num))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, resList)
    plt.show()

# 方差:  217.37257667653614
# 均值等于:  34.930843145232174
# 总体标准差:  14.743560515578865
# 总和:  261981.3235892413
# 平均总收益 52396.26471784826
# 74.69938371878453
# 0.0
# JainIndex =  0.8487881212786267
# print_result(1500)

# 方差:  251.8992047346618
# 均值等于:  28.979401360626937
# 总体标准差:  15.871332796418258
# 总和:  289794.0136062694
# 平均总收益 57958.80272125389
# max =  71.0948129413164
# min =  0.0
# JainIndex =  0.7692607197245933
# print_result(2000)

# 方差:  268.2021957008262
# 均值等于:  24.867193044065303
# 总体标准差:  16.376879913488594
# 总和:  310839.9130508163
# 平均总收益 62167.982610163264
# max =  66.61367149431976
# min =  -0.08909673973022336
# JainIndex =  0.6974865761507617
# print_result(2500)

# 方差:  270.27221078085154
# 均值等于:  21.719988342798587
# 总体标准差:  16.43995774875506
# 总和:  325799.8251419788
# 平均总收益 65159.96502839576
# max =  74.21315472456824
# min =  -0.08909673973022336
# JainIndex =  0.6357665151574352
# print_result(3000)


# 方差:  266.3291749708182
# 均值等于:  19.163863842998623
# 总体标准差:  16.31959481638004
# 总和:  335367.6172524759
# 平均总收益 67073.52345049518
# max =  69.15783300386641
# min =  0.0
# JainIndex =  0.5796458600841532
# print_result(3500)

def print_each_cost_vehicle(num):
    income = {'0.0013706717475823545': 0, '0.0018275623301098054': 0, '0.002284452912637257': 0}
    vnum = {'0.0013706717475823545': 0, '0.0018275623301098054': 0, '0.002284452912637257': 0}
    for i in range(0, 5):
        with open(
                "../result/Fair_Augmentation_Matching_4CLASS/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl".format(
                        i, num),
                "rb") as file:
            result = pickle.load(file)
            for item in result[19]:
                income[str(item[1])] += float(item[2])
                vnum[str(item[1])] += 1
    listq = ['0.0013706717475823545', '0.0018275623301098054', '0.002284452912637257']
    for i in range(0,3):
        print(income[listq[i]]/vnum[listq[i]])

        # "../result/Bipartite_Matching/LONG_TERM_WITH_MDP_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl"
        # 2500  {'0.0013706717475823545': 158281.69370956073, '0.0018275623301098054': 103399.94902002467, '0.002284452912637257': 49158.27032123055}
print_each_cost_vehicle(2500)


# 37.698655263393356
# 37.2749858009407
# 40.29814453296948
# "../result/Fair_Augmentation_Matching_4CLASS_nostateValue/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl"
#  2500 {'0.0013706717475823545': 157504.98169045744, '0.0018275623301098054': 153572.94149987568, '0.002284452912637257': 169332.80332753775}


# {'0.0013706717475823545': 157531.348048865, '0.0018275623301098054': 153666.8925861204, '0.002284452912637257': 175596.66977337605}
# "../result/Fair_Augmentation_Matching_gama/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl"

# {'0.0013706717475823545': 162165.60175867067, '0.0018275623301098054': 159103.49734140857, '0.002284452912637257': 178867.52235693563}
# "../result/Fair_Augmentation_Matching_4CLASS/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl"

# {'0.0013706717475823545': 135719.89683644573, '0.0018275623301098054': 113029.00025498163, '0.002284452912637257': 108883.96386836171}
# "../result/Fair_Augmentation_Matching_4CLASS_nostate_nofairmatch/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl"

# {'0.0013706717475823545': 135574.91473018142, '0.0018275623301098054': 115646.52102778584, '0.002284452912637257': 110434.3683370002}
# "../result/Fair_Augmentation_Matching_4CLASS_state_nofairmatch_nodispatch/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_{0}_{1}_60_68400_75600.pkl"


