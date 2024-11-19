# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/8 7:49 下午
import pickle
import random
import time
from collections import defaultdict

from env import vehicle
from env.location import VehicleLocation
from env.vehicle import Vehicle
from utility import Day
# test_str = "4,12,13,24,41,42,43,45,48,50,68,74,75,79,87,88,90,100,103,104,105," \
#            "107,113,114,116,120,125,127,128,137,140,141,142,143,144,148,151,152,153,158,161,162,163,164" \
#            ",166,170,186,194,202,209,211,224,229,230,231,232,233,234,236,237,238,239,243,244,246,249,261,262,263"
# print(', '.join(test_str.split(',')))
# test
# test_str

# t1 = time.time()
# for i in range(800):
#     print(i)
#     time.sleep(0.1)
# t2 = time.time()
# print("运行时间")
# print(t2 - t1)

# a = {'a':{'b':2,'c':3,'e':5},'c':{'d':3},'e':{'f':4}}
# if 'f' in a['a'].keys():
#     print('TRUE')
# else:
#     print('FALSE')
# p = VehicleLocation(1)
# vehicles = Vehicle(1,p,3)
# vehicles.set_vehicle_per_hour_income(68401,25.1)
# vehicles1 = Vehicle(2,p,2)
# vehicles1.set_vehicle_per_hour_income(68401,24.0)
# vehicles2 = Vehicle(3,p,1)
# vehicles2.set_vehicle_per_hour_income(68401,26.0)
#
# listV = [vehicles.per_hour_income,vehicles2.per_hour_income,vehicles1.per_hour_income]
# print(listV)
# listV = [vehicles,vehicles2,vehicles1]
# print(listV)

# listV.sort(key=lambda x: x.per_hour_income)
# print(listV)
# print(listV[1].per_hour_income)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
income = []
x = []
y = []
featureList = ['income']

with open(
        "result/fair_cluster_levelvehicle_state_matching_method/LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5_result_09_ROAD_MODE_0_1500_60_64800_79200.pkl", "rb") as file:
    result = pickle.load(file)
# with open("result/Fair_Augmentation_Matching_4CLASS_online/LONG_TERM_WITH_FAIR_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_0_2500_60_64800_79200.pkl", "rb") as file:
#     result = pickle.load(file)
# with open("result/Fair_Cluster_Matching/LONG_TERM_WITH_NO_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_0_3500_60_68400_75600.pkl", "rb") as file:
#     result = pickle.load(file)
    cnt = 0
    for item in result[19]:
        x.append(cnt)
        y.append(float(item[2]))
        temp = [float(item[2])]
        income.append(temp)
        cnt += 1

mdl = pd.DataFrame.from_records(income, columns=featureList)
mdlNew = np.array(mdl[['income']])
print(income)
print(type(income))
clf = KMeans(n_clusters=4)
clf.fit(mdlNew)
print(clf.labels_[0])
mdl['label'] = clf.labels_

y_ = clf.predict(mdlNew)
index = np.argwhere(y_ == 0).reshape(-1)
print("labels = ",mdl['label'][index])
c = mdl['label'].value_counts()

print(mdl.values)
print(c)
print("centers: ", clf.cluster_centers_)
print(clf.labels_[0])

import matplotlib.pyplot as plt


colors = []
for item in clf.labels_:
    if item == 0:
        colors.append(20)
    elif item == 1:
        colors.append(40)
    elif item == 2:
        colors.append(60)
    elif item == 3:
        colors.append(80)


plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()
plt.show()



