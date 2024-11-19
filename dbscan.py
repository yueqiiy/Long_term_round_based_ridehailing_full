import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import dbscan
income = []
x = []
y = []
featureList = ['income']

with open("result/laf_matching_WEIGHT_21/LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH_ROAD_MODE_0_3000_60_68400_75600.pkl", "rb") as file:
    result = pickle.load(file)
    cnt = 0
    for item in result[19]:
        x.append(cnt)
        y.append(float(item[2]))
        temp = [float(item[2])]
        income.append(temp)
        cnt += 1

core_samples,cluster_ids = dbscan(income, eps = 0.3, min_samples=2)
# cluster_ids中-1表示对应的点为噪声点

print(core_samples)
print(cluster_ids)
# df = pd.DataFrame(np.c_[income,cluster_ids],columns = ['feature1','feature2','cluster_id'])
# df['cluster_id'] = df['cluster_id'].astype('i2')
#
# df.plot.scatter('feature1','feature2', s = 100,
# c = list(df['cluster_id']),cmap = 'rainbow',colorbar = False,
# alpha = 0.6,title = 'sklearn DBSCAN cluster result')
import matplotlib.pyplot as plt
colors = []
for item in cluster_ids:
    if item == 0:
        colors.append(20)
    elif item == 1:
        colors.append(40)
    elif item == 2:
        colors.append(60)
    elif item == 3:
        colors.append(80)
    else :
        colors.append(100)


plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()
plt.show()

