# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/9 4:25 下午

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 每天的订单数量变化
def every_day_order_num():
    order_num = []
    for day in range(1, 31):
        order_data = pd.read_csv('./raw_data/temp/Manhattan/order_data_{:03d}.csv'.format(day))
        order_num.append(len(order_data))
    print(order_num)
    plt.grid()
    plt.plot(order_num)
    plt.show()


# 工作日和周末每个小时订单变化
def week_and_weekend_order_num():
    weekend = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
    week_day_num = []
    weekend_num = []
    for day in range(1, 31):
        day_num = np.zeros(24, np.int)
        order_day = pd.read_csv('./raw_data/temp/Manhattan/order_data_{:03d}.csv'.format(day))
        for i in range(24):
            demand = order_day[order_day.request_time >= i*3600]
            demand = demand[demand.request_time < (i+1) * 3600]
            # print(len(demand))
            day_num[i] = len(demand)
        if day in weekend:
            weekend_num.append(day_num)
        else:
            week_day_num.append(day_num)
    # print(len(week_day_num))
    # print(len(weekend_num))
    print(week_day_num)
    print(weekend_num)
    # 计算工作日和周末的每小时订单数量的均值和方差
    print(np.mean(week_day_num, axis=0))
    print(np.mean(weekend_num, axis=0))
    print(np.std(week_day_num, axis=0))
    print(np.std(weekend_num, axis=0))
    with open('./order_distribution.csv', 'w') as file:
        file.write(','.join(['第一行工作日订单均值', '第二行工作日订单标准差', '第三行周末订单均值', '第四行周末订单标准差'])+'\n')
        file.write(','.join([str(i) for i in np.mean(week_day_num, axis=0)])+'\n')
        file.write(','.join([str(i) for i in np.std(week_day_num, axis=0)])+'\n')
        file.write(','.join([str(i) for i in np.mean(weekend_num, axis=0)])+'\n')
        file.write(','.join([str(i) for i in np.std(weekend_num, axis=0)])+'\n')


week_and_weekend_order_num()

