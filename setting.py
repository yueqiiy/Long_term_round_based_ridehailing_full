# Round_based_ridehailing
# Author: Jackye
# Time : 2020/7/1 9:29 上午

import numpy as np

POINT_LENGTH = 0  # 计算小数点后面保留的精度
VALUE_EPS = 1E-8  # 浮点数相等的最小精度
INT_ZERO = 0
FLOAT_ZERO = 0.0
FIRST_INDEX = INT_ZERO
POS_INF = np.PINF
NEG_INF = np.NINF
MILE_TO_KM = 1.609344
MILE_TO_M = 1609.344
SECOND_OF_DAY = 86_400  # 24 * 60 * 60 一天有多少秒

ROAD_MODE = "ROAD_MODE"
GRID_MODE = "GRID_MODE"
EXPERIMENTAL_MODE = ROAD_MODE
NEAREST_MATCHING_METHOD = "Nearest_Matching"
FairClusterLevelVehicleStateMatchingMethod = "fair_cluster_levelvehicle_state_matching_method"
LafMatchingMethod = "laf_matching"
ILPMatchingMethod = "Ilp_matching"
LongTermILPMatchingMethod = "Long_Term_Ilp_matching"
LongTermMeanILPMatchingMethod = "Long_Term_Mean_ILP_Matching"
JainILPMatchingMethod = "Jain_ILP_MatchingMethod"
ReassignMethod = "Reassign"
WorstFirstMethod = "Worstfirst"
FairNoClusterMatchingMethod = "fair_no_cluster_matching"
Hungarian_Algorithm = "hungarian_algorithm"
MATCHING_METHOD = FairClusterLevelVehicleStateMatchingMethod
PKL_PATH = "train_09_all_2400"
# 一组参数实验的重复次数
MAX_REPEATS = 5 # 训练的时候设置为1
# 订单分配算法的执行时间间隔 单位 s. 如果是路网环境 [10 15 20 25 30], 如果是网格环境 默认为1.
TIME_SLOT = 60
# 距离精度误差, 表示一个车辆到某一个点的距离小于这一个数, 那么就默认这个车已经到这个点上了 单位 m. 如果是实际的路网一般取10.0m, 如果是网格环境一般取0.0.
DISTANCE_EPS = 10.0
# 模拟一天的时刻最小值/最大值 单位 s.
# 如果是路网环境 MIN_REQUEST_TIME <= request_time < MAX_REQUEST_TIME 并且有 MAX_REQUEST_TIME - MIN_REQUEST_TIME 并且可以整除 TIME_SLOT.
# 如果是网格环境 MIN_REQUEST_TIME = 0, MIN_REQUEST_TIME = 500.

# MIN_REQUEST_TIME, MAX_REQUEST_TIME = 19 * 60 * 60, 21 * 60 * 60
MIN_REQUEST_TIME, MAX_REQUEST_TIME = 18 * 60 * 60, 22 * 60 * 60
# 实验环境中的车辆数目
VEHICLE_NUMBER = 2500
# 实验环境中的车辆速度 单位 m/s. 对于任意的环境 VEHICLE_SPEED * TIME_SLOT >> DISTANCE_EPS. 纽约市规定是 (MILE_TO_KM * 25 / 3.6) m/s ≈ 10 m/s
# 但是根据资料纽约市的车辆速度只有7.2mph ~ 9.1mph 约等于 4.0 m/s (http://mini.eastday.com/a/190331120732879.html)
VEHICLE_SPEED = 4.0     # 通过平均速度计算得到速度为3.96m/s
# 实际无用 可以删除
N_SEATS = 4

# 是否进行状态信息记录 进行状态信息记录时 MAX_REPEATS = 1
LEARNING = False
LONG_TERM = True    # 是否是长期收益
EMPTY_VEHICLE_DISPATCH = True   # 是否进行空车调度
FAIR_DISPATCH = "FAIR"
FAIR_LEVEL_DISPATCH = "Fair_Level_dispatch"
LAF_DISPATCH = "LAF"
TEST_DISPATCH = "TEST"
NEW_FAIR_MATCH = "New_Fair_dispatch"
DISPATCH_METHOD = NEW_FAIR_MATCH


TYPED = True    # 是否区分车的成本类型 主要用来和2018kdd论文进行对比
TYPED_LEVEL_LEARNING_RESULT_FILE = "../result/learning/TYPED_LEVEL_STATE_VALUE_{0}_{1}_{2}.pkl".format(
    TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME)
TYPED_LEARNING_RESULT_FILE = "../result/learning/TYPED_STATE_VALUE_{0}_{1}_{2}.pkl".format(
    TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME)     # 学习结果文件路径
LEARNING_RESULT_FILE = "../result/learning/STATE_VALUE_{0}_{1}_{2}.pkl".format(
    TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME)     # 学习结果文件路径
GAMMA = 0.9     # 折扣因子
WEEKEND = [1, 2, 8, 9, 15, 16, 22, 23, 29, 30]
DAY = 3

# 订单缩放比
ORDER_NUMBER_RATIO = 1.0  # 就是实际生产出来的订单数目乘于一个比例
# 车辆油耗与座位数据存放地址
FUEL_CONSUMPTION_DATA_FILE = "../data/vehicle_data/fuel_consumption_and_seats.csv"
# 直接与此常数相乘可以得到单位距离的成本 $/m/(单位油耗)
VEHICLE_FUEL_COST_RATIO = 2.5 / 6.8 / MILE_TO_M
# 乘客最大等待时间可选范围 单位 s
WAIT_TIMES = [3 * 60, 4 * 60, 5 * 60, 6 * 60, 7 * 60, 8 * 60]

"""
    地图数据和订单数据
"""

# 与地理相关的数据存放点
Manhattan = "Manhattan"
GEO_NAME = Manhattan
GEO_DATA_FILE = {
    "base_folder": "../data/{0}/network_data".format(GEO_NAME),
    "graph_file": "{0}.graphml".format(GEO_NAME),
    "osm_id2index_file": "osm_id2index.pkl",
    "index2osm_id_file": "index2osm_id.pkl",
    "shortest_distance_file": "shortest_distance.npy",
    "same_zone_distance_file": "same_zone_distance.npy",
    "shortest_path_file": "shortest_path.npy",
    "adjacent_index_file": "adjacent_index.pkl",
    "access_index_file": "access_index.pkl",
    "near_zone_file": "near_zone.pkl",  # 划分区域地图时的邻近区域字典文件
    "adjacent_location_osm_index_file": "{0}/adjacent_location_osm_index.pkl".format(TIME_SLOT),
    "adjacent_location_driven_distance_file": "{0}/adjacent_location_driven_distance.pkl".format(TIME_SLOT),
    "adjacent_location_goal_index_file": "{0}/adjacent_location_goal_index.pkl".format(TIME_SLOT),
}

# 统计学处理后订单分布的模型数据存放地址
ORDER_DATA_FILES = {
    "pick_region_model": "../data/{0}/order_data/pick_region_model.pkl".format(GEO_NAME),
    "drop_region_model": "../data/{0}/order_data/drop_region_model.pkl".format(GEO_NAME),
    "demand_model_file": "../data/{0}/order_data/demand_model.npy".format(GEO_NAME),
    "demand_location_model_file": "../data/{0}/order_data/demand_location_model.npy".format(GEO_NAME),
    "demand_transfer_model_file": "../data/{0}/order_data/demand_transfer_model.npy".format(GEO_NAME),
    "unit_fare_model_file": "../data/{0}/order_data/unit_fare_model.npy".format(GEO_NAME),
}

# 生成订单、车辆的数据和结果存放路线
INPUT_VEHICLES_DATA_FILES = ["../data/input/vehicles_data/{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, i, VEHICLE_NUMBER, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in range(MAX_REPEATS)]
INPUT_ORDERS_DATA_FILES = ["../data/input/orders_data/{0}_{1}_{2}_{3}.csv".format(EXPERIMENTAL_MODE, i, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in range(MAX_REPEATS)]
SAVE_RESULT_FILES = ["../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(MATCHING_METHOD, EXPERIMENTAL_MODE, i, VEHICLE_NUMBER, TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in range(MAX_REPEATS)]

# 用于学习的订单数据
INPUT_LEARNING_ORDERS_DATA_FILES = ["../data/input/orders_data/{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, DAY, i, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in range(MAX_REPEATS)]
SAVE_LEARNING_RESULT_FILES = ["../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD, DAY, EXPERIMENTAL_MODE, i, VEHICLE_NUMBER, TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in range(MAX_REPEATS)]
