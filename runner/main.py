# planning使用的main文件
import sys
sys.path.append("/data/yueqi/Long_term_round_based_ridehailing_full")


import time

from setting import *
from runner.simulator import Simulator


def create_orders(_simulator: Simulator):
    print("create order data")
    for epoch in range(MAX_REPEATS):
        _simulator.create_order_env(INPUT_ORDERS_DATA_FILES[epoch])


def create_vehicles(_simulator: Simulator):
    print("create vehicle data: ", VEHICLE_NUMBER)
    for epoch in range(MAX_REPEATS):
        _simulator.create_vehicle_env(INPUT_VEHICLES_DATA_FILES[epoch])


def run_simulation(_simulator: Simulator):
    if LONG_TERM:
        print("long term optimization")
    else:
        print("short term optimization")
    if LEARNING:
        print("in learning process")
    else:
        print("in planning process")
    if EMPTY_VEHICLE_DISPATCH:
        print("activate empty vehicle dispatch")
        print("use {0} dispatch".format(DISPATCH_METHOD))
    print("current algorithm: ", MATCHING_METHOD)
    print("current time: ", MIN_REQUEST_TIME/3600, "-", MAX_REQUEST_TIME/3600)
    print("time slot: ", TIME_SLOT)
    for vehicle_num in range(1500, 3501, 500):
        print("vehicle num: ", vehicle_num)
        _INPUT_VEHICLES_DATA_FILES = [
            "../data/input/vehicles_data/{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, i, vehicle_num,
                                                                         MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in
            range(MAX_REPEATS)]
        if LONG_TERM:
            if EMPTY_VEHICLE_DISPATCH:
                if DISPATCH_METHOD == FAIR_DISPATCH:
                    _SAVE_RESULT_FILES = [
                        "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD,
                                                                               "LONG_TERM_WITH_STATE_7CLASS_NO_EMPTY_VEHICLE_DISPATCH",
                                                                               EXPERIMENTAL_MODE, i, vehicle_num,
                                                                               TIME_SLOT, MIN_REQUEST_TIME,
                                                                               MAX_REQUEST_TIME) for i
                        in
                        range(MAX_REPEATS)]
                elif DISPATCH_METHOD == NEW_FAIR_MATCH:
                    _SAVE_RESULT_FILES = [
                        "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}_elbow.pkl".format(MATCHING_METHOD,
                                                                               "LONG_TERM_WITH_NEW_FAIR_EMPTY_VEHICLE_DISPATCH_5",
                                                                               EXPERIMENTAL_MODE, i, vehicle_num,
                                                                               TIME_SLOT, MIN_REQUEST_TIME,
                                                                               MAX_REQUEST_TIME) for i
                        in
                        range(MAX_REPEATS)]
                elif DISPATCH_METHOD == LAF_DISPATCH:
                    _SAVE_RESULT_FILES = [
                        "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD,
                                                                               "LONG_TERM_WITH_LAF_EMPTY_VEHICLE_DISPATCH",
                                                                               EXPERIMENTAL_MODE, i, vehicle_num,
                                                                               TIME_SLOT, MIN_REQUEST_TIME,
                                                                               MAX_REQUEST_TIME) for i
                        in
                        range(MAX_REPEATS)]

                elif DISPATCH_METHOD == TEST_DISPATCH:
                    _SAVE_RESULT_FILES = [
                        "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD,
                                                                               "LONG_TERM_WITH_TEST_EMPTY_VEHICLE_DISPATCH_DispatchState",
                                                                               EXPERIMENTAL_MODE, i, vehicle_num,
                                                                               TIME_SLOT, MIN_REQUEST_TIME,
                                                                               MAX_REQUEST_TIME) for i
                        in
                        range(MAX_REPEATS)]
                elif DISPATCH_METHOD == FAIR_LEVEL_DISPATCH:
                    _SAVE_RESULT_FILES = [
                        "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD,
                                                                               "LONG_3_TERM_WITH_FAIR_BEST_LEVEL_EMPTY_VEHICLE_DISPATCH_DispatchState",
                                                                               EXPERIMENTAL_MODE, i, vehicle_num,
                                                                               TIME_SLOT, MIN_REQUEST_TIME,
                                                                               MAX_REQUEST_TIME) for i
                        in
                        range(MAX_REPEATS)]


            else:
                _SAVE_RESULT_FILES = [
                    "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD, "LONG_TERM",
                                                                           EXPERIMENTAL_MODE, i, vehicle_num,
                                                                           TIME_SLOT, MIN_REQUEST_TIME,
                                                                           MAX_REQUEST_TIME) for i
                    in
                    range(MAX_REPEATS)]
        else:
            _SAVE_RESULT_FILES = [
            "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(MATCHING_METHOD, EXPERIMENTAL_MODE, i, vehicle_num,
                                                               TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in
                range(MAX_REPEATS)]
        if LEARNING:
            _INPUT_ORDERS_DATA_FILES = [
                "../data/input/orders_data/{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, DAY, i, MIN_REQUEST_TIME,
                                                                       MAX_REQUEST_TIME)
                for i in range(MAX_REPEATS)]
            _SAVE_LEARNING_RESULT_FILES = [
                    "../result/learning/{0}_{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(MATCHING_METHOD, DAY, i, vehicle_num, TIME_SLOT,
                                                                            MIN_REQUEST_TIME, MAX_REQUEST_TIME,
                                                                            ) for
                    i in range(MAX_REPEATS)]
            _SAVE_RESULT_FILES = [
                "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.pkl".format(MATCHING_METHOD, "LEARNING",
                                                                       EXPERIMENTAL_MODE, DAY, i, vehicle_num,
                                                                       TIME_SLOT, MIN_REQUEST_TIME,
                                                                       MAX_REQUEST_TIME) for i
                in
                range(MAX_REPEATS)]
        else:
            _INPUT_ORDERS_DATA_FILES = [
                "../data/input/orders_data/{0}_{1}_{2}_{3}.csv".format(EXPERIMENTAL_MODE, i, MIN_REQUEST_TIME, MAX_REQUEST_TIME)
                for i in range(MAX_REPEATS)]


        for epoch in range(MAX_REPEATS):
            print(_INPUT_VEHICLES_DATA_FILES[epoch])
            print(_INPUT_ORDERS_DATA_FILES[epoch])
            print(_SAVE_RESULT_FILES[epoch])
            _simulator.reset()  # 这一步很重要一定要做
            _simulator.load_env(_INPUT_VEHICLES_DATA_FILES[epoch], _INPUT_ORDERS_DATA_FILES[epoch])
            # _simulator.simulate()
            _simulator.fair_simulate()
            # _simulator.laf_simulate()
            _simulator.save_simulate_result(_SAVE_RESULT_FILES[epoch])
            if LEARNING:
                _simulator.save_simulate_learning_result(_SAVE_LEARNING_RESULT_FILES[epoch])
                _simulator.save_simulate_laf_learning_result(_SAVE_LEARNING_RESULT_FILES[epoch])


if __name__ == '__main__':
    simulator = Simulator()  # 这一步已经创建了所有的网络数目，平台情况

    # 直接run 不需要生成车辆以及订单
    # create_orders(simulator)
    # 生成数据, 这个要单独执行, 这句执行过程要把下面的注释掉
    # create_vehicles(simulator)

    # 运行算法， 这个要在上面语句已经成功执行之后执行，这句执行要把上面的注释掉
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("MAX_REPEATS = ",MAX_REPEATS)
    #
    # run_simulation(simulator)
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


