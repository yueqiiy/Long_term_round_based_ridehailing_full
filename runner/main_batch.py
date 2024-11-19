# learning 使用的main文件
# batch主要是用来一次性运行多天的数据 便于得到多天的车辆状态信息变化
import sys
sys.path.append("/data/yueqi/Long_term_round_based_ridehailing_full")
from setting import *
from utility import Day
from runner.simulator import Simulator

def create_orders(_simulator: Simulator):
    print("create order data")
    if LEARNING:
        for day in range(1, 31):
            if day in WEEKEND:
                continue
            d = Day()
            d.day = day
            print("main: ", day)
            # print("day: ", day)
            _INPUT_LEARNING_ORDERS_DATA_FILES = [
                "../data/input/orders_data/LEARNING_{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, day, i, MIN_REQUEST_TIME,
                                                                           MAX_REQUEST_TIME) for i in
                range(MAX_REPEATS)]

            for epoch in range(MAX_REPEATS):
                _simulator.create_order_env(_INPUT_LEARNING_ORDERS_DATA_FILES[epoch])
    else:
        for epoch in range(MAX_REPEATS):
            _simulator.create_order_env(INPUT_ORDERS_DATA_FILES[epoch])


def create_vehicles(_simulator: Simulator):
    for epoch in range(MAX_REPEATS):
        _simulator.create_vehicle_env(INPUT_VEHICLES_DATA_FILES[epoch])


def run_simulation(_simulator: Simulator):
    for day in range(4, 31):
        if day in WEEKEND:
            continue
        print("day: ", day)
        if LONG_TERM:
            print("long term optimization")
        else:
            print("short term optimization")
        if LEARNING:
            print("in learning process")
        else:
            print("in planning process")
        if EMPTY_VEHICLE_DISPATCH:
            print("activate empty vehicle dispatch: "+DISPATCH_METHOD)
        print("current algorithm: ", MATCHING_METHOD)
        print("current time: ", MIN_REQUEST_TIME/3600, "-", MAX_REQUEST_TIME/3600)
        print("time slot: ", TIME_SLOT)
        for vehicle_num in range(2500, 2501, 500):
            print("vehicle num: ", vehicle_num)
            _INPUT_VEHICLES_DATA_FILES = [
                "../data/input/vehicles_data/LEARNING_{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, i, vehicle_num,
                                                                         MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in
                range(MAX_REPEATS)]
            # if LONG_TERM:
            #     if EMPTY_VEHICLE_DISPATCH:
            #         _SAVE_RESULT_FILES = [
            #             "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD,
            #                                                                    "LONG_TERM_WITH_EMPTY_VEHICLE_DISPATCH",
            #                                                                    EXPERIMENTAL_MODE, i, vehicle_num,
            #                                                                    TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i
            #             in
            #             range(MAX_REPEATS)]
            #     else:
            #         _SAVE_RESULT_FILES = [
            #             "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format(MATCHING_METHOD, "LONG_TERM",
            #                                                                    EXPERIMENTAL_MODE, i, vehicle_num,
            #                                                                    TIME_SLOT, MIN_REQUEST_TIME,
            #                                                                    MAX_REQUEST_TIME) for i
            #             in
            #             range(MAX_REPEATS)]
            # else:
            #     _SAVE_RESULT_FILES = [
            #     "../result/{0}/{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(MATCHING_METHOD, EXPERIMENTAL_MODE, i, vehicle_num,
            #                                                        TIME_SLOT, MIN_REQUEST_TIME, MAX_REQUEST_TIME) for i in
            #         range(MAX_REPEATS)]

            _INPUT_ORDERS_DATA_FILES = [
                "../data/input/orders_data/LEARNING_{0}_{1}_{2}_{3}_{4}.csv".format(EXPERIMENTAL_MODE, day, i, MIN_REQUEST_TIME,
                                                                           MAX_REQUEST_TIME)
                for i in range(MAX_REPEATS)]
            if TYPED:
                _SAVE_LEARNING_RESULT_FILES = [
                            "../result/learning_4CLASS/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format("TYPED_LEVEL", MATCHING_METHOD, day, i, vehicle_num, TIME_SLOT,
                                                                                    MIN_REQUEST_TIME, MAX_REQUEST_TIME,
                                                                                    ) for
                            i in range(MAX_REPEATS)]

            _SAVE_RESULT_FILES = [
                    "../result/learning_fair_greedy/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.pkl".format("LEARNING_"+MATCHING_METHOD,
                                                                           EXPERIMENTAL_MODE, day, i, vehicle_num,
                                                                           TIME_SLOT, MIN_REQUEST_TIME,
                                                                           MAX_REQUEST_TIME) for i
                    in
                    range(MAX_REPEATS)]

            print("result_dir: ", _SAVE_RESULT_FILES)  # 创建pkl 文件
            # print("learning_result_dir", _SAVE_LEARNING_RESULT_FILES)  # 保存历史状态转移
            _SAVE_STATES_VALUE_RESULT_FILES = "../result/learning_random_class_vehiclestate/TYPED_LEVEL_States_Values_Dispatch.pkl"
            print("_SAVE_STATES_VALUE_RESULT_FILES = ", _SAVE_STATES_VALUE_RESULT_FILES)
            for epoch in range(MAX_REPEATS):
                _simulator.reset()  # 这一步很重要一定要做
                _simulator.load_env(_INPUT_VEHICLES_DATA_FILES[epoch], _INPUT_ORDERS_DATA_FILES[epoch])
                _simulator.simulate_learning()
                # _simulator.save_simulate_result(_SAVE_RESULT_FILES[epoch])
                if LEARNING:
                    # _simulator.save_simulate_learning_result(_SAVE_LEARNING_RESULT_FILES[epoch])
                    _simulator.save_simulate_VS_learning_result(_SAVE_STATES_VALUE_RESULT_FILES)


if __name__ == '__main__':
    simulator = Simulator()  # 这一步已经创建了所有的网络数目，平台情况

    # create_orders(simulator)
    # 生成数据, 这个要单独执行, 这句执行过程要把下面的注释掉
    # create_vehicles(simulator)

    # 运行算法， 这个要在上面语句已经成功执行之后执行，这句执行要把上面的注释掉
    run_simulation(simulator)
