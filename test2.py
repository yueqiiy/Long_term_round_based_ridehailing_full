import os
import random
from collections import defaultdict

import numpy as np
import gym
import xs_gym

# env = gym.make("myEnv-v0")
# obs = env.reset()
# print(env.action_space.sample)
# while True:
#     _,_,done,_ = env.step(2)
#     if done:
#         break
# print(obs)
# v_nums = np.array([0.25, 0.25, 0.25, 0.25])
#
# for i in range(10):
#     print(np.random.choice([1500, 2000, 2500, 3000], p=v_nums.ravel()))

#
# import random
# import pulp
#
# # Number of drivers and orders
# num_drivers = 50
# num_orders = 100
#
# # Randomly generate drivers data
# drivers = []
# for i in range(num_drivers):
#     drivers.append((random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
#
# # Randomly generate orders data
# orders = []
# for i in range(num_orders):
#     orders.append((random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
#
# # Define the LP problem
# prob = pulp.LpProblem("Taxi Dispatch", pulp.LpMaximize)
#
# # Define variables
# x = pulp.LpVariable.dicts("x", [(i, j) for i in range(num_drivers) for j in range(num_orders)],
#                         lowBound=0,
#                         upBound=1,
#                         cat=pulp.LpInteger)
#
# # Define the objective
# prob += pulp.lpSum([(drivers[i][2] + orders[j][2] - np.sqrt((drivers[i][0] - orders[j][0])**2 + (drivers[i][1] - orders[j][1])**2)) * x[(i, j)] for i in range(num_drivers) for j in range(num_orders)])
#
# # Add constraints
# for i in range(num_drivers):
#     prob += pulp.lpSum([x[(i, j)] for j in range(num_orders)]) <= 1
#
# for j in range(num_orders):
#     prob += pulp.lpSum([x[(i, j)] for i in range(num_drivers)]) <= 1
#
# # Solve the LP problem
# prob.solve()
#
# # Print the result
# print("Optimal Value:", pulp.value(prob.objective))
# print("\nMatched Drivers and Orders:")
# drivers1 = []
# for i in drivers:
#     drivers1.append(i)
# for v in prob.variables():
#     if v.varValue == 1:
#         temp = v.name.replace("(","").replace(",","").replace(")","").split("_")
#         driver_id = int(temp[1])
#         order_id = int(temp[2])
#         drivers1[driver_id] += orders[order_id]
#         print("Driver {0} matches Order {1}".format(driver_id, order_id))

import random
# import numpy as np
from scipy.spatial.distance import cdist
from pulp import *


def generate_data(n_drivers, n_orders, dim):
    drivers = np.random.rand(n_drivers, dim)
    orders = np.random.rand(n_orders, dim)
    rewards = np.random.rand(n_orders)
    return drivers, orders, rewards


def allocate_orders(drivers, orders, rewards, cost_coef=1):
    n_drivers, dim = drivers.shape
    n_orders = len(rewards)

    distances = cdist(drivers, orders, metric='euclidean')
    driver_income = [LpVariable("driver_{}".format(i), 0, None, LpContinuous) for i in range(n_drivers)]
    model = LpProblem("Maximize driver's income while minimizing variance", LpMaximize)

    # Constraint 1: each order can only be matched to one driver
    for j in range(n_orders):
        model += lpSum([LpVariable("x_{}_{}".format(i, j), 0, 1, LpBinary) for i in range(n_drivers)]) <= 1

    # Constraint 2: each driver can only match one order
    for i in range(n_drivers):
        model += lpSum([LpVariable("x_{}_{}".format(i, j), 0, 1, LpBinary) for j in range(n_orders)]) <= 1

    # Objective: Minimize variance of driver's income
    mean = lpSum(driver_income) / n_drivers
    variance = lpSum([(income - mean) ** 2 for income in driver_income])
    model += variance

    # Driver's income
    for i in range(n_drivers):
        for j in range(n_orders):
            model += driver_income[i] >= rewards[j] - cost_coef * distances[i][j] * LpVariable("x_{}_{}".format(i, j),
                                                                                               0, 1, LpBinary)

    # Solve the linear programming problem
    status = model.solve()
    if status != LpStatusOptimal:
        return None

    # Get the allocation result
    allocation = []
    for i in range(n_drivers):
        for j in range(n_orders):
            if value(LpVariable("x_{}_{}".format(i, j))) == 1:
                allocation.append((i, j))
    return allocation


# Generate data
n_drivers = 20
n_orders = 50
dim = 2
drivers, orders, rewards = generate_data(n_drivers, n_orders, dim)

# Allocate orders to drivers
allocation = allocate_orders(drivers,orders,rewards)
print(allocation)