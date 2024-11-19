# Long_term_round_based_ridehailing
# Author: Jackye
# Time : 2020/7/8 7:49 下午
import pickle
import time
# test_str = "4,12,13,24,41,42,43,45,48,50,68,74,75,79,87,88,90,100,103,104,105," \
#            "107,113,114,116,120,125,127,128,137,140,141,142,143,144,148,151,152,153,158,161,162,163,164" \
#            ",166,170,186,194,202,209,211,224,229,230,231,232,233,234,236,237,238,239,243,244,246,249,261,262,263"
# print(', '.join(test_str.split(',')))
# test
# test_str

# t1 = time.time()
# for i in range(1000):
#     print(i)
#     time.sleep(0.1)
# t2 = time.time()
# print("运行时间")
# print(t2 - t1)

# tem = [i for i in range(1,2)]
# print(tem)
import pulp as pl
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']
costs = {'CHICKEN': 0.013,
         'BEEF': 0.008,
         'MUTTON': 0.010,
         'RICE': 0.002,
         'WHEAT': 0.005,
         'GEL': 0.001}

proteinPercent = {'CHICKEN': 0.100,
                  'BEEF': 0.200,
                  'MUTTON': 0.150,
                  'RICE': 0.000,
                  'WHEAT': 0.040,
                  'GEL': 0.000}

fatPercent = {'CHICKEN': 0.080,
              'BEEF': 0.100,
              'MUTTON': 0.110,
              'RICE': 0.010,
              'WHEAT': 0.010,
              'GEL': 0.000}

fibrePercent = {'CHICKEN': 0.001,
                'BEEF': 0.005,
                'MUTTON': 0.003,
                'RICE': 0.100,
                'WHEAT': 0.150,
                'GEL': 0.000}

saltPercent = {'CHICKEN': 0.002,
               'BEEF': 0.005,
               'MUTTON': 0.007,
               'RICE': 0.002,
               'WHEAT': 0.008,
               'GEL': 0.000}

#创建问题实例，求最小极值
prob = LpProblem("The Whiskas Problem", LpMinimize)

#构建Lp变量字典，变量名以Ingr开头，如Ingr_CHICKEN，下界是0
ingredient_vars = LpVariable.dicts("Ingr",Ingredients,0)

#添加目标方程
prob += lpSum([costs[i]*ingredient_vars[i] for i in Ingredients])

#添加约束条件
prob += lpSum([ingredient_vars[i] for i in Ingredients]) == 100
prob += lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0
prob += lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0
prob += lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0
prob += lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4

#求解
prob.solve()
#查看解的状态
print("Status:", LpStatus[prob.status])
#查看解
for v in prob.variables():
    print(v.name, "=", v.varValue)
#另外一种查看解的方式
# for i in Ingredients:
#   print(ingredient_vars[i],"=",ingredient_vars[i].value())




