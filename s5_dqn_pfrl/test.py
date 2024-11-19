import os
import logging.handlers

import matplotlib.pyplot as plt
import numpy as np
a = []
# fp = open("out_2500_09_1000_10per.log","rb")
a.append(103)
step = 96

fp = open("out_2500_09_1000_10per.log","rb")
# a.append(105)
# step = 98
cnt = 0

for i in range(999):
    a.append(a[i]+step)
p = 0
res = []
for line in fp.readlines():
    cnt += 1
    if cnt == a[p]:
        p += 1
        temp = str(line).split(":")[1].split("=")[1].replace("*","")[:-3]
        res.append(float(temp))

print(res)
x= np.arange(0,len(res))
plt.title("FAIR")
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, res)
plt.show()

