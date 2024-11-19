# Long_term_round_based_ridehailing_full
这是一个使用python实现的基于价值函数和动态聚类的公平匹配与司机调度算法。

## 文件说明：
agent文件夹：定义平台、订单以及车辆的信息

algorithm文件夹：实现订单匹配与司机调度算法

data文件夹：存放预处理后的车辆数据以及订单数据，可以通过https://pan.baidu.com/s/1ugobYfSoHZ7pBFtYDFaNUA?pwd=6tfg 链接获取

env文件夹：定义交通路网环境

preprocess文件夹：订单数据的清洗与筛选以及路网节点的聚类，可以通过https://pan.baidu.com/s/1ugobYfSoHZ7pBFtYDFaNUA?pwd=6tfg 链接获取

result文件夹：.pkl文件存放最终结果

runner文件夹：加载环境以及数值实验

## 实验复现：
运行runner文件夹中的main.py文件

python main.py
