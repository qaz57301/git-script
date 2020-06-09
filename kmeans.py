import pandas as pd
import numpy as np

path = "E:/cv/data/binary/"
file = "train_set.csv"
data1 = pd.read_csv(path+file)
x = np.array(data1)[:,(1,6)]

points = x
#数据points的行数，即列数
dimensions = len(points[0])   #列数
new_center = []
# print(points[0])
# print(points[1])
for dimension in range(dimensions):
    dim_sum = 0
    for p in points:
        dim_sum+=p[dimension]
    new_center.append(dim_sum/float(len(points)))
    print(new_center)
