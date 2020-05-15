import numpy as np
import pandas as pd
import math

a = np.arange(10)
b = a*0.703457 - 1.51632619
d = 1/(1+np.exp(b))
print(d)

x1=np.arange(10).reshape(-1,1)
x0 = np.ones((10,1)).reshape(-1,1)
xdata = pd.DataFrame(columns=['x0'],data=x0)
xdata['x1'] = x1
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])


def sigmoid(x):
    return 1/(1+np.exp(x))

datamatrix = np.mat(xdata)
labelmat = np.mat(y).transpose()
m,n = np.shape(datamatrix)
alpha = 0.01
maxcycle = 100
print()
# weights = np.ones((n,1))
# for k in range(maxcycle):
#     h = sigmoid(datamatrix*weights)
#     error = (labelmat - h)
#     weights = weights - alpha * datamatrix.transpose()*error
#
# print(weights)
#
# y_predict = 1/(np.exp(datamatrix*weights)+1)
# print(y_predict)
# print(1-y_predict)



