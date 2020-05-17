import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


x = np.arange(10).reshape(-1,1)
x = sm.add_constant(x)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])

##逻辑函数
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

datamatIn = x
classlabels = y
##梯度上升函数
dataMatrix = np.mat(datamatIn)
labelMat = np.mat(classlabels).transpose()
#求矩阵的行数和列数
m,n = np.shape(dataMatrix)
##步长，可以自己设置
alpha = 0.001
#最大循环次数
maxtry = 500
##初始化权重行向量
weights = np.ones((n,1))
#循环迭代求权重
for k in range(maxtry):
    ##求出h(theta)
    h = sigmoid(dataMatrix*weights)
    ##向量的偏差
    error = (labelMat - h)
    ##weights是n行1列的向量
    weights = weights + alpha*dataMatrix.transpose()*error


h = sigmoid(dataMatrix*weights)
print(h)
print(weights)

xx = np.arange(10).reshape(-1,1)
model = LogisticRegression()
model.fit(xx,y)
y1=model.predict(xx)
coef = model.coef_
intercept = model.intercept_
# print(coef,intercept)
# print(model.get_params())
# print(y1)

# help(LogisticRegression)




