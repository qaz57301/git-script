"""
https://blog.csdn.net/tian_tian_hero/article/details/89409472
https://blog.csdn.net/weixin_43408110/article/details/87827432
事件的几率（odds）,是指该事件发生的概率与不发生的概率的比值。如果事件发生的概率为p，那么不发生的概率是1-p，那么发生的几率是p/(1-p),
该事件发生的对数几率或logit函数为 log(p/(1-p))
通过对数，可以将输出转换到整个实数范围内，
log(p/(1-p)) = w.T*x
"""
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
import pandas as pd
from numpy import *
from matplotlib import pyplot as plt
path = "E:/cv/data/"

# def sigmoid(x):
#     return 1.0/(1+exp(-x))

irdata = pd.read_csv(path + 'iris_1.csv')
#先将数据划分为训练和测试样本
data_train,data_test = train_test_split(irdata,test_size = 0.2,random_state = 1234)
col_names = irdata.columns
x = data_train[col_names[:-1]]   #自变量
y = data_train[col_names[-1]]    #因变量

#逻辑回归
model = LogisticRegression()
model.fit(x,y)
print(model.fit(x,y))

coef = model.coef_   #回归系数
print(coef)
print("==================")
print(x.columns.tolist)
print("++++++++++++++++++")
# print(model)
print(model.intercept_)
# print("intercept=%s"%model.intercept_[0])
# coef_regression = pd.Series(index=['Intercept']+x.columns.tolist(),data=model.intercept_[0]+coef.tolist()[0])
