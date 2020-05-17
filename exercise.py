import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBRegressor
# from xgboost.ensemble import GradientBoostingRegressor


path = 'E:/tianchi/data/Weibo Data/'
"""
uid:用户标记
mid：博文标记，抽样&字段加密
time：发博时间，精确到天
forward_count：博文发表一周后的转发数
comment_count：博文发表一周后的评论数
like_count：博文发表一周后的赞数
content: 博文内容
"""
columns = ['uid','mid','time','forward_count','comment_count','like_count','content']

train_data = pd.read_table(path+'weibo_train_data.txt',sep='\t',nrows=1000,header=None)
test_data = pd.read_csv()
train_data.columns=columns

train_data['deliver_date'] = train_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date())
train_data['deliver_hour'] = train_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
train_data['deliver_weekday'] = train_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').weekday())+1

test_data['deliver_hour'] = test_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
test_data['deliver_weekday'] = test_data ['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').weekday())+1


model = LinearRegression(normalize=True)

del_columns = ['uid', 'mid', 'time','content']
train_data_1 = train_data.drop(del_columns,axis=1)





