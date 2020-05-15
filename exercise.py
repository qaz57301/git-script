import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression


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
train_data.columns=columns

def is_same(m,n):
    if m==n:
        return 1
    else:
        return 0


train_data['deliver_date'] = train_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date())
train_data['deliver_hour'] = train_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
train_data['deliver_weekday'] = train_data['time'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').weekday())+1
uid_isforward = list(train_data[train_data['forward_count']>0]['uid'].unique())
train_data['is_forward']=train_data.apply(lambda x: x['uid'] in uid_isforward,axis=1)
print(train_data['is_forward'])
# train_data[train_data['forward_count']>0].to_csv(path+'train_data.csv',encoding='UTF_8_sig')
# print(train_data[train_data['forward_count']==0].groupby('deliver_weekday')['forward_count'].describe())
# print(train_data[train_data['forward_count']>0].groupby('deliver_weekday')['forward_count'].describe().unstack())
# print(train_data[train_data['forward_count']==0].groupby('deliver_hour')['forward_count'].describe())
# print(train_data[train_data['forward_count']>0].groupby('deliver_hour')['forward_count'].describe())

# model = LinearRegression(normalize=True)





