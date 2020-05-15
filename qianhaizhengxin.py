import pandas as pd
import xgboost as xgb

path = 'E:/tianchi/data/qhzx/'
train_B = pd.read_csv(path+'B_train.csv')
test_B = pd.read_csv(path+'B_test.csv')
train_B_info = train_B.describe()
n = len(train_B)
threshod=0.99*len(train_B)
train_B=train_B.drop('UserInfo_170',axis=1)


for col in train_B.columns:
    if train_B[col].isnull().sum()>=threshod or len(train_B[col].dropna().unique())==1:
        train_B=train_B.drop(col,axis=1)




relatioin = train_B.corr()
# print(relatioin)
# relatioin.to_csv(path+'relation.csv')

# del_column = []
# len = relatioin.shape[0]
# save_columns = []
# for i in range(len):
#     if relatioin.columns[i] not in del_column:
#         save_columns.append(relatioin.columns[i])
#         for j in range(i+1,len):
#             if relatioin.iloc[i,j]>0.98 and relatioin.columns[j] not in del_column:
#                 del_column.append(relatioin.columns[j])




train_B_flag = train_B['flag']
train_B_1 = train_B.drop('flag',axis=1)
train_B_2 = train_B_1.drop('no',axis=1)


dtrain_B = xgb.DMatrix(data = train_B_2,label = train_B_flag)
Trate = 0.25
params = {'booster':'gbtree',
          'eta':0.1,
          'max_depth': 4,
          'max_delta_step': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'base_score': Trate,
          'objective': 'binary:logistic',
          'lambda': 5,
          'alpha': 8,
          'random_seed': 100
          }

params['eval_metric']='auc'
xgb_model = xgb.train(params,dtrain_B,num_boost_round=200,maximize=True,verbose_eval=True)

# print(train_B_1.columns)
test_B_1 =xgb.DMatrix(test_B[train_B_2.columns].fillna(-999))
# print(test_B_1.head)

res = xgb_model.predict(test_B_1)
test_B['pred'] = res
test_B[['no','pred']].to_csv(path+'submit.csv',index=None)





