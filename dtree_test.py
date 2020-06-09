from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys
import pandas as pd
from utils.data_manipulation import divide_on_feature
from utils.data_operation import calculate_entropy


def calculate_entropy(y):
    # 计算log2(x)
    log2=lambda x:math.log(x)/math.log(2)
    lable_value = np.unique(y)
    entropy = 0
    for label in lable_value:
        count = len(y[y==label])
        p = count/len(y)
        entropy += -p*log2(p)
    return entropy

def calculate_information_gain(y,y1,y2):
    



path = "E:/cv/data/binary/"
file = "train_set.csv"
data1 = pd.read_csv(path+file)
x = np.array(data1)[:,1:-1]
y = np.array(data1)[:,-1]
# print(x)
# print("==========")
# print(y)
n_samples,n_features = np.shape(x)
# 不确定性最大值
largest_impurity = 0
#Feature index and threshold
best_criteria = None
best_sets = None


#Check if expansion of y is needed;
if len(np.shape(y))==1:
    y=np.expand_dims(y,axis=1)

xy = np.concatenate((x,y),axis=1)

min_samples_split=2
max_depth=float("inf")
current_depth=0



feature_i = 0
feature_values = np.expand_dims(x[:, feature_i], axis=1)
unique_values = np.unique(feature_values)

if n_samples>=min_samples_split and current_depth<=max_depth:
    feature_values = np.expand_dims(x[:,feature_i],axis=1)
    unique_values = np.unique(feature_values)
    unique_values = [48]
    #Iterate through all unique values of feature columns i and calculate the impurity
    for threshold in unique_values:
        xy1,xy2 = divide_on_feature(xy,feature_i,threshold)
        if len(xy1)>0 and len(xy2)>0:
            y1 = xy1[:,n_features:]
            y2 = xy2[:,n_features:]

            #Caculate impurity
            impurity = _impurity_calculation(y, y1, y2)















