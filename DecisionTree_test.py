import numpy as np
import pandas as pd
#https://blog.csdn.net/z962013489/article/details/80024574


def info_entry(p):
    return -1*sum(p*np.log10(p))

#计算信息增益
def gain_entry(feat_data):
    class_gain = info_entry(list(feat_data.iloc[:,-1].value_counts(normalize=True)))
    #ent_dict存放每个特征的概率字典
    ent_dict = {}
    for col in feat_data.columns:
        #ent_dict[col]存放每个特征里面每个取值的概率
        ent_dict[col]={}
        ent_dict[col]=info_entry(list(feat_data[col].value_counts(normalize=True)))
        class_gain-=ent_dict[col]
    return class_gain

#计算信息增益
def gain_entry(feat_data):
    y=feat_data.iloc[:, -1]
    category= list(set(feat_data.iloc[:, -1]))
    # for label in category:
    #     p = len([label_ for label_ in category if label_==label])/len(y)
    #ent_dict存放每个特征的概率字典
    # ent_dict = {}
    for lab in category:
        for col in feat_data.columns:
            # ent_dict[col]={}
            # ent_dict[col]存放每个特征里面每个取值的概率
            c_lab = list(set(feat_data[col]))
            ent_dict = len(feat_data[col]==c_lab&feat_data.iloc[:, -1]==lab)/len(feat_data[col]==c_lab)



    return class_gain



path = 'E:/tianchi/data/decisiontreedata/'
file_name = 'decisitontree.csv'
feat_data = pd.read_csv(path+file_name,encoding='gbk')
y = feat_data.iloc[:, -1]
category = list(set(feat_data.iloc[:, -1]))
# for label in category:
#     p = len([label_ for label_ in category if label_==label])/len(y)
# ent_dict存放每个特征的概率字典
# ent_dict = {}
for lab in category:
    for col in feat_data.columns:
        # ent_dict[col]={}
        # ent_dict[col]存放每个特征里面每个取值的概率
        c_lab = list(set(feat_data[col]))
        ent_dict = len(feat_data[col] == c_lab & feat_data.iloc[:, -1] == lab) / len(feat_data[col] == c_lab)