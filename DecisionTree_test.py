import numpy as np
import pandas as pd
#https://blog.csdn.net/z962013489/article/details/80024574


def info_entry(p):
    return -1*sum(p*np.log2(p))

#计算信息增益
def gain_entry(feat_data):
    class_gain = info_entry(list(feat_data.iloc[:,-1].value_counts(normalize=True)))
    columns = feat_data.columns
    labels = columns[-1]              #datafrmae最后一列
    category = list(set(feat_data[labels]))       ##将最后一列去重获得类别
    col_labs = columns[0:-1]          #datafrmae去掉最后一列
    ent_dict = {}    #存放各特征信息增益值
    m = feat_data.shape[0]
    #ent_dict存放每个特征的信息增益
    ent_dict = {}
    for col in col_labs:
        #每一列的信息增益
        ent_dict[col] = {}
        p_lab = []
        for i in list(set(feat_data[col])):
            #每个特征中的每个取值的长度
            n = len(feat_data[feat_data[col] == i])
            for lab in category:
                #每个特征每个取值对应不同分类的概率
                p = len(feat_data[(feat_data[col] == i) & (feat_data[labels] == lab)]) / len(
                    feat_data[feat_data[col] == i])
                if p > 0:
                    p1 = -1 * (n / m) * p * np.log2(p)
                else:
                    p1 = 0
                p_lab.append(p1)
        ent_dict[col] = class_gain - sum(p_lab)
    node = max(ent_dict, key=ent_dict.get)
    return node

