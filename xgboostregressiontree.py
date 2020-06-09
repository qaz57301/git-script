import pandas as pd


def calculationDiffCount(datas):
    # 将输入的数据汇总（input dataset）
    results = {}
    results = dict(datas.iloc[:, -1].value_counts())
    return results


def gini(rows):
    #计算gini值（calculate GINI）
    length = len(rows)
    results = calculationDiffCount(rows)
    imp = 0.0
    for i in results:
        imp+=(results[i]/length)**2
    return 1-imp



path = 'E:/tianchi/data/decisiontreedata/'
datas = pd.read_csv(path+'buycomputer.csv',encoding='gbk')
# datas = open(path+'buycomputer.csv','r')
results = {}
results = dict(datas.iloc[:,-1].value_counts())
# print(label_dict)
imp = 0.0
for i in results:
    











