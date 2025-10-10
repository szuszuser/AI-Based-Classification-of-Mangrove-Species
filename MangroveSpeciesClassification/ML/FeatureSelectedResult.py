import copy
import pandas as pd
import numpy as np
import gdal
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import openpyxl as op

columns = []  # 原始列名称
train_data = pd.read_excel(r"train_data.xlsx", sheet_name='01_spectral_feature', usecols=columns)
test_data = pd.read_excel(r"test_data.xlsx", sheet_name='01_spectral_feature', usecols=columns)

features_name = []  # 筛选后的特征

xtrain = train_data[features_name]
ytrain = train_data.iloc[:, 0]  # (490,)  训练数据：标签
xtest = test_data[features_name]
ytest = test_data.iloc[:, 0]  # (330,)  测试数据：标签

# 建立评估模型

result_list = []
estimator = RandomForestClassifier()
estimator.fit(xtrain, ytrain)
y_pre = estimator.predict(xtest)
overall_accuracy = estimator.score(xtest, ytest)  # 计算总体精度
kappa = cohen_kappa_score(ytest, y_pre)   # 计算kappa系数
cm = confusion_matrix(list(ytest), y_pre)   # 计算混淆矩阵
feature_importacne = estimator.feature_importances_
print(overall_accuracy)
result_list.append(overall_accuracy)
print(kappa)
result_list.append(kappa)
print(cm)
result_list.append(cm)

pd.DataFrame(result_list).to_excel(r"") # 输出结果
file = open(r'', "wb") # 模型
# 将模型写入文件：
pickle.dump(estimator, file)
# 最后关闭文件：
file.close()
