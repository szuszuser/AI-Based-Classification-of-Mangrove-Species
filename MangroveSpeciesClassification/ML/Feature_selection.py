# -*- encoding: utf-8 -*-

import random
import math
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from Genetic_algorithm import GA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class Featureselection(object):
    def __init__(self, population_size=50):
        self.columns = [] # 输入数据列名称
        self.train_data = pd.read_excel(r"train_data.xlsx", sheet_name='01_spectral_feature', usecols=self.columns)
        self.test_data = pd.read_excel(r"data.xlsx", sheet_name='01_spectral_feature', usecols=self.columns)
        self.populationSize = population_size
        self.ga = GA(cross_rate=0.7,
                     mutation_rate=0.1,
                     population_size=self.populationSize,
                     gene_length=len(self.columns) - 1,
                     fitness_function=self.fitnessFunction()
                     )

    def accuracy_score(self, order):
        """
        适应度函数
        :param order: 输入的基因/染色体
        :return: score = accuracy_score精度
        """
        features = self.columns[1:]  # 获取特征的名称
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])
        # 划分特征和标签
        self.xtrain = self.train_data[features_name]  # (490,44)  训练数据：特征
        self.ytrain = self.train_data.iloc[:, 0]  # (490,)  训练数据：标签
        self.xtest = self.test_data[features_name]  # (330,44)  测试数据：特征
        self.ytest = self.test_data.iloc[:, 0]  # (330,)  测试数据：标签
        # 建立评估模型
        estimator = RandomForestClassifier()  # 利用RandomForest分类算法作为评估模型
        estimator.fit(self.xtrain, self.ytrain)  # 训练模型
        self.ypredict = estimator.predict(self.xtest)  # 预测值
        score = estimator.score(self.xtest, self.ytest)  # 分类精度 = 适应度
        kappa = cohen_kappa_score(self.ytest, self.ypredict)  # Kappa系数
        cm = confusion_matrix(list(self.ytest), self.ypredict)  # 混淆矩阵
        print(features_name)
        print("分类精度：", score)
        return score

    def fitnessFunction(self):
        return lambda life: self.accuracy_score(life.gene)

    def run(self, n=0):
        distance_list = []
        generate = [index for index in range(1, n+1)]  # 列表推导式：循环后加入列表
        while n > 0:
            self.ga.next()
            # distance = self.accuracy_score(self.ga.best.gene)
            distance = self.ga.score
            distance_list.append(distance)
            print(("第%d代 : 当前最好特征组合的线下验证结果为：%f") % (self.ga.generation, distance))
            n -= 1

        print('当前最好特征组合：')
        string = []
        flag = 0
        features = self.columns[1:]
        for index in self.ga.gene:
            if index == 1:
                string.append(features[flag])
            flag += 1
        print(string)
        print('最高分类精度为', self.ga.score)


def main():
    fs = Featureselection(population_size=50)
    rounds = 100
    fs.run(rounds)


if __name__ == '__main__':
    main()

