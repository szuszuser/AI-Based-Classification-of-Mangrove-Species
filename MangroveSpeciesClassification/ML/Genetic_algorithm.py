import copy
import random
import pandas as pd
from Life import Life
import numpy as np


# 定义遗传算法类
class GA(object):
    """遗传算法类"""
    # 初始化变量
    def __init__(self, cross_rate, mutation_rate, population_size, gene_length, fitness_function):
        self.crossRate = cross_rate  # 交叉概率
        self.mutationRate = mutation_rate  # 变异概率
        self.populationSize = population_size  # 种群容量/个体数
        self.geneLength = gene_length  # 染色体长度/特征个数
        self.fitnessFunction = fitness_function  # 适应度函数
        self.lives = []  # 种群
        self.best = None  # 保存这一代中最好的个体
        self.gene = np.random.randint(0, 2, self.geneLength)  #
        self.score = -1  # 保存全局最高的适应度
        self.generation = 0  # 第几代
        self.crossCount = 0  # 交叉数量
        self.mutationCount = 0  # 变异个数
        self.bounds = 0.0  # 适配度总和，用于计算选择概率
        self.initPopulation()  # 初始化种群

    def initPopulation(self):
        """初始化种群"""
        self.lives = []
        for i in range(self.populationSize):
            gene = np.random.randint(0, 2, self.geneLength)
            random.shuffle(gene)   # 重新排序
            life = Life(gene)
            self.lives.append(life)

    def judge(self):
        """计算适配度进行评估"""
        self.bounds = 0.0  # 适配度之和，用于选择时计算概率
        self.best = self.lives[0]  # 假设种群中的第一个染色体被选中

        for life in self.lives:
            life.score = self.fitnessFunction(life)
            self.bounds += life.score
            if self.best.score < life.score:
                self.best = life

        if self.score < self.best.score:
            self.score = copy.deepcopy(self.best.score)
            self.gene = copy.deepcopy(self.best.gene)

        self.best.score = copy.deepcopy(self.score)
        self.best.gene = copy.deepcopy(self.gene)

    def corss(self, parent1, parent2):
        """
        交叉函数
        随机交叉长度为n的片段，n随机产生
        """
        index1 = random.randint(0, self.geneLength - 1)  # 随机生成交叉初始位置
        index2 = random.randint(index1, self.geneLength - 1)  # 随机生成交叉终止位置

        for index in range(len(parent1.gene)):
            if (index >= index1) and (index <= index2):
                parent1.gene[index], parent2.gene[index] = parent2.gene[index], parent1.gene[index]

        self.crossCount += 1
        return parent1.gene

    def mutation(self, chromosome):
        """
        突变函数
        """
        newChromosome = chromosome[:]  # 产生一个新的基因序列/染色体，避免变异的时候影像父种群
        # 随机选择两个位置的基因进行交换——变异
        index1 = random.randint(0, self.geneLength - 1)
        index2 = random.randint(0, self.geneLength - 1)
        newChromosome[index1], newChromosome[index2] = newChromosome[index2], newChromosome[index1]
        self.mutationCount += 1
        return newChromosome

    def getOne(self):
        """
        选择个体函数
        轮盘赌选择子代个体
        """
        r = random.uniform(0, self.bounds)   # 从均匀分布中采样
        for life in self.lives:
            r -= life.score
            if r <= 0:
                return life

        raise Exception("选择错误", self.bounds)

    def newChild(self):
        """
        产生新的后代
        """
        parent1 = self.getOne()
        # 根据交叉概率进行交叉
        rate = random.random()  #  交叉概率：在[0,1)中随机生成一个浮点数
        if rate < self.crossRate:
            parent2 = self.getOne()
            gene = self.corss(parent1, parent2)
        else:
            gene = parent1.gene

        # 根据变异概率进行变异
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)

        return Life(gene)

    def next(self):
        """
        产生下一代
        """
        # 评估当前一代个体的适应度
        self.judge()
        # 产生下一代
        newLives = []
        newLives.append(self.best)  # 把适应度最好的个体加入到下一代
        while len(newLives) < self.populationSize:
            newLives.append(self.newChild())
        self.population = newLives
        self.generation += 1








