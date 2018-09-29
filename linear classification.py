# @Time    : 2018/9/17 15:43
# @Author  : Guo Yuanfang
# @File    : linear classification.py
# @Software: PyCharm


# 本篇中考虑机器学习各种线性分类算法实现

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# ——————————分类问题——————————

## 利用 make_classification函数生成数据集

data = datasets.make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, n_classes=2)
## data[0]: X, data[1] = y
data2fea = datasets.make_classification(n_samples=10, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
                                        n_clusters_per_class=1)


# 感知机

def perceptron(X, y):
    # y = wTx + b
    # y ∈ {1, -1}
    # wTx + b > 0 -> y = 1
    # wTx + b < 0 -> y = -1
    if set(y) == set((0, 1)):
        y = np.sign(y - 0.5)

    num_sample, num_feature = X.shape
    eta = 0.5  # learning rate

    # 初始化 w,b
    w = np.zeros(num_feature)
    b = 0
    # 选取sample（顺序选取/随机选取）
    while True:
        count = 0
        for i in range(num_sample):
            if y[i] * (w.dot(X[i]) + b) <= 0:
                w = w + eta * y[i] * X[i]
                b = b + eta * y[i]
                break
            count += 1
        if count == num_sample:
            break

    return w, b


def logistics(X, y):
    # 随机梯度下降法
    num_sample, num_feature = X.shape
    eta = 0.05  # fixed learning rate
    num_sample, num_feature = X.shape
    one = np.ones((num_sample, 1))
    Xh = np.mat(np.hstack((X, one)))
    yh = np.mat(y).T
    wh = np.mat(np.ones((num_feature + 1, 1)))  # 保存上一个w值
    wn = np.mat(np.zeros((num_feature + 1, 1)))  # 保存最新的w值

    grad = 0
    step = 0
    while np.linalg.norm(wh - wn) > 0.001:
        step += 1
        print("step{}: {}".format(step, wn))

        sam_selected = np.random.randint(0, num_sample)  # 选择某一条sample求梯度

        xs = np.mat(Xh[sam_selected])
        ys = np.mat(y[sam_selected])

        ex = np.exp(wn * xs) / (np.exp(wn * xs) + 1)
        grad = (-ys * xs + xs * ex)
        wh = wn
        wn = wn - eta * grad
    print("\n随机梯度下降算法共迭代{}次得到答案w:\n {}".format(step, wn))
    return


# 调库实现


def test_perceptron():
    # test perceptron

    # XTemp = np.array([[3, 3], [5, 4], [4, 3], [4, -4], [1, 1]])
    # XTempT = XTemp.T
    # yTemp = np.array([1, 1, 1, -1, -1])
    data2fea = datasets.make_classification(n_samples=10, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
                                            n_clusters_per_class=1)
    XTemp = data2fea[0]
    yTemp = data2fea[1]
    if set(yTemp) == set((0, 1)):
        yTemp = np.sign(yTemp - 0.5)
    ind1 = np.argwhere(yTemp == 1)
    ind0 = np.argwhere(yTemp == -1)
    plt.scatter(XTemp[ind1, 0], XTemp[ind1, 1])
    plt.scatter(XTemp[ind0, 0], XTemp[ind0, 1])

    w, b = perceptron(XTemp, yTemp)

    minX = XTemp.min(0)[0]
    maxX = XTemp.max(0)[0]
    plt.plot([minX, maxX], [-(b + w[0] * minX) / w[1], -(b + w[0] * maxX) / w[1]])

    plt.show()
    print('w: {}'.format(w))
    print('b: {}'.format(b))


def test_logistics():
    # test logistics
    # XTemp = np.array([[3, 3], [5, 4], [4, 3], [4, -4], [1, 1]])
    # XTempT = XTemp.T
    # yTemp = np.array([1, 1, 1, -1, -1])
    data2fea = datasets.make_classification(n_samples=10, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
                                            n_clusters_per_class=1)
    XTemp = data2fea[0]
    yTemp = data2fea[1]
    if set(yTemp) == set((0, 1)):
        yTemp = np.sign(yTemp - 0.5)
    ind1 = np.argwhere(yTemp == 1)
    ind0 = np.argwhere(yTemp == -1)
    plt.scatter(XTemp[ind1, 0], XTemp[ind1, 1])
    plt.scatter(XTemp[ind0, 0], XTemp[ind0, 1])

    w, b = logistics(XTemp, yTemp)

    minX = XTemp.min(0)[0]
    maxX = XTemp.max(0)[0]
    plt.plot([minX, maxX], [-(b + w[0] * minX) / w[1], -(b + w[0] * maxX) / w[1]])

    plt.show()
    print('w: {}'.format(w))
    print('b: {}'.format(b))


class multi_classification:
    def __init__(self, model='ovo'):
        self.model = model

    def fit(self, X, y):
        self.num_sample, self.num_feature = X.shape
        set_target = set(y)
        num_target = len(set_target)

        dict_x = {}
        dict_y = {}
        classifiers = []
        for i in set_target:
            dict_x[i] = X[y == i]
            dict_y[i] = y[y == i]
        if self.model == 'ovo':
            # 训练各个单对单分类器
            for i in set_target:
                for j in set_target:
                    if i < j:
                        X_temp = np.vstack((dict_x[i], dict_x[j]))
                        y_temp = np.hstack((dict_y[i], dict_y[j]))

                        classifier = svm.SVC()  # 单个分类器
                        classifier.fit(X_temp, y_temp)
                        classifiers.append(classifier)
        elif self.model == 'ovr':
            for i in range(num_target):
                y_temp = y.copy()
                y_temp = y_temp + 1
                y_temp[y_temp != i+1] = 0

                classifier = svm.SVC()
                classifier.fit(X,y_temp)
                classifiers.append(classifier)
        else:
            print("No such model!")

        self.set_target = set_target
        self.num_target = num_target
        self.classifiers = classifiers

    def get_mode(self,x):
        # 取每一行的众数
        # x须为numpy数组，dtype为int
        m, n = x.shape
        modes = np.zeros(m)
        x = x.astype(int)
        for i in range(len(x)):
            bin_count = np.bincount(x[i])
            if self.model == 'ovr':
                bin_count = bin_count[1:]
            modes[i] = np.argmax(bin_count)
        return modes

    def predict(self, X):

        y_preds = np.zeros((self.num_sample, len(self.classifiers)))
        for i in range(len(self.classifiers)):
            y_preds[:, i] = self.classifiers[i].predict(X)
        y_pred = self.get_mode(y_preds)

        print(y_pred)
        return y_pred


# def multi_cl_ovo(X,y):
#     set_target = set()
#     num_target = len(set_target)
#
#     dict_x = {}
#     dict_y = {}
#     lrs = [[i for i in range(num_target)] for i in range(num_target)] # list of one-vs-one LinearRegression
#
#
#     for i in set_target:
#         dict_x[i] = X[y==i]
#         dict_y[i] = y[y==i]
#
#     # 训练各个单对单分类器
#     for i in set_target:
#         for j in set_target:
#             if i < j:
#                 X_temp = np.vstack((dict_x[i],dict_x[j]))
#                 y_temp = np.vstack((dict_y[i],dict_y[j]))
#                 lr = LinearRegression() # 单个分类器
#                 lr.fit(X_temp,y_temp)
#                 lrs[i,j] = lr
#     return

def multi_cl_ovr():
    return


def test_multi_ovo():
    # 产生数据
    ## sklearn中iris数据集，为一个多（3）目标分类数据集
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    Xtrain, y_train, X_test, y_test = train_test_split(X_iris, y_iris, test_size=0.3)

    movo = multi_classification()
    movo.fit(X_iris, y_iris)
    movo.predict(X_iris)

    return
def test_multi_ovr():
    # 产生数据
    ## sklearn中iris数据集，为一个多（3）目标分类数据集
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    Xtrain, y_train, X_test, y_test = train_test_split(X_iris, y_iris, test_size=0.3)

    movo = multi_classification('ovr')
    movo.fit(X_iris, y_iris)
    movo.predict(X_iris)

    return




if __name__ == '__main__':
    # test_perceptron()
    # test_logistics()
    # test_multi_ovo()
    test_logistics()