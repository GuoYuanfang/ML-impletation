# @Time    : 2018/9/17 15:43
# @Author  : Guo Yuanfang
# @File    : Linear.py
# @Software: PyCharm


# 本篇中考虑线性机器学习各种算法

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# ——————————回归问题——————————
# 产生数据
# 最小二乘法


# 解析解

# 梯度下降法

# 随机梯度下降法


# ——————————分类问题——————————
# 产生数据

## sklearn中iris数据集，为一个多（3）目标分类数据集
iris = datasets.load_iris()

## 利用 make_classification函数生成数据集
## make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
## n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
## flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
## shuffle=True, random_state=None)


data = datasets.make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, n_classes=2)
## data[0]: X, data[1] = y
data2fea = datasets.make_classification(n_samples=10, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
                                        n_clusters_per_class=1)

## 二分类
### 将iris数据筛选为二分类问题

X_iris = iris.data[iris.target <= 1]
y_iris = iris.target[iris.target <= 1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=0)


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
    num_sample, num_feature = X.shape

if __name__ == '__main__':
    #     # test perceptron
    #
    #     # XTemp = np.array([[3, 3], [5, 4], [4, 3], [4, -4], [1, 1]])
    #     # XTempT = XTemp.T
    #     # yTemp = np.array([1, 1, 1, -1, -1])
    #     data2fea = datasets.make_classification(n_samples=10, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
    #                                             n_clusters_per_class=1)
    #     XTemp = data2fea[0]
    #     yTemp = data2fea[1]
    #     if set(yTemp) == set((0, 1)):
    #         yTemp = np.sign(yTemp - 0.5)
    #     ind1 = np.argwhere(yTemp == 1)
    #     ind0 = np.argwhere(yTemp == -1)
    #     plt.scatter(XTemp[ind1, 0], XTemp[ind1, 1])
    #     plt.scatter(XTemp[ind0, 0], XTemp[ind0, 1])
    #
    #     w, b = perceptron(XTemp, yTemp)
    #
    #     minX = XTemp.min(0)[0]
    #     maxX = XTemp.max(0)[0]
    #     plt.plot([minX, maxX], [-(b + w[0] * minX) / w[1], -(b + w[0] * maxX) / w[1]])
    #
    #     plt.show()
    #     print('w: {}'.format(w))
    #     print('b: {}'.format(b))
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

# 逻辑回归

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
