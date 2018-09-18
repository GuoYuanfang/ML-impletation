# @Time    : 2018/9/18 10:19
# @Author  : Guo Yuanfang
# @File    : linear regression.py
# @Software: PyCharm


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ——————————回归问题——————————
# 产生数据
dataR = make_regression(n_features=1, n_informative=1, noise=10)  # data-regression


# 最小二乘法
## 解析解
def closed_formed(X, y):
    num_sample, num_feature = X.shape
    one = np.ones((num_sample, 1))
    Xh = np.mat(np.hstack((X, one)))
    yh = np.mat(y).T
    try:
        wh = (Xh.T * Xh).I * Xh.T * yh
    except:
        print("Matrix not invertable!")

    print(wh)

    ypred = wh.T * Xh.T

    if X.shape[1] == 1:
        plt.scatter(X, y)
        plt.plot(X, ypred.T, 'r')
        plt.show()

    return wh


# 梯度下降法
def gradient_decent(X, y):
    return


# 随机梯度下降法

# 调库实现
def sk_linear_regression(X, y):
    clf = LinearRegression()
    clf.fit(X, y)
    ypred = clf.predict(X)
    print(clf.score(X, y))
    if X.shape[1] == 1:
        plt.scatter(X, y)
        plt.plot(X, ypred, 'r')
        plt.show()
    print(clf.coef_)


if __name__ == '__main__':
    X = dataR[0]
    y = dataR[1]
    # sk_linear_regression(X,y)
    closed_formed(X, y)
