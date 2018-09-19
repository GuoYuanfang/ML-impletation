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
dataR = make_regression(n_samples=10, n_features=2, n_informative=1, noise=10)  # data-regression


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
    # eta = 0.05  # fixed learning rate
    num_sample, num_feature = X.shape
    one = np.ones((num_sample, 1))
    Xh = np.mat(np.hstack((X, one)))
    yh = np.mat(y).T
    wh = np.mat(np.ones((num_feature + 1, 1)))  # 保存上一个w值
    wn = np.mat(np.zeros((num_feature + 1, 1)))  # 保存最新的w值
    step = 0  # 记录迭代次数
    while True:
        if np.linalg.norm(wn - wh) < 0.001:
            break
        step += 1
        eta = np.exp(-step)  # descending gradient
        print("step{}: {}".format(step, wn))
        grad = 2 * Xh.T * (Xh * wn - yh)
        wh = wn
        wn = wn - grad * eta
    print("\n梯度下降算法共迭代{}次得到答案w:\n {}".format(step, wn))
    return wn


# 随机梯度下降法

def stochastic_gradient_decent(X, y):
    eta = 0.05  # fixed learning rate
    num_sample, num_feature = X.shape
    one = np.ones((num_sample, 1))
    Xh = np.mat(np.hstack((X, one)))
    yh = np.mat(y).T
    wh = np.mat(np.ones((num_feature + 1, 1)))  # 保存上一个w值
    wn = np.mat(np.zeros((num_feature + 1, 1)))  # 保存最新的w值
    step = 0  # 记录迭代次数
    while True:
        if np.linalg.norm(wn - wh) < 0.001:
            break
        # 输出信息
        step += 1
        print("step{}: {}".format(step, wn))

        sam_selected = np.random.randint(0, num_sample)  # 选择某一条sample求梯度

        xs = np.mat(Xh[sam_selected])
        ys = y[sam_selected]
        print(Xh.shape)
        print(xs.shape)
        print(wn.shape)
        grad = 2 * xs.T * (xs * wn - ys)
        wh = wn
        wn = wn - grad * eta

    print("\n随机梯度下降算法共迭代{}次得到答案w:\n {}".format(step, wn))
    return wn


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
    # closed_formed(X, y)
    # gradient_decent(X, y)
    stochastic_gradient_decent(X, y)
