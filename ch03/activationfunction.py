import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    """
    阶跃函数
    :param x: numpy的数组
    :return: 函数值
    """
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    """
    sigmoid函数：1 / （1 + exp(-x)）
    :param x: 参数
    :return: 函数值
    """
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


if __name__ == '__main__':

    print(np.sum(softmax(np.array([0.3, 2.9, 4.]))))
