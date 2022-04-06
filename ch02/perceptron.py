import numpy as np


def AND(x1, x2):
    w = np.array([0.5, 0.5])
    theta = -0.7
    tmp = np.sum(w * np.array([x1, x2])) + theta
    if tmp <= 0:
        return 0
    return 1


def NAND(x1, x2):
    w = np.array([-0.5, -0.5])
    theta = 0.7
    tmp = np.sum(w * np.array([x1, x2])) + theta
    if tmp <= 0:
        return 0
    return 1


def OR(x1, x2):
    w = np.array([0.5, 0.5])
    theta = -0.2
    tmp = np.sum(w * np.array([x1, x2])) + theta
    if tmp <= 0:
        return 0
    return 1


def XOR(x1, x2):
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    return AND(s1, s2)


print(XOR(1, 1))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(0, 0))
