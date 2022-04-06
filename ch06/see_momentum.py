import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_gradient

from ch06.sgd import SGD


def f(x):
    return np.power(x[0], 2) / 20 + np.power(x[1], 2)


op = SGD()
start = np.array([-7.0, 2.0])
point = []
v = np.zeros_like(start)
lr = 0.01
for i in range(10000):
    point.append(start.copy())
    grad = numerical_gradient(f, start)
    v = 0.9 * v - lr * grad
    start += v
point = np.array(point)
print(point)
plt.scatter(point[:, 0], point[:, 1])
plt.show()
