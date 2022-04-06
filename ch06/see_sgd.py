import numpy as np
import matplotlib.pyplot as plt

from ch06.sgd import SGD


def f(x):
    return np.power(x[0], 2) / 20 + np.power(x[1], 2)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad

# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
#
# X, Y = np.meshgrid(x, y)
#
# Z = f(np.array([X, Y]))
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X, Y, Z, color='black')
# ax.set_title('wireframe')
# plt.show()


start = np.array([-7.0, 2.0])
point = []

lr = 0.95
for i in range(10000):
    point.append(start.copy())
    grad = numerical_gradient(f, start)
    start -= lr * grad
point = np.array(point)
print(point)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter(point[:, 0], point[:, 1])
plt.show()
