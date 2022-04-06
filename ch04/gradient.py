import numpy as np
import matplotlib.pyplot as plt

h = 1e-4


def numerical_diff(f, x):
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_gradient_no_batch(f, x):
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


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


colle_x1 = []
colle_x2 = []


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        colle_x1.append(x[0])
        colle_x2.append(x[1])
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


if __name__ == '__main__':
    init_x = np.array([-3., 4.])
    min_x = gradient_descent(function_2, init_x, lr=1e-10)
    print(min_x)
    plt.scatter(colle_x1, colle_x2)
    plt.show()
