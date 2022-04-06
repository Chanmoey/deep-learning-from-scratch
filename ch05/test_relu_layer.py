import numpy as np

from ch05.layer import Relu

x = np.array([1, 2, 3, -2, 2, -5, -6])
relu = Relu()
fx = relu.forward(x)
print(fx)
bx = relu.backward(x)
print(bx)
