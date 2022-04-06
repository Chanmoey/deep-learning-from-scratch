import numpy as np

a = np.array([[1, 2, 3], [0, 1, 2], [3, 0, 1]])
b = np.array([[2, 0, 1], [0, 1, 2], [1, 0, 2]])
d = np.dot(a, b)
print(d)
print(np.sum(np.sum(d, axis=0), axis=0))
