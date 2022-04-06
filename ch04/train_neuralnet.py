import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= lr * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(loss)

plt.plot([i for i in range(len(train_loss_list))], train_loss_list)
plt.show()
