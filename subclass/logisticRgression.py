#%%
import tensorflow as tf2
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# %matplotlib inline

#%%
# generate some data 2-dimension. shape = (10, 2)
sample_n = 100
meana = np.array([1, 1])
cova = np.array([[0.1, 0],[0, 0.1]])

meanb = np.array([2, 2])
covb = np.array([[0.1, 0],[0, 0.1]])

x_red = np.random.multivariate_normal(mean=meana, cov = cova, size=sample_n)
x_green = np.random.multivariate_normal(mean=meanb, cov = covb, size=sample_n)

y_red = np.array([1] * sample_n)
y_green = np.array([0] * sample_n)

plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
plt.show()

X = np.concatenate([x_red, x_green])
X = np.concatenate([np.ones((sample_n*2, 1)), X], axis = 1)
y = np.concatenate([y_red, y_green])

y = y[:, None]

assert X.shape == (sample_n*2, 3)
assert y.shape == (sample_n*2, 1)

#%%
class LogR():
    def __init__(self):
        self.W = tf2.Variable(np.random.randn(3, 1))

    def __call__(self, x):
        return tf2.sigmoid(x @ self.W)
        # return x @ self.W

#%%
def loss(pred, label):
    return tf2.reduce_mean(tf2.losses.binary_crossentropy(label, pred))

def loss_np(pred, label):
    return tf2.reduce_mean(- y[None, :]*np.log(pred) - (1.0 - y[None, :]) * np.log(1.0 - pred))

def loss_svm(pred, label, model):
    label = tf2.cast(label, tf2.float32)
    # label = (label - 0.5) * 2
    pred = tf2.cast(pred, tf2.float32)
    o = tf2.reduce_mean(tf2.maximum(0.0, 1.0 - tf2.cast(pred * label, tf2.float32)))  + tf2.cast(0.0 * (tf2.transpose(model.W) @ model.W), tf2.float32)
    return tf2.reshape(o, (1,))

#%%
# def sigmoid(X):
#     return 1.0 / 1.0 + np.exp(-X)

#%%
@tf2.function
def train(model, inputs, outputs, loss = loss, opt = tf2.optimizers.Adam(0.1)):
    with tf2.GradientTape() as g:
        g.watch(model.W)
        closs = loss(model(X), outputs)
        # closs = loss_svm(model(X), outputs, model)
    #     # tf2.print(closs)
    dW = g.gradient(closs, model.W)
    # model.W -= lr * dW
    # model.W.assign_sub(lr * dW)
    opt.apply_gradients(zip([dW], [model.W]))
    return closs

#%%
m = LogR()
epochs = np.arange(1000)
l = []
for i in epochs:
    closs = train(m, X, y)
    l.append(closs)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %(i, m.W[0, 0], m.W[1,0], closs))
    
    # if i % 100:
    #     x = np.arange(-1, 3, step = 0.01)
    #     y = - m.W[0,0] / m.W[2, 0] - m.W[1, 0] * x/ m.W[2, 0]
    #     plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
    #     plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
    #     plt.plot(x, y)
    #     plt.show()

#%%
plt.plot(epochs, l)
plt.show()
#%%
x = np.arange(-1, 3, step = 0.01)
y = - m.W[0,0] / m.W[2, 0] - m.W[1, 0] * x/ m.W[2, 0]
plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
plt.plot(x, y)
plt.show()
#%%