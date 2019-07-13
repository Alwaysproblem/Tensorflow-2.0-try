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

noise = np.random.randn(10, 1)
# print(noise.shape)
W = np.array([[5, 3]]).T
x = np.array(np.arange(10))[None, :].T
# print(x.shape)
x_ = np.ones((10, 1))
X = np.concatenate((x_, x), axis = 1)

print(X.shape, W.shape)
y = X @ W + noise
print(y)

plt.scatter(x, y, s = 10, marker='o', c = 'red')


#%%
class linearM():
    def __init__(self):
        self.W = tf2.Variable(np.ones((2,1)))
    
    def __call__(self, x):
        return x @ self.W


def loss(pred, label):
    return tf2.reduce_mean(tf2.square(pred - label))

opt = tf2.keras.optimizers.Adam(0.1)
# opt = tf2.optimizers.SGD(0.02)

#%%
@tf2.function
def train(model, inputs, outputs, loss = loss, opt = opt):
    # can directly use the mminimize funciton
    opt.minimize(lambda: loss(model(inputs), outputs), [model.W])
    a = loss(model(X), y)
    return a

#%%
m = linearM()
W_hat_l = [m.W.numpy()]
cl = []
epochs = np.arange(500)
#%%
for i in epochs:
    closs = train(m, X, y)
    cl.append(closs)
    W_hat_l.append(m.W.numpy())
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %(i, m.W[0, 0], m.W[1,0], closs))

W_hat_l = np.concatenate(W_hat_l, axis = 1)

plt.plot(epochs, cl)
#%%
plt.plot(epochs, [W[0, 0]] * len(epochs), 'r--')
plt.plot(epochs, [W[1, 0]] * len(epochs), 'b--')
plt.plot(epochs, W_hat_l[0, :-1], 'r.-')
plt.plot(epochs, W_hat_l[1, :-1], 'b.-')


#%%
