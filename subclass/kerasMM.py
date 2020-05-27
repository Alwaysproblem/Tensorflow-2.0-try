#%%
import tensorflow as tf2
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# %matplotlib inline

# logdir = os.path.split(os.path.realpath(__file__))[0]
#%%
# generate some data 2-dimension. shape = (10, 2)

noise = np.random.randn(10, 1)
# print(noise.shape)
W = np.array([[5., 3.]]).T
x = np.array(np.arange(10))[None, :].T
# print(x.shape)
x_ = np.ones((10, 1))
X = np.concatenate((x_, x), axis = 1)

print(X.shape, W.shape)
y = X @ W + noise
print(y)

# plt.scatter(x, y, s = 10, marker='o', c = 'red')
# plt.show()

#%%
class linearM(tf2.keras.Model):
    def __init__(self):
        super().__init__()
        self.W = tf2.cast(tf2.Variable(np.ones((2,1))), tf2.float32)
    
    def call(self, x):
        return x @ self.W

def loss(pred, label):
    return tf2.reduce_mean(tf2.square(pred - label))

opt = tf2.keras.optimizers.Adam(0.1)
# opt = tf2.optimizers.SGD(0.02)

#%%
# @tf2.function
def train(model, inputs, outputs, loss = loss, opt = opt):
    # can directly use the mminimize funciton
    inputs = tf2.cast(inputs, tf2.float32)

    Loss = loss(model(inputs), outputs)
    opt.minimize(lambda: loss(model(inputs), outputs), model.trainable_variables)
    
    return Loss

#%%
m = linearM()
W_hat_l = [m.W.numpy()]
cl = []
epochs = np.arange(500)

# writer = tf2.summary.create_file_writer(logdir + "/logs/" + time.strftime('%Y-%m-%d_%H-%M-%S'))

# @tf2.function
# def to(step, name, data):
#     with writer.as_default():
#         tf2.summary.scalar(name, data, step=step)


# def tfHis(step, data):
#     with writer.as_default():
#         tf2.summary.histogram("w", data, step=step)

#%%
for i in epochs:
    closs = train(m, X, y)
    # to(i, "closs", closs.numpy())
    # tfHis(i, m.W.numpy())
    cl.append(closs)
    W_hat_l.append(m.W.numpy())
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %(i, m.W[0, 0], m.W[1,0], closs))
    # writer.flush()

W_hat_l = np.concatenate(W_hat_l, axis = 1)

plt.plot(epochs, cl)
plt.show()
#%%
plt.plot(epochs, [W[0, 0]] * len(epochs), 'r--')
plt.plot(epochs, [W[1, 0]] * len(epochs), 'b--')
plt.plot(epochs, W_hat_l[0, :-1], 'r.-')
plt.plot(epochs, W_hat_l[1, :-1], 'b.-')
plt.show()

#%%
