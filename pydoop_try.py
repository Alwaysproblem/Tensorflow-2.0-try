#%%
import pydoop.hdfs as hdfs
import tensorflow as tf2
import numpy as np
from tensorflow.keras import layers, Input

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

X = np.concatenate([x_red, x_green])
# X = np.concatenate([np.ones((sample_n*2, 1)), X], axis = 1)
y = np.concatenate([y_red, y_green])
y = y[:, None]

#%%
def lr(input_dim = 2, output_dim = 1, hidden = 32):
    inputs = Input(name="data", shape=(input_dim,))
    lr_l = layers.Dense(hidden, activation="relu", name = "linear", )(inputs)
    outputs = layers.Dense(output_dim, 
            activation='sigmoid', use_bias=True)(lr_l)
    
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)

    return model
#%%
model = lr()
model.summary()

#%%
model.compile('adam', loss=tf2.losses.BinaryCrossentropy())
model.fit(X, y, epochs=1, batch_size = 10)
model.save("hdfs:///tmp/yongxi/tfoutput/mnist_export/")