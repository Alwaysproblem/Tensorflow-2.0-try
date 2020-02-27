#%%
import tensorflow as tf2
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, Input

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
model.predict(X[0: 1])
# %%
model_clone = tf2.keras.models.clone_model(model)
model_clone.summary()
# %%
model_clone.set_weights(model.get_weights())
model_clone.predict(X[0:1])

# %% clone one layer
def oneLayer(clone_layer, input_dim = 2, output_dim = 1, hidden = 32):
    inputs = Input(name="data", shape=(input_dim,))
    lr_layer = layers.Dense(hidden).from_config(clone_layer.get_config())
    lr_l = lr_layer(inputs)
    model = tf2.keras.Model(inputs=inputs, outputs=lr_l)
    return model

#%%
model_onelayer = oneLayer(model.get_layer("linear"))
#%% clone one layer with weights.
w = model.get_layer("linear").get_weights()
model_onelayer.get_layer("linear").set_weights(w)
# %%
model_onelayer.predict(X[0:1])

# %%
