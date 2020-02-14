#%%
import tensorflow as tf
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, Input
import tensorflow.keras.backend as K
from tensorflow.python import ops
import tensorflow.python.ops as math_ops

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# add this piece of code, you can debug freely.
tf.config.experimental_run_functions_eagerly(True)
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
# plt.show()

X = np.concatenate([x_red, x_green])
# X = np.concatenate([np.ones((sample_n*2, 1)), X], axis = 1)
y = np.concatenate([y_red, y_green])
y = y[:, None]
y = np.array(y, dtype=np.float32)

#%% need to be test.
def loss_carrier(extra_param1, extra_param2):
    def loss(y_true, y_pred):
        #x = complicated math involving extra_param1, extraparam2, y_true, y_pred
        #remember to use tensor objects, so for example keras.sum, keras.square, keras.mean
        #also remember that if extra_param1, extra_maram2 are variable tensors instead of simple floats,
        #you need to have them defined as inputs=(main,extra_param1, extraparam2) in your keras.model instantiation.
        #and have them defind as keras.Input or tf.placeholder with the right shape.
        x = y_true - y_pred
        return x
    return loss

#%%
def lr_multiouput(input_dim = 2, output_dim = 1, hidden = 32):
    inputs = Input(name="data", shape=(input_dim, ))
    lr_l = layers.Dense(hidden, activation="relu")(inputs)
    outputs = layers.Dense(output_dim, 
            activation='sigmoid', use_bias=True)(lr_l)
    
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, lr_l])

    return model
#%%
model= lr_multiouput()

#%% 
# the first method to solve multivarient customized loss function.
# this not hold on with the dataset API.

@tf.function
def loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred[0])
    l = tf.math.reduce_sum(tf.square(y_pred[1]), axis = -1)
    return l + bce

model.summary()
model.compile('adam', loss=loss)
model.fit(X, [y, y], epochs=2, batch_size = 10)

# %% the secend method the tensorflow turtorial deveplop to customize loss function.

class LossLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.ones = self.add_weight(name="constant_loss", shape = (1,), 
                    initializer = tf.keras.initializers.ones(), trainable=False)
        super().build(input_shape)

    def call(self, X, mask=None):
        self.add_loss(self.ones)
        return X


# class MetricsLayer(tf.keras.layers.Layer):
#     def __init__(self, metrics, **kwargs):
#         super().__init__(self, **kwargs)
#         self.metric = metrics
    

#     def call(self, X, mask=None):
#         self.add_metric(self.metric())
#         return X

#%%
def lr_layer_add(input_dim = 2, output_dim = 1, hidden = 32):
    inputs = Input(name="data", shape=(input_dim, ))
    lr_l = layers.Dense(hidden, activation="relu")(inputs)
    outputs = layers.Dense(output_dim, 
            activation='sigmoid', use_bias=True)(lr_l)
    outputs = LossLayer()(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = lr_layer_add()
#%%
model.summary()
model.compile('adam', loss=tf.keras.losses.BinaryCrossentropy())
model.fit(X, y, epochs=2, batch_size = 10)

#%% try the custom metric function.
class MetricLayer(tf.keras.layers.Layer):
    def call(self, X):
        self.add_metric(tf.math.reduce_std(X[0]) + tf.math.reduce_std(X[1]), name="std_of_activation", aggregation="mean")
        return X[0]

def lr_metric(input_dim = 2, output_dim = 1, hidden = 32):
    inputs = Input(name="data", shape=(input_dim, ))
    lr_l = layers.Dense(hidden, activation="relu")(inputs)
    outputs = layers.Dense(output_dim, 
            activation='sigmoid', use_bias=True)(lr_l)
    outputs = MetricLayer()([outputs, lr_l])
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

#%%
model = lr_metric()
#%%
model.summary()
model.compile('adam', loss=tf.keras.losses.BinaryCrossentropy())
model.fit(X, y, epochs=2, batch_size = 10)

#%% 
# or you can use this code to substitute for model.fit function
# and custom the loss_function
# epochs = 3
# for epoch in range(epochs):
#   print('Start of epoch %d' % (epoch,))

#   # Iterate over the batches of the dataset.
#   for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

#     # Open a GradientTape to record the operations run
#     # during the forward pass, which enables autodifferentiation.
#     with tf.GradientTape() as tape:

#       # Run the forward pass of the layer.
#       # The operations that the layer applies
#       # to its inputs are going to be recorded
#       # on the GradientTape.
#       logits = model(x_batch_train, training=True)  # Logits for this minibatch

#       # Compute the loss value for this minibatch.
#       loss_value = loss_fn(y_batch_train, logits)

#     # Use the gradient tape to automatically retrieve
#     # the gradients of the trainable variables with respect to the loss.
#     grads = tape.gradient(loss_value, model.trainable_weights)

#     # Run one step of gradient descent by updating
#     # the value of the variables to minimize the loss.
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))

#     # Log every 200 batches.
#     if step % 200 == 0:
#         print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
#         print('Seen so far: %s samples' % ((step + 1) * 64))

# # Start of epoch 0