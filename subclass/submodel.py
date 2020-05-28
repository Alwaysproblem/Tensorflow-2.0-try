# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#%%
import tensorflow as tf
from tensorflow.keras import Input, layers
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
from sklearn.model_selection import train_test_split as tvsplit
# disable logging warning and error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.experimental_run_functions_eagerly(True)
#%%
sample_n = 100
epochs = 50
#%%
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

X = np.concatenate([x_red, x_green]).astype(np.float32)
y = np.concatenate([y_red, y_green]).astype(np.float32)

#%%
X_train, X_test, y_train, y_test = tvsplit(X, y)

#%%
class Logistic(tf.keras.models.Model):
    def __init__(self, hidden_size = 5, output_size=1, dynamic=False, **kwargs):
        '''
        name: String name of the model.
        dynamic: (Subclassed models only) Set this to `True` if your model should
            only be run eagerly, and should not be used to generate a static
            computation graph. This attribute is automatically set for Functional API
            models.
        trainable: Boolean, whether the model's variables should be trainable.
        dtype: (Subclassed models only) Default dtype of the model's weights (
            default of `None` means use the type of the first input). This attribute
            has no effect on Functional API models, which do not have weights of their
            own.
        '''
        super().__init__(dynamic=dynamic, **kwargs)
        self.InputLayer = tf.keras.Input(shape=(2,))
        # self.InputLayer = {"a": tf.keras.Input(shape=(2,)), "b": tf.keras.Input(shape=(2,))}
        self.hidden_size = hidden_size
        self.dense = layers.Dense(hidden_size, name = "linear")
        self.outlayer = layers.Dense(output_size, 
                        activation = 'sigmoid', name = "out_layer")
        
        # add this method, it will build model automatically as TF says
        self._set_inputs(self.InputLayer)

    # # this is for model.summary and build manually
    # def _build(self, input_shape):
    #     """Builds the model based on input shapes received.
    #     This is to be used for subclassed models, which do not know at instantiation
    #     time what their inputs look like.
    #     This method only exists for users who want to call `model.build()` in a
    #     standalone way (as a substitute for calling the model on real data to
    #     build it). It will never be called by the framework (and thus it will
    #     never throw unexpected errors in an unrelated workflow).
    #     Args:
    #         input_shape: Single tuple, TensorShape, or list of shapes, where shapes
    #             are tuples, integers, or TensorShapes.
    #     Raises:
    #         ValueError:
    #         1. In case of invalid user-provided data (not of type tuple,
    #             list, or TensorShape).
    #         2. If the model requires call arguments that are agnostic
    #             to the input shapes (positional or kwarg in call signature).
    #         3. If not all layers were properly built.
    #         4. If float type inputs are not supported within the layers.
    #         In each of these cases, the user should build their model by calling it
    #         on real tensor data.
    #     """
    #     from tensorflow.python.framework import errors
    #     from tensorflow.python.framework import tensor_shape
    #     from tensorflow.python.framework import func_graph
    #     from tensorflow.python.keras import backend
    #     from tensorflow.python.keras.engine import base_layer
    #     from tensorflow.python.keras.engine import base_layer_utils
    #     from tensorflow.python.eager import context
        
    #     if self._is_graph_network:
    #         super().build(input_shape)
    #         return

    #     # If subclass network
    #     if input_shape is None:
    #         raise ValueError('Input shape must be defined when calling build on a '
    #                         'model subclass network.')
    #     valid_types = (tuple, list, tensor_shape.TensorShape, dict)
    #     if not isinstance(input_shape, valid_types):
    #         raise ValueError('Specified input shape is not one of the valid types. '
    #                         'Please specify a batch input shape of type tuple or '
    #                         'list of input shapes. User provided '
    #                         'input type: {}'.format(type(input_shape)))

    #     if input_shape and not self.inputs:
    #         # We create placeholders for the `None`s in the shape and build the model
    #         # in a Graph. Since tf.Variable is compatible with both eager execution
    #         # and graph building, the variables created after building the model in
    #         # a Graph are still valid when executing eagerly.
    #         if context.executing_eagerly():
    #             graph = func_graph.FuncGraph('build_graph')
    #         else:
    #             graph = backend.get_graph()
    #         with graph.as_default():
    #             if isinstance(input_shape, list):
    #                 x = [base_layer_utils.generate_placeholders_from_shape(shape)
    #                     for shape in input_shape]
    #             elif isinstance(input_shape, dict):
    #                 x = {
    #                     k: base_layer_utils.generate_placeholders_from_shape(shape)
    #                     for k, shape in input_shape.items()
    #                 }
    #             else:
    #                 x = base_layer_utils.generate_placeholders_from_shape(input_shape)

    #         kwargs = {}
    #         call_signature = self._call_full_argspec
    #         call_args = call_signature.args
    #         # Exclude `self`, `inputs`, and any argument with a default value.
    #         if len(call_args) > 2:
    #             if call_signature.defaults:
    #                 call_args = call_args[2:-len(call_signature.defaults)]
    #             else:
    #                 call_args = call_args[2:]
    #             for arg in call_args:
    #                 if arg == 'training':
    #                     # Case where `training` is a positional arg with no default.
    #                     kwargs['training'] = False
    #                 else:
    #                     # Has invalid call signature with unknown positional arguments.
    #                     raise ValueError(
    #                         'Currently, you cannot build your model if it has '
    #                         'positional or keyword arguments that are not '
    #                         'inputs to the model, but are required for its '
    #                         '`call` method. Instead, in order to instantiate '
    #                         'and build your model, `call` your model on real '
    #                         'tensor data with all expected call arguments.')
    #         elif len(call_args) < 2:
    #             # Signature without `inputs`.
    #             raise ValueError('You can only call `build` on a model if its `call` '
    #                             'method accepts an `inputs` argument.')
    #         try:
    #             self.call(x, **kwargs)
    #         except (errors.InvalidArgumentError, TypeError):
    #             raise ValueError('You cannot build your model by calling `build` '
    #                             'if your layers do not support float type inputs. '
    #                             'Instead, in order to instantiate and build your '
    #                             'model, `call` your model on real tensor data (of '
    #                             'the correct dtype).')

    #     base_layer.Layer.build(self, input_shape)

    # def build(self):
    #     """
    #     InputLayer should be like:
    #     InputLayer = [tf.keras.Input(shape=(2,)), ...]
    #     InputLayer = {"a": tf.keras.Input(shape=(2,)), ...}
    #     InputLayer = tf.keras.Input(shape=(2,))
    #     """
    #     if not hasattr(self, "InputLayer"):
    #         raise AttributeError("User should define InputLayer in sub-class model.")
        
    #     from tensorflow import is_tensor
    #     if isinstance(self.InputLayer, list):
    #         input_shape = [tuple(i.shape.as_list()) for i in self.InputLayer]
    #     elif isinstance(self.InputLayer, dict):
    #         # input_shape = [tuple(self.InputLayer[i].shape.as_list()) for i in self.InputLayer]
    #         input_shape = {i: tuple(self.InputLayer[i].shape.as_list()) for i in self.InputLayer}
    #     elif is_tensor(self.InputLayer):
    #         input_shape = tuple(self.InputLayer.shape.as_list())
    #     else:
    #         raise TypeError(f"the type of self.InputLayer is {type(self.InputLayer)} expected either not nested list, dict or Tensor")

    #     self._build(input_shape)
    #     if not hasattr(self, 'call'):
    #         raise AttributeError("User should define 'call' method in sub-class model.")
    #     _ = self.call(self.InputLayer)


    # for 2.2 there is no need to think about signature definination name.
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32)])
    # @tf.function
    def call(self, X):
        # tf.print(X)
        # X = X["a"]
        X = self.dense(X)
        Y = self.outlayer(X)
        return Y

#%%
model = Logistic()
model.summary()

# %%
optimizer=tf.keras.optimizers.Adam()
loss=tf.keras.losses.BinaryCrossentropy()
metrics=tf.keras.metrics.AUC()

#%%
# @tf.function # in graph mode
def losses(y_true, y_pred, sample_weight=None, loss_obj=loss):
    return loss_obj(y_true, y_pred, sample_weight)

#%%
# @tf.function # in graph mode
def Metrics(y_true, y_pred, sample_weight=None, metrics=metrics):
    metrics.update_state(y_true, y_pred, sample_weight)
    return metrics.result()
#%%
# @tf.function # in graph mode
def grad(model, inputs, labels):
    
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss_value = losses(labels, pred)
        labels = tf.expand_dims(labels, axis=1)
        metr = Metrics(labels, pred)
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables), metr

# %%
# @tf.function # in graph mode
def train_on_batch(model, inputs, labels, opt=optimizer):
    closs, cgrad, cmetric = grad(model, inputs, labels)
    opt.apply_gradients(zip(
        cgrad,
        model.trainable_variables
    ))
    return closs, cmetric

#%%
for e in range(epochs):
    for ind in range(len(X_train)):
        loss_value, cmetric = train_on_batch(model, X_train[ind][None, :], 
                np.expand_dims(y_train[ind], axis=0))
    # tf.print(f"Epochs {e}: loss {loss_value.numpy()}, metric:{cmetric.numpy()}") # in graph mode
    print(f"Epochs {e}: loss {loss_value.numpy()}, metric:{cmetric.numpy()}")

#%%
model.save("save/subclass/1")