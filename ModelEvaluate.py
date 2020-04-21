#%%
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
_, mnist_test = datasets['train'], datasets['test']

#%%
untest = tf.keras.models.load_model(
    "modelSaved.h5", custom_objects=None, compile=True
)

#%%
mnist_test = mnist_test.batch(32)
#%%
untest.evaluate(mnist_test)

# %%
