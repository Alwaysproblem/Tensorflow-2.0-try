#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
#%%
tf.keras.backend.set_floatx('float64')
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
header = dataframe.head()
#%%
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

#%%
batch_size = 5 # A small batch sized is used for demonstration purposes
ds = df_to_dataset(dataframe, shuffle=False, batch_size=batch_size)

#%%
for feature_batch, label_batch in ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )

#%%
example_batch, label = next(iter(ds))
# example_batch['oldpeak'] = tf.
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

#%%
age = feature_column.numeric_column("age")
demo(age)

# %%
chol = feature_column.embedding_column(
    feature_column.categorical_column_with_identity(
      "chol", 300), 3)

demo(chol)
#%%
inputs = {h:layers.Input(name = h,shape=(1,), dtype = tf.int32) for h in example_batch}
features = layers.DenseFeatures(chol)(inputs)
outputs = layers.Dense(1, activation='sigmoid')(features)
model = tf.keras.Model(inputs = inputs, outputs=outputs)

# model = tf.keras.Sequential([
#   layers.DenseFeatures(chol),
#   layers.Dense(1, activation='sigmoid')
# ])

model.compile(optimizer='sgd', loss=tf.losses.BinaryCrossentropy(), )
# model.build(input_shape = (None, None))
# model.summary()
model.fit(x = example_batch, y=label, epochs = 1)
# %%
