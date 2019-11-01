import itertools
import tensorflow as tf

def gen():
    for i in range(1, 4):
        yield (i, [1] * i)

ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

ds = ds.repeat(3)

for value in ds.take(4):
    print(value)
# (1, array([1]))
# (2, array([1, 1]))