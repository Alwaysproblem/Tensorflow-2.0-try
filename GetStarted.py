#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#%%
x0 = tf.constant(1.0)
x1 = tf.constant(2.0)

with tf.GradientTape(persistent=True) as g:
    g.watch([x0, x1])
    y = x0**2

    # with tf.GradientTape() as gg:
    #     gg.watch(x1)
    z = tf.math.log(x1)

tf.print(g.gradient(y, x0))
tf.print(y * g.gradient(z, x1))

del g