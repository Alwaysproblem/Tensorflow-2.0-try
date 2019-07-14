#%%
import tensorflow as tf2
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
tfd = tfp.distributions

printf = tf2.print
#%%
# n = tfd.Normal(0., 1.)

# n.log_prob()
#%%
a = np.random.randint(0, 10000, size=(3, 2, 4))
b = np.random.randint(0, 10000, size=(3, 4, 1))

outcome = np.tensordot(a, b, axes = [[2], [1]])
print(outcome.shape)

#%%
