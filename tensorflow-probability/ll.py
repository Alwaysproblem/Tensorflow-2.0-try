#%%
from tensorflow_probability import edward2 as ed
import tensorflow as tf

#%%
# @ed.make_log_joint_fn
def logistic_regression(features):
  coeffs = ed.Normal(loc=0., scale=1.,
                     sample_shape=features.shape[1], name="coeffs")
  outcomes = ed.Bernoulli(logits=tf.tensordot(features, coeffs, [[1], [0]]),
                          name="outcomes")
  return outcomes

log_joint = ed.make_log_joint_fn(logistic_regression)

#%%
features = tf.random_normal([3, 2])
coeffs_value = tf.random_normal([2])
outcomes_value = tf.round(tf.random_uniform([3]))
output = log_joint(features, coeffs=coeffs_value, outcomes=outcomes_value)