import tensorflow as tf

class mnist_model(tf.keras.models.Model):
    def __init__(self, hparams, input_shape ,**kwargs):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu)
        self.dropout = layers.Dropout(hparams[HP_DROPOUT])
        self.outLayer = layers.Dense(10, activation=tf.nn.softmax)
        self.build(input_shape)

    # since tf 2.2 there is no need to think about input spec

    # @tf.function
    def call(self, X):
        X = self.flatten(X)
        X = self.dense(X)
        X = self.dropout(X)
        y = self.outLayer(X)

        return y
