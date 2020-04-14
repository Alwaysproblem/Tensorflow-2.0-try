# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


def main_func():
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

    #%%
    sample_n = 100
    epochs = 200
    #%%
    meana = np.array([1, 1])
    cova = np.array([[0.1, 0],[0, 0.1]])

    meanb = np.array([2, 2])
    covb = np.array([[0.1, 0],[0, 0.1]])

    x_red = np.random.multivariate_normal(mean=meana, cov = cova, size=sample_n)
    x_green = np.random.multivariate_normal(mean=meanb, cov = covb, size=sample_n)

    y_red = np.array([1] * sample_n)
    y_green = np.array([0] * sample_n)

    X = np.concatenate([x_red, x_green]).astype(np.float32)
    y = np.concatenate([y_red, y_green]).astype(np.float32)


    #%%
    X_train, X_test, y_train, y_test = tvsplit(X, y)

    #%%
    mirrored_strategy = tf.distribute.MirroredStrategy()

    def logstic(input_size=2, hidden_size = 5, output_size=1):
        Inputs = Input(shape=(input_size,), name="Inputs")
        linear1 = layers.Dense(hidden_size)(Inputs)
        Outputs = layers.Dense(output_size, activation=tf.keras.activations.sigmoid)(linear1)
        model = tf.keras.Model(inputs=Inputs, outputs=Outputs, name="Logistic")
        return model
    
    with mirrored_strategy.scope(): 
        model = logstic()
        #%%
        model.compile(optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=["accuracy"])
    model.summary()
    #%%
    # tensorboard
    # logdir = "logs" + os.path.sep + "standard" + os.path.sep + datetime.now().strftime("""%Y%m%d-%H%M%S""")
    # callbacks = [
    #     tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    # ]
    #%%
    # model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test, y_test))
    history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test))
    #%%
    # tf.io.write_file("./")

if __name__ == "__main__":
    main_func()