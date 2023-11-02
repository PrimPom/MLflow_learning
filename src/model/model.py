import tensorflow as tf

keras = tf.keras


class Model:
    @classmethod
    def create_basic_model(cls, hyper_params):
        model = keras.models.Sequential([
            keras.layers.Dense(1, input_shape=(2,))
        ])
        # optimizer = keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
        learning_rate = hyper_params.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        # print("Learning rate :"+str(learning_rate))
        model.compile(loss=keras.losses.Huber(),
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=["mae"], run_eagerly=True)
        return model


    @classmethod
    def checkpoint_callback(cls, checkpoint_path):
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
        return cp_callback
