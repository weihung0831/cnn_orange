import pretty_errors
import tensorflow as tf


def cnn_orange():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    inputs = tf.keras.layers.Input(shape=(300, 300, 3))
    x = tf.keras.layers.GaussianNoise(0.05)(inputs)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(13, 13), padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(11, 11), padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 9), padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=8, activation='relu')(x)
    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    x = tf.keras.layers.Dense(units=64, activation='tanh')(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(units=3, activation='softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
 

# model = model2()
# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True, to_file='model.png')
