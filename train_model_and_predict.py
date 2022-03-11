import datetime

import numpy as np
import pretty_errors
import tensorflow as tf
from icecream import ic

from plot import training_model_plot


def compile_and_train_model(model, train_x, train_y):
    model = model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy'])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=5,
                                                  verbose=0,
                                                  mode='max'),
                 tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_test/model_{epoch:02d}_{val_loss:.2f}.tf',                                                    
                                                    save_best_only=True)]
    history = model.fit(x=train_x,
                        y=train_y,
                        epochs=100,
                        batch_size=1,
                        validation_split=0.2,
                        callbacks=callbacks)
    training_model_plot(history)

    return model


def pred_test_img(model, test_img):
    model = model
    model.summary()

    y_pred = model.predict(test_img)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred.astype('int')
    ic(y_pred)

    return y_pred
