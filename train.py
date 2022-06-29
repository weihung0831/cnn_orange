import matplotlib.pyplot as plt
import numpy as np
import pretty_errors
import tensorflow as tf
from icecream import ic
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, GaussianNoise,
                                     GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.optimizers import Adam


def cnn_orange(inputs_shape):
    inputs = Input(inputs_shape)
    x = GaussianNoise(0.05)(inputs)
    x = Conv2D(8, (13, 13), padding='valid', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(16, (11, 11), padding='valid', activation='relu')(x)
    x = Conv2D(32, (9, 9), padding='valid', activation='relu')(x)
    x = Conv2D(64, (7, 7), padding='valid', activation='relu')(x)
    x = Conv2D(128, (5, 5), padding='valid', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def visualize_training_results(history, history_plot_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(212)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(history_plot_path)


def training(model, x_train, y_train, history_plot_path):
    model.compile(Adam(learning_rate=1e-4),
                  loss="CategoricalCrossentropy",
                  metrics="accuracy")
    history = model.fit(
        x_train,
        y_train,
        batch_size=8,
        epochs=200,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(
                filepath="./model/cnn_orange/model-{val_loss:.2f}.tf",
                monitor='val_loss',
                mode='min')
        ])
    visualize_training_results(history, history_plot_path)


def main():
    data = np.load("./dataset/dataset.npz")
    x_train, y_train = data["train_img"], data["train_label"]
    # ic(x_train, y_train)
    ic(x_train.shape, y_train.shape)

    training(cnn_orange(x_train.shape[1:]),
             x_train / 255.,
             y_train,
             history_plot_path="./model/cnn_orange/history.png")


if __name__ == '__main__':
    main()
