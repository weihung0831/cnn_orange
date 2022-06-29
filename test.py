import matplotlib.pyplot as plt
import numpy as np
import pretty_errors
import tensorflow as tf
from icecream import ic
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model


def testing(test_img, y_true, class_names, model_path,
            confusion_matrix_fig_path):
    model = load_model(model_path)
    y_pred = model.predict(test_img)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    # ic(y_true, y_pred, y_true.shape, y_pred.shape)

    ic(accuracy_score(y_true, y_pred))
    ic(classification_report(y_true, y_pred, target_names=class_names))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                           display_labels=class_names).plot()
    plt.savefig(confusion_matrix_fig_path)


def main():
    data = np.load(file="./dataset/dataset.npz")
    x_test, y_test = (data["test_img"], data["test_label"])
    ic(x_test.shape, y_test.shape)

    testing(
        x_test,
        y_test,
        class_names=["green", "green_yellow", "yellow"],
        model_path="./model/cnn_orange/model.tf/",
        confusion_matrix_fig_path="./model/cnn_orange/confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
