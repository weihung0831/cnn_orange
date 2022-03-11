import time

import pretty_errors
import tensorflow as tf
from icecream import ic
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from plot import matrix_plot
from processing import load_test_data
from train_model_and_predict import pred_test_img


def result_csv(file_list, y_true, y_pred):
    result = np.column_stack((file_list, y_true.astype('int'), y_pred))
    result = pd.DataFrame(result)
    result.columns = ['image_file', 'label', 'prediction']
    result.to_csv('./model/cnn_orange/Results.csv')
    
    return result


if __name__ == '__main__':
    time_start = time.time()

    test_img, y_true, file_list = load_test_data(test_yellow_dir='./dataset/test/yellow/',
                                                 test_green_yellow_dir='./dataset/test/green_yellow/',
                                                 test_green_dir='./dataset/test/green/')
    model = tf.keras.models.load_model('model/cnn_orange/model_12_0.39.tf')
    y_pred = pred_test_img(model, test_img)

    accuracy_score = accuracy_score(y_true, y_pred)
    ic(accuracy_score)

    matrix = confusion_matrix(y_true, y_pred)
    matrix_plot = matrix_plot(matrix)

    classification_report = classification_report(y_true, y_pred,
                                                  target_names=['yellow', 'green_yellow', 'green'])
    ic(classification_report)

    result_csv = result_csv(file_list, y_true, y_pred)

    time_end = time.time()
    spend_time = ((time_end - time_start).__format__('.2f')) + 's'
    ic(spend_time)
