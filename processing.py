import os

import cv2
import numpy as np
import pretty_errors
from icecream import ic
from sklearn.preprocessing import OneHotEncoder


def load_img(dir, label_num):
    image, label = [], []
    image_files = os.listdir(dir)
    ic(len(image_files))
    image = np.zeros((len(image_files), 300, 300, 3))
    label = np.ones((len(image_files), 1)) * label_num
    i = 0
    for files in image_files:
        ic(files)
        path = os.path.join(dir, files)
        img = cv2.imread(path)
        resize = cv2.resize(img, (300, 300))
        # resize = np.expand_dims(resize, axis=-1)
        ic(resize.shape)
        image[i] = resize
        i += 1

    return image, label

        
def concatenate_image(train_green_img, train_green_yellow_img, train_yellow_img):
    train_x = np.concatenate([train_green_img, train_green_yellow_img, train_yellow_img])
    # ic(train_x)
    
    return train_x


def concatenate_label(train_green_label, train_green_yellow_label, train_yellow_label):
    train_y = np.concatenate([train_green_label, train_green_yellow_label, train_yellow_label])
    # ic(train_y)
    
    return train_y


def onehot_encoder(label):
    onehot = OneHotEncoder()
    onehot.fit(label)
    onehot = onehot.transform(label).toarray()

    return onehot


def load_train_data(train_yellow_dir, train_green_yellow_dir, train_green_dir):
    train_yellow_img, train_yellow_label = load_img(train_yellow_dir, 0)
    train_green_yellow_img, train_green_yellow_label = load_img(train_green_yellow_dir, 1)
    train_green_img, train_green_label = load_img(train_green_dir, 2)
    ic(len(os.listdir(train_yellow_dir)))
    ic(len(os.listdir(train_green_yellow_dir)))
    ic(len(os.listdir(train_green_dir)))
    train_x = concatenate_image(train_yellow_img, train_green_yellow_img, train_green_img)
    train_y = concatenate_label(train_yellow_label, train_green_yellow_label, train_green_label)
    train_y = train_y.reshape(-1, 1)
    train_y = onehot_encoder(train_y)
    # ic(train_x)
    # ic(train_y)
    np.random.seed(123)
    np.random.shuffle(train_x)
    np.random.seed(123)
    np.random.shuffle(train_y)
    # ic(train_x)
    # ic(train_y)
    
    return train_x, train_y


def load_test_data(test_yellow_dir, test_green_yellow_dir, test_green_dir):
    test_yellow_img, test_yellow_label = load_img(test_yellow_dir, 0)
    test_green_yellow_img, test_green_yellow_label = load_img(test_green_yellow_dir, 1)
    test_green_img, test_green_label = load_img(test_green_dir, 2)
    # ic(len(os.listdir(test_yellow_dir)))
    # ic(len(os.listdir(test_green_yellow_dir)))
    # ic(len(os.listdir(test_green_dir)))
    file_list = []
    for dir in [test_yellow_dir, test_green_yellow_dir, test_green_dir]:
        file_list += sorted(os.listdir(dir))
    test_img = np.concatenate([test_yellow_img, test_green_yellow_img, test_green_img])
    y_true = np.concatenate([test_yellow_label, test_green_yellow_label, test_green_label])
    # ic(test_img)
    # ic(y_true)
    
    return test_img, y_true, file_list
