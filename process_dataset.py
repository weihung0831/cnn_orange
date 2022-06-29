import pretty_errors
from icecream import ic
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def load_image(directory, label_num):
    image, label = [], []
    for i in tqdm(os.listdir(directory)):
        path = os.path.join(directory, i)
        img = cv2.imread(path)
        img = cv2.resize(img, (300, 300))
        image.append(img)
        label.append(label_num)
    image = np.array(image, dtype="float32")
    label = np.array(label)

    return image, label


def main():
    train_green_img, train_green_label = load_image(
        directory="./dataset/train/green/", label_num=0)
    train_green_yellow_img, train_green_yellow_label = load_image(
        directory="./dataset/train/green_yellow/", label_num=1)
    train_yellow_img, train_yellow_label = load_image(
        directory="./dataset/train/yellow/", label_num=2)
    test_green_img, test_green_label = load_image(
        directory="./dataset/test/green/", label_num=0)
    test_green_yellow_img, test_green_yellow_label = load_image(
        directory="./dataset/test/green_yellow/", label_num=1)
    test_yellow_img, test_yellow_label = load_image(
        directory="./dataset/test/yellow/", label_num=2)

    x_train = np.concatenate(
        [train_green_img, train_green_yellow_img, train_yellow_img])
    x_test = np.concatenate(
        [test_green_img, test_green_yellow_img, test_yellow_img])
    y_train = np.concatenate(
        [train_green_label, train_green_yellow_label, train_yellow_label])
    y_test = np.concatenate(
        [test_green_label, test_green_yellow_label, test_yellow_label])

    np.random.seed(123)
    np.random.shuffle(x_train)
    np.random.seed(123)
    np.random.shuffle(y_train)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train = OneHotEncoder().fit(y_train).transform(y_train).toarray()
    y_test = OneHotEncoder().fit(y_test).transform(y_test).toarray()

    np.savez("./dataset/dataset.npz",
             train_img=x_train,
             test_img=x_test,
             train_label=y_train,
             test_label=y_test)


if __name__ == "__main__":
    main()
