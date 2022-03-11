import os
import time

import pretty_errors
import tensorflow as tf
from icecream import ic

from model import cnn_orange
from processing import load_train_data
from train_model_and_predict import compile_and_train_model
import matplotlib.pyplot as plt

time_start = time.time()

train_x, train_y = load_train_data(train_yellow_dir='./dataset/train/yellow/',
                                   train_green_yellow_dir='./dataset/train/green_yellow/',
                                   train_green_dir='./dataset/train/green/')
model = compile_and_train_model(cnn_orange(), train_x / 255., train_y)

time_end = time.time()
spend_time = ((time_end - time_start).__format__('.2f')) + 's'
ic(spend_time)
