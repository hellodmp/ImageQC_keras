# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np


def create_data():
    x_train = np.linspace(-10*np.pi, 10*np.pi, 50000)
    x_train = np.array(x_train).reshape((len(x_train), 1))
    y_train=np.sin(x_train)

    #part2: test data
    x_test = np.linspace(-16,16,100)
    x_test = np.array(x_test).reshape((len(x_test), 1))
    y_test = np.sin(x_test)
    print x_test, y_test
    return x_train, y_train, x_test, y_test


