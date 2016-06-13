# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np


def create_data():
    x_train = np.linspace(-2*np.pi, 2*np.pi, 1000)
    x_train = np.array(x_train).reshape((len(x_train), 1))
    y_train=np.sin(x_train)+2*x_train

    #part2: test data
    x_test = np.linspace(-20,20,100)
    x_test = np.array(x_test).reshape((len(x_test), 1))
    y_test = np.sin(x_test)+2*x_test
    print x_test, y_test
    return x_train, y_train, x_test, y_test


