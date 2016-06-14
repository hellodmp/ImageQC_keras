# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import h5py


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


def get_data(train_path, test_path):
    train_file = h5py.File(train_path)
    x_train = train_file['features']
    y_train = train_file['doses']

    test_file = h5py.File(test_path)
    x_test = test_file['features']
    y_test = test_file['doses']

    return x_train, y_train, x_test, y_test



