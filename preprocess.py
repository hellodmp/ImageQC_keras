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
    # print x_test, y_test
    return x_train, y_train, x_test, y_test


def batch_generator1(x_dataset, y_dataset, batch_size, ):
    Xbatch = np.zeros(batch_size)
    Ybatch = np.zeros(batch_size)
    batch_idx = 0
    while True:
        for i in range(0, len(x_dataset)):
            Xbatch[batch_idx] = x_dataset[i]
            Ybatch[batch_idx] = y_dataset[i]
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch)


def get_data(train_path, test_path):
    train_file = h5py.File(train_path)
    x_train = train_file['features']
    y_train = train_file['doses']

    test_file = h5py.File(test_path)
    x_test = test_file['features']
    y_test = test_file['doses']
    return x_train, y_train, x_test, y_test


def batch_generator(dataset_path, batch_size):
    dataset = h5py.File(dataset_path)
    features = dataset['features']
    doses = dataset['doses']
    feature_size = len(features[0])
    Xbatch = np.zeros((batch_size, feature_size))
    Ybatch = np.zeros(batch_size)
    batch_idx = 0
    for i in range(0,len(features)):
        Xbatch[batch_idx] = features[i]
        Ybatch[batch_idx] = doses[i]
        batch_idx += 1
        if batch_idx == batch_size:
            batch_idx = 0
            yield (Xbatch, Ybatch)



