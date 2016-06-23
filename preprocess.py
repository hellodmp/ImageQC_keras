# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import h5py


def read_data(dir, path_list):
    features = []
    doses = []
    for path in path_list:
        dataset = h5py.File(dir + path, 'r')
        features.extend(dataset['features'][:])
        doses.extend(dataset['features'][:])
        dataset.close()
    return features, doses


def create_data():
    dir = "/home/dmp/ct/data/outptv/"
    train_list = ["V13244.h5", "V13265.h5", "V13275.h5","V13285.h5",
                  "V13296.h5", "V13317.h5", "V13346.h5", "V16531.h5"]
    test_list = ["V16552.h5", "V16578.h5"]
    x_train, y_train = read_data(dir, train_list)
    x_test, y_test = read_data(dir, test_list)
    return x_train, y_train, x_test, y_test


'''
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
'''

if __name__ == '__main__':
    dir = "/home/dmp/ct/data/outptv/"
    train_list = ["V13244.h5", "V13265.h5", "V13275.h5","V13285.h5",
                  "V13296.h5", "V13317.h5", "V13346.h5", "V16531.h5"]
    test_list = ["V16552.h5", "V16578.h5"]
    x_train, y_train = read_data(dir, train_list)
    x_test, y_test = read_data(dir, test_list)
    print len(x_train), len(x_test)



