# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import h5py


def create_data(path_list, output_path):
    output_file = h5py.File(output_path,'w')
    for path in path_list:
        file = h5py.File(path, 'r')
        output_file.create_dataset('features', data = file['features'][:])
        output_file.create_dataset('doses', data = file['doses'][:])
        file.close()
    output_file.close()




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


if __name__ == '__main__':
    test_list = ["/home/dmp/ct/data/V16552.h5", "/home/dmp/ct/data/V16578.h5"]
    output_path = "/home/dmp/ct/data/test.h5"
    create_data(test_list, output_path)



