# -*- coding: utf-8 -*-

import numpy as np
import h5py

def read_data(dir, path_list, norm_list):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(0, len(path_list)-2):
        dataset = h5py.File(dir + path_list[i], 'r')
        x_train.extend(dataset['features'][:])
        y_train.extend(dataset['doses'][:]/norm_list[i])
        dataset.close()

    for i in range(len(path_list)-2, len(path_list)):
        dataset = h5py.File(dir + path_list[i], 'r')
        x_test.extend(dataset['features'][:])
        y_test.extend(dataset['doses'][:]/norm_list[i])
        dataset.close()
    return x_train, y_train, x_test, y_test

def read_test_data(dir, path_list, norm_list):
    x_test_list = []
    y_test_list = []
    for i in range(0, len(path_list)):
        x_data = []
        y_data = []
        dataset = h5py.File(dir + path_list[i], 'r')
        x_data.extend(dataset['features'][:])
        y_data.extend(dataset['doses'][:]/norm_list[i])
        dataset.close()
        x_test = np.zeros((len(x_data),len(x_data[0])))
        y_test = np.zeros(len(y_data))
        id = 0
        for i in range(0,len(y_data)):
            x_test[id] = x_data[i]
            y_test[id] = y_data[i]
            id += 1
        x_test_list.append(x_test)
        y_test_list.append(y_test)
    return x_test_list, y_test_list

def create_data():
    #dir = "/home/dmp/ct/data/outptv/"
    dir = "/home/dmp/ct/data/inptv/"
    path_list = ["V13244.h5", "V13265.h5", "V13275.h5","V13285.h5","V13296.h5",
                  "V13317.h5", "V13346.h5", "V16531.h5","V16552.h5", "V16578.h5"]

    norm_list = [5080, 5080, 5080, 5080, 5080, 5080, 5080, 5080, 5080, 5080]

    x_data1, y_data1, x_data2, y_data2, = read_data(dir, path_list, norm_list)
    x_train = np.zeros((len(x_data1), len(x_data1[0])))
    y_train = np.zeros(len(y_data1))
    id = 0
    for i in range(0,len(x_data1)):
        x_train[id] = x_data1[i]
        y_train[id] = y_data1[i]
        id += 1

    x_test = np.zeros((len(x_data2),len(x_data2[0])))
    y_test = np.zeros(len(y_data2))
    id = 0
    for i in range(0,len(y_data2)):
        x_test[id] = x_data2[i]
        y_test[id] = y_data2[i]
        id += 1
    return x_train, y_train, x_test, y_test

'''
def norm():
    dir = "/home/dmp/ct/data/inptv/"
    path_list = ["V13244.h5", "V13265.h5", "V13275.h5","V13285.h5",
                  "V13296.h5", "V13317.h5", "V13346.h5", "V16531.h5","V16552.h5", "V16578.h5"]
    dose_list = []
    for path in path_list:
        dataset = h5py.File(dir + path, 'r')
        print path
        doses = dataset['doses'][:]
        total = 0
        count = 0
        for dose in doses:
            total += dose[0]
            count += 1
        dataset.close()
        dose_list.append(total/count)
    return dose_list

if __name__ == '__main2__':
    dose_list = norm()
    print dose_list


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_data()
    print len(x_train), len(y_train), len(x_test), len(y_test)
    for i in range(0,10):
        print x_train[i], y_train[i]
    for i in range(0,10):
        print x_test[i], y_test[i]

'''


