# -*- coding: utf-8 -*-
from keras.models import model_from_json
from keras.optimizers import SGD

from model import Model
import matplotlib.pyplot as plt
import preprocess
import numpy as np


def image_train(x_train, y_train, x_test):
    netModel = Model().image_model()
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #netModel.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])

    print "STARTING TRAINING"
    netModel.fit(x_train, y_train, nb_epoch=1, batch_size=10, shuffle=True, verbose=1)
    # model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)
    y_predict = netModel.predict(x_test, batch_size=10)
    save_model(netModel, "model.json", "weights.h5")
    return y_predict

def image_train(x_train, y_train, x_test, model_path="model.json", weight_path="weights.h5"):
    netModel = Model().image_model()
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #netModel.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])

    print "STARTING TRAINING"
    netModel.fit(x_train, y_train, nb_epoch=1, batch_size=10, shuffle=True, verbose=1)
    # model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)
    y_predict = netModel.predict(x_test, batch_size=10)
    save_model(netModel, model_path, weight_path)
    return y_predict

def save_model(model, model_path, weight_path):
    json_string = model.to_json()
    open(model_path,'w').write(json_string)
    model.save_weights(weight_path)


def caculate_erros(x_test, y_test, y_predict):
    errors = np.zeros(13)
    counts = np.zeros(13)
    for i in range(0,len(y_predict)):
        r = abs(x_test[i][1])
        index = int(r/2.5 + 0.5)
        if index > 12:
            index = 12
        errors[index] += abs(y_test[index]-y_predict[index])
        counts[index] += 1

    for i in range(0, len(errors)):
        if(counts[i] > 0):
            errors[i] = errors[i]/counts[i]
    return errors


def load_model(model_path, weight_path):
    model = model_from_json(open(model_path).read())
    model.load_weights(weight_path)
    return model

def test(model_path, weight_path, dir):
    path_list = ["V19627_outptv.h5", "V19737_outptv.h5","V19799_outptv.h5","V19868_outptv.h5","V20101_outptv.h5","V20115_outptv.h5"]
    norm_list = [5080]*len(path_list)
    model = load_model(model_path, weight_path)
    x_test_list, y_test_list = preprocess.read_test_data(dir, path_list, norm_list)
    for i in range(0,len(x_test_list)):
        x_test = x_test_list[i]
        y_test = y_test_list[i]
        y_predict = model.predict(x_test, batch_size=10)
        errors = caculate_erros(x_test, y_test, y_predict)
        for i in range(0, len(errors)):
            print errors[i]
        print ''


def choose_erros(x_test, y_test, y_predict):
    errors = []
    count = 0
    for i in range(0,len(y_predict)):
        if x_test[i][3] < 0 or x_test[i][4] < 0 or x_test[i][5] < 0:
            errors.append(y_test[i]-y_predict[i])
    for i in range(0, len(errors)):
        count += errors[i]
    return count/len(errors)

def choose(model_path, weight_path, dir):
    '''
    path_list = ["V19627_outptv.h5", "V19737_outptv.h5","V19799_outptv.h5","V19868_outptv.h5","V20101_outptv.h5",
                 "V20115_outptv.h5", "V20116_outptv.h5", "V20164_outptv.h5", "V20176_outptv.h5", "V20243_outptv.h5",
                 "V20247_outptv.h5"]
    norm_list = [5080, 5080, 5080, 5080, 5080, 5080,5080, 5080, 5080, 5080, 5080]
    '''
    path_list = ["V13244.h5", "V13265.h5", "V13275.h5","V13285.h5","V13296.h5",
              "V13317.h5", "V13346.h5", "V16531.h5","V16552.h5", "V16578.h5"]
    norm_list = [5080, 5080, 5080, 5080, 5080, 5080, 5080, 5080, 5080, 5080]
    model = load_model(model_path, weight_path)
    x_test_list, y_test_list = preprocess.read_test_data(dir, path_list, norm_list)
    for i in range(0,len(x_test_list)):
        x_test = x_test_list[i]
        y_test = y_test_list[i]
        y_predict = model.predict(x_test, batch_size=10)
        error = choose_erros(x_test, y_test, y_predict)
        print path_list[i], error

'''
if __name__ == '__main__':
    #dir = "/home/dmp/ct/data/outptv/"
    dir = "/home/dmp/ct/data/inptv/"
    path_list = ["V13244.h5", "V13265.h5", "V13275.h5","V13285.h5","V13296.h5",
                  "V13317.h5", "V13346.h5", "V16531.h5","V16552.h5", "V16578.h5"]
    x_train, y_train, x_test, y_test = preprocess.create_data(dir, path_list)
    y_predict = image_train(x_train, y_train, x_test)

    print len(y_test), len(y_predict)
    errors = caculate_erros(x_test, y_test, y_predict)
    for i in range(0, len(errors)):
        print errors[i]
'''


if __name__ == '__main__':
    dir = "/home/dmp/ct/data/refine/"
    #dir = "/home/dmp/ct/data/inptv/"
    path_list = ["V13244.h5", "V13285.h5", "V13317.h5", "V13346.h5", "V16531.h5","V16552.h5",
                 "V19799_outptv.h5","V19868_outptv.h5","V20101_outptv.h5","V20243_outptv.h5"]
    x_train, y_train, x_test, y_test = preprocess.create_data(dir, path_list)
    for i in range(0,20):
        print "i=",i
        y_predict = image_train(x_train, y_train, x_test, "model"+str(i)+".json", "weights"+str(i)+".h5")
        errors = caculate_erros(x_test, y_test, y_predict)
        for i in range(0, len(errors)):
            print errors[i]


'''
if __name__ == '__main__':
   test("model.json", "weights/weights-best.h5", "/home/dmp/ct/data/test_outPTV/")
   #choose("model.json", "weights/weights-best.h5", "/home/dmp/ct/data/outptv/")
'''
