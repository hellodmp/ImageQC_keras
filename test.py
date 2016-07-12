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
    norm_list = [5080, 5080, 5080, 5080, 5080, 5080]
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

'''
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = preprocess.create_data()
    y_predict = image_train(x_train, y_train, x_test)

    print len(y_test), len(y_predict)
    errors = caculate_erros(x_test, y_test, y_predict)
    for i in range(0, len(errors)):
        print errors[i]
'''


if __name__ == '__main__':
   test("model.json", "weights/weights-best.h5", "/home/dmp/ct/test_outPTV/")
