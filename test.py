# -*- coding: utf-8 -*-

from model import Model
import matplotlib.pyplot as plt
import preprocess
import numpy as np


def train(x_train, y_train, x_test, y_test):
    netModel = Model().create_model()
    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    print "STARTING TRAINING"
    print len(x_train)
    # netModel.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)
    train_data_generator = preprocess.batch_generator1(x_train, y_train, 10)
    netModel.fit_generator(
        generator=train_data_generator,
        samples_per_epoch=len(x_train),
        nb_epoch=50)

    out = netModel.predict(x_test, batch_size=10)
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test,'r')
    # print y_test, out
    ax.plot(x_test, out, 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def image_train(x_train, y_train, x_test):
    netModel = Model().image_model()
    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    print "STARTING TRAINING"
    netModel.fit(x_train, y_train, nb_epoch=10, batch_size=20, verbose=0)
    # model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)
    y_predict = netModel.predict(x_test, batch_size=10)
    return y_predict


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = preprocess.create_data()
    y_predict = image_train(x_train, y_train, x_test)
    file = open('/home/dmp/ct/data/outptv/result.txt')
    for i in range(0,len(y_predict)):
        file.writelines(str(y_predict[i])+":"+str(y_test[i]))
    file.close()

