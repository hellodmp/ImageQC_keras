# -*- coding: utf-8 -*-

from model import Model
import matplotlib.pyplot as plt
import preprocess
import numpy as np


def batch_generator1(x_dataset, y_dataset, batch_size):
    Xbatch = np.zeros(batch_size)
    Ybatch = np.zeros(batch_size)
    batch_idx = 0
    for i in range(0,len(feature_dataset)):
        Xbatch[batch_idx] = x_dataset[i]
        Ybatch[batch_idx] = y_dataset[i]
        batch_idx += 1
        if batch_idx == batch_size:
            batch_idx = 0
            yield (Xbatch, Ybatch)

def train(x_train, y_train, x_test, y_test):
    netModel = Model().create_model()
    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    print "STARTING TRAINING"
    #netModel.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)
    train_data_generator = batch_generator1(x_train, y_train, 10)
    model.fit_generator(
		generator=train_data_generator,
		samples_per_epoch=10,
		nb_epoch=64)
    out = netModel.predict(x_test, batch_size=10)
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test,'r')
    print y_test, out
    ax.plot(x_test, out, 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def batch_generator(feature_dataset, dose_dataset, batch_size):
    Xbatch = np.zeros((batch_size, 9))
    Ybatch = np.zeros(batch_size)
    batch_idx = 0
    for i in range(0,len(feature_dataset)):
        Xbatch[batch_idx] = feature_dataset[i]
        Ybatch[batch_idx] = dose_dataset[i]
        batch_idx += 1
        if batch_idx == batch_size:
            batch_idx = 0
            yield (Xbatch, Ybatch)



def image_train(x_train, y_train, x_test, y_test):
    netModel = Model().image_model()
    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    print "STARTING TRAINING"
    netModel.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)
    model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = preprocess.create_data()
    train(x_train, y_train, x_test, y_test)
    # x_train, y_train, x_test, y_test = preprocess.get_data('./data/rt_data.h5', './data/rt_data.h5')
    # (x, y) = batch_generator(x_train, y_train,10)
    # print (x,y)



