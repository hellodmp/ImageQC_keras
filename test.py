# -*- coding: utf-8 -*-

from model import Model
import matplotlib.pyplot as plt
import preprocess


def train(x_train, y_train, x_test, y_test):

    netModel = Model().create_model()
    netModel.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
    print "STARTING TRAINING"
    netModel.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)
    out = netModel.predict(x_test, batch_size=10)
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test,'r')
    print y_test, out
    ax.plot(x_test, out, 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = preprocess.create_data()
    train(x_train, y_train, x_test, y_test)


