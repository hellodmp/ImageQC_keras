# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU


class Model:
    def __init__(self):
        pass

    def image_model1(self):
        model = Sequential()
        model.add(Dense(output_dim=64, init='uniform', W_regularizer='L2', b_regularizer='L2', input_dim=9))
        model.add(Activation('relu'))
        model.add(Dense(output_dim=10))
        model.add(Activation('relu'))
        model.add(Dense(output_dim=1))
        return model

    def image_model(self):
        model = Sequential()
        model.add(Dense(output_dim=13, init='uniform', input_dim=10))
        #model.add(Dense(output_dim=30, init='uniform', input_dim=10))
        #model.add(Dropout(0.3))
        model.add(Activation('tanh'))
        model.add(Dense(output_dim=1))
        return model
