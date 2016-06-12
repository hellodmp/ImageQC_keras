# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU


class Model:
    def __init__(self):
        pass

    def create_model(self):
        model = Sequential()
        model.add(Dense(500, init='uniform', input_dim=1))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        return model
