# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU


class Model:
    def __init__(self):
        pass


    def image_model(self):
        model = Sequential()
        model.add(Dense(50, init='uniform', input_dim=9))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dense(1))
        return model
