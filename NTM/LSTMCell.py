import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class LSTMCell:
    def __init__(self, name, inputSize, stateSize):
        self.name = name;
        self.inputSize = inputSize
        self.stateSize = stateSize

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.setupStart(input)

            cc = tf.concat([input,self.prevOutput], axis=-1)

            forgetGate = tf.sigmoid(helper.mapBatch("forgetGate", cc, self.stateSize))
            saveGate = tf.sigmoid(helper.mapBatch("saveGate", cc, self.stateSize))           
            outputGate = tf.sigmoid(helper.mapBatch("outputGate", cc, self.stateSize))
            update = tf.tanh(helper.mapBatch("update", cc, self.stateSize))           

            self.prevState = (self.prevState * forgetGate) + (saveGate * update)
            self.prevOutput = outputGate * tf.tanh(self.prevState)
            return self.prevOutput

    def setupStart(self, input):
        self.prevState = tf.get_variable("startState", initializer=tf.random_normal([self.stateSize]))

        if(len(input.get_shape())==2):
            self.prevState = tf.reshape(tf.tile(self.prevState, [tf.shape(input)[0]]), [tf.shape(input)[0]] + [self.stateSize])

        self.prevOutput = tf.tanh(self.prevState)
