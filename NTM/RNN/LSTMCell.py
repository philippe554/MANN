import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN.RNNBase import *

class LSTMCell(RNNBase):
    def __init__(self, name, stateSize):
        super().__init__(name)
        self.stateSize = stateSize

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                if(len(input.get_shape())==2):
                    batchSize = tf.shape(input)[0]
                else:
                    batchSize = None

                self.prevState = helper.getTrainableConstant("startState", self.stateSize, batchSize)
                self.prevOutput = tf.tanh(self.prevState)

            cc = tf.concat([input,self.prevOutput], axis=-1)

            forgetGate = tf.sigmoid(helper.map("forgetGate", cc, self.stateSize))
            saveGate = tf.sigmoid(helper.map("saveGate", cc, self.stateSize))           
            outputGate = tf.sigmoid(helper.map("outputGate", cc, self.stateSize))
            update = tf.tanh(helper.map("update", cc, self.stateSize))           

            self.prevState = (self.prevState * forgetGate) + (saveGate * update)
            self.prevOutput = outputGate * tf.tanh(self.prevState)
            return self.prevOutput
