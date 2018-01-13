import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN import *

class LSTMCell(RNN):
    def __init__(self, name, stateSize):
        super().__init__(name)
        self.stateSize = stateSize

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.prevState = self.getTrainableConstant("startState", self.stateSize, tf.shape(input)[0])
                self.prevOutput = tf.tanh(self.prevState)

            cc = tf.concat([input,self.prevOutput], axis=-1)

            forgetGate = tf.sigmoid(helper.mapBatch("forgetGate", cc, self.stateSize))
            saveGate = tf.sigmoid(helper.mapBatch("saveGate", cc, self.stateSize))           
            outputGate = tf.sigmoid(helper.mapBatch("outputGate", cc, self.stateSize))
            update = tf.tanh(helper.mapBatch("update", cc, self.stateSize))           

            self.prevState = (self.prevState * forgetGate) + (saveGate * update)
            self.prevOutput = outputGate * tf.tanh(self.prevState)
            return self.prevOutput
