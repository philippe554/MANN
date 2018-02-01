import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN.RNNBase import *

class GRUCell(RNNBase):
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

                self.output = helper.getTrainableConstant("startOuput", self.stateSize, batchSize)

            cc = tf.concat([input,self.output], axis=-1)

            z = tf.sigmoid(helper.map("updateGate", cc, self.stateSize))
            r = tf.sigmoid(helper.map("resetGate", cc, self.stateSize))
            h = tf.tanh(helper.map("outputGate", tf.concat([input,r*self.output], axis=-1), self.stateSize))
            self.output = (1-z)*self.output + z*h
            
            return self.output

