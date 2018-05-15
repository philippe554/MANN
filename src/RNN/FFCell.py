import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN.RNNBase import *

class FFCell(RNNBase):
    def __init__(self, name, outputSize, AF=tf.tanh):
        super().__init__(name)
        self.outputSize = outputSize
        self.AF = AF

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if self.AF == None:
                return helper.map("forward", input, self.outputSize)
            else:
                return self.AF(helper.map("forward", input, self.outputSize))


    def buildNoUnroll(self, input):
        with tf.variable_scope(self.name):
            inSize = input.get_shape()[-1]
            inLen = input.get_shape()[-2]

            i = tf.reshape(input, [-1, inLen * inSize])

            o = helper.map("forward", i, inLen * self.outputSize)

            if self.AF != None:
                o = self.AF(o)

            return tf.reshape(o, [-1, inLen, self.outputSize])



