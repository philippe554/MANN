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
            return self.AF(helper.map("forward", input, self.outputSize))

