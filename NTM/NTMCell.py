import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from LSTMCell import *
from RNN import *

class NTMCell(RNN):
    def __init__(self, name, outputSize, memoryBitSize, memoryLength, controllerSize, head):
        super().__init__(name)
        self.outputSize = outputSize
        self.memoryBitSize = memoryBitSize
        self.memoryLength = memoryLength
        self.controllerSize = controllerSize
        self.head = head

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.setup(input)
                self.head.setup(self.batchSize, self.memoryBitSize, self.memoryLength, self.controllerSize)

            LSTMOuput = self.LSTM.buildTimeLayer(tf.concat([input, self.prevRead], axis=-1), first)

            self.prevRead, self.wRead = self.head.buildRead(self.M, LSTMOuput)
            self.M, self.wWrite = self.head.buildWrite(self.M, LSTMOuput)

            #Just used for plotting
            w = tf.concat([self.wWrite, self.wRead], axis=-1)

            return helper.map("output", LSTMOuput, self.outputSize), w

    def setup(self, firstInput):
        if(len(firstInput.get_shape())==2):
            self.batchSize = tf.shape(firstInput)[0]
        else:
            self.batchSize = None

        self.LSTM = LSTMCell("controller", self.controllerSize)

        with tf.variable_scope("init"):
            self.prevRead = helper.getTrainableConstant("PrevRead", self.memoryBitSize, self.batchSize)
            self.M = tf.reshape(helper.getTrainableConstant("M", self.memoryLength * self.memoryBitSize, self.batchSize), [-1, self.memoryLength, self.memoryBitSize])
