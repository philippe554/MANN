import tensorflow as tf
import pandas as pd
import numpy as np

import helper

#Abstract class, shows a template for reading heads. Overwrite the not implemented functions

class MANNHeadPrototype:
    def __init__(self, name):
        self.name = name;

    def setup(self, batchSize, memoryBitSize, memoryLength, controllerSize):
        self.batchSize = batchSize
        self.memoryBitSize = memoryBitSize
        self.memorylength = memoryLength
        self.controllerSize = controllerSize

        self.setupStartVariables()

    def setupStartVariables(self):
        raise NotImplementedError

    def buildRead(self, M, O):
        raise NotImplementedError

    def buildWrite(self, M, O):
        raise NotImplementedError

    def getCosSimSoftMax(self, k, M, b):
        assert helper.check(k, [self.memoryBitSize], self.batchSize)
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(b, [1], self.batchSize)

        dot = tf.squeeze(tf.matmul(M, tf.expand_dims(k, axis=-1)), axis=-1)
        l1 = tf.sqrt(tf.reduce_sum(tf.pow(k, 2), axis=-1, keep_dims=True))
        l2 = tf.sqrt(tf.reduce_sum(tf.pow(M, 2), axis=-1))
        cosSim = tf.divide(dot, l1 * l2 + 0.001)

        result = tf.nn.softmax((b * cosSim) + 0.001)

        assert helper.check(result, [self.memorylength], self.batchSize)
        return result