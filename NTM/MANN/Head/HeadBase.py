import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class HeadBase:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        self.wList = []

        if self.mode not in ["Read", "Write"]:
            raise ValueError("Set a valid mode")

    def setup(self, batchSize, memoryBitSize, memoryLength):
        self.batchSize = batchSize
        self.memoryBitSize = memoryBitSize
        self.memorylength = memoryLength

        with tf.variable_scope(self.name):
            with tf.variable_scope("init"):
                self.setupStartVariables()

    def buildHead(self, memory, O):
        with tf.variable_scope(self.name):
            self.getW(O, memory)

            if self.mode == "Read":
                self.readList.append(tf.squeeze(tf.matmul(tf.expand_dims(self.getLastW(),axis=-2), memory.getLast()),axis=-2))
                assert helper.check(self.getLastRead(), [self.memoryBitSize], self.batchSize)

            else:
                erase = tf.sigmoid(helper.map("map_erase", O, self.memoryBitSize))
                add = tf.tanh(helper.map("map_add", O, self.memoryBitSize))

                m = tf.multiply(memory.getLast(), 1 - tf.matmul(tf.expand_dims(self.getLastW(), axis=-1),tf.expand_dims(erase, axis=-2)))
                memory.new(m + tf.matmul(tf.expand_dims(self.getLastW(), axis=-1),tf.expand_dims(add, axis=-2)))

                assert helper.check(memory.getLast(), [self.memorylength, self.memoryBitSize], self.batchSize)

    def getLastW(self):
        if len(self.wList) == 0:
            return self.wFirst
        else:
            return self.wList[-1]

    def getLastRead(self):
        if len(self.readList) == 0:
            return self.readFirst
        else:
            return self.readList[-1]

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