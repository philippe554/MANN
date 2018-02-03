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

    def getCosSimSoftMaxExtra(self, k, M, b, extra):
        assert helper.check(k, [extra, self.memoryBitSize], self.batchSize)
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(b, [extra], self.batchSize)

        dot = tf.matmul(M, k, transpose_b=True)
        assert helper.check(dot, [self.memorylength, extra], self.batchSize)

        l1 = tf.sqrt(tf.reduce_sum(tf.pow(k, 2), axis=-1, keep_dims=True))
        l2 = tf.sqrt(tf.reduce_sum(tf.pow(M, 2), axis=-1))
        cosSim = tf.divide(tf.transpose(dot, perm=[0,2,1]), l1 * l2 + 0.001)
        assert helper.check(cosSim, [extra, self.memorylength], self.batchSize)

        result = tf.nn.softmax((tf.expand_dims(b, axis=-1) * cosSim) + 0.001)
        assert helper.check(result, [extra, self.memorylength], self.batchSize)

        return result

    def writeToMemory(self, memory, erase, add, w):
        m = tf.multiply(memory.getLast(), 1 - tf.matmul(tf.expand_dims(w, axis=-1),tf.expand_dims(erase, axis=-2)))
        memory.new(m + tf.matmul(tf.expand_dims(w, axis=-1),tf.expand_dims(add, axis=-2)))

        assert helper.check(memory.getLast(), [self.memorylength, self.memoryBitSize], self.batchSize)

    def readFromMemory(self, memory, w, multiple=None):
        if multiple is None:
            assert helper.check(w, [self.memorylength], self.batchSize)

            self.readList.append(tf.squeeze(tf.matmul(tf.expand_dims(w,axis=-2), memory.getLast()),axis=-2))
            assert helper.check(self.getLastRead(), [self.memoryBitSize], self.batchSize)
        else:
            M = memory.getLast()
            assert helper.check(w, [multiple, self.memorylength], self.batchSize)
            assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchSize)

            r = tf.matmul(w, M)
            assert helper.check(r, [multiple, self.memoryBitSize], self.batchSize)

            r = tf.reshape(r, [-1, multiple * self.memoryBitSize])
            assert helper.check(r, [multiple * self.memoryBitSize], self.batchSize)

            self.readList.append(r)
            


