import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from LSTMCell import *
from RNN import *

class NTMCell(RNN):
    def __init__(self, name, outputSize, memoryBitSize, memoryLength, controllerSize):
        super().__init__(name)
        self.outputSize = outputSize
        self.memoryBitSize = memoryBitSize
        self.memorylength = memoryLength
        self.controllerSize = controllerSize                    

    def buildTimeLayer(self, input, batchSize=None, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.LSTM = LSTMCell("controller", self.controllerSize)

                self.prevRead = self.getTrainableConstant("ssPrevRead", self.memoryBitSize, batchSize)
                self.M = tf.reshape(self.getTrainableConstant("ssM", self.memorylength * self.memoryBitSize, batchSize), [-1, self.memorylength, self.memoryBitSize])
                self.wRead = self.getTrainableConstant("sswRead", self.memorylength, batchSize)
                self.wWrite = self.getTrainableConstant("sswWrite", self.memorylength, batchSize)

            I = tf.tanh(helper.map("prep", tf.concat([input, self.prevRead], axis=-1), self.controllerSize))
            LSTMOuput = self.LSTM.buildTimeLayer(I, batchSize, first)

            self.wRead = self.processHead(LSTMOuput, self.M, self.wRead, "read")
            self.wWrite = self.processHead(LSTMOuput, self.M, self.wWrite, "write")

            self.prevRead = self.read(self.M, self.wRead)

            erase = tf.sigmoid(helper.map("map_erase", LSTMOuput, self.memoryBitSize))
            add = tf.tanh(helper.map("map_add", LSTMOuput, self.memoryBitSize))
            self.M = self.write(erase, add, self.M, self.wWrite)

            outputGate = tf.sigmoid(helper.map("outputGate", LSTMOuput, self.memoryBitSize))
            return helper.map("output", outputGate * self.prevRead, self.outputSize) #no final sigmoid/tanh

    def processHead(self, O, M, w_, name):
        with tf.variable_scope(name, reuse=True):
            k = tf.nn.softplus(helper.map("map_k", O, self.memoryBitSize))
            b = tf.nn.softplus(helper.map("map_b", O, 1))
            g = tf.sigmoid(helper.map("map_g", O, 1))
            s = tf.nn.softmax(helper.map("map_s", O, 5))
            y = tf.nn.softplus(helper.map("map_y", O, 1)) + 1

            wc = self.getWc(k, M, b)
            wg = self.getWg(wc, g, w_)
            wm = self.getWm(wg, s)
            w = self.getW(wm, y)
            return w

    def getWc(self, k, M, b):
        assert helper.check(k, [None, self.memoryBitSize])
        assert helper.check(M, [None, self.memorylength, self.memoryBitSize])
        assert helper.check(b, [None, 1])

        dot = tf.squeeze(tf.matmul(M, tf.expand_dims(k, axis=-1)), axis=-1)
        l1 = tf.norm(k, axis=-1)
        l2 = tf.norm(M, axis=-1)
        cosSim = tf.divide(dot, l1 * l2 + 0.001)
        result = tf.nn.softmax((b * cosSim) + 0.001)

        assert helper.check(result, [None, self.memorylength])
        return result

    def getWg(self, wc, g, w_):
        assert helper.check(wc, [None, self.memorylength])
        assert helper.check(g, [None, 1])
        assert helper.check(w_, [None, self.memorylength])

        result = g*wc + (1-g)*w_

        assert helper.check(result, [None, self.memorylength])
        return result

    def getWm(self, wg, s):
        assert helper.check(wg, [None, self.memorylength])
        assert helper.check(s, [None, 5])

        size = self.memorylength
        shiftSize = 2

        def shift(i):
            if(i<0):
                return size+i
            if(i>=size):
                return i-size
            return i

        def indices(i):
            indices = [shift(i+j) for j in range(shiftSize, -shiftSize-1, -1)]
            return tf.reduce_sum(tf.gather(wg, indices, axis=-1) * s, axis=-1)

        print(indices(0).get_shape())

        result = tf.concat([indices(i) for i in range(0,size)], axis=-1)

        print(result.get_shape())

        assert helper.check(result, [None, self.memorylength])
        return result

    def getW(self, wm,y):
        assert helper.check(wm, [None, self.memorylength])
        assert helper.check(y, [None, 1])

        pow = tf.pow(wm, y)
        result =  pow / (tf.reduce_sum(pow, axis=-1)+0.001)

        assert helper.check(result, [None, self.memorylength])
        return result

    def read(self, M, w):
        assert helper.check(M, [None, self.memorylength, self.memoryBitSize])
        assert helper.check(w, [None, self.memorylength])

        result = tf.squeeze(tf.matmul(tf.expand_dims(w,axis=-2),M),axis=-2)

        assert helper.check(result, [None, self.memoryBitSize])
        return result

    def write(self, erase, add, M, w):
        assert helper.check(erase, [None, self.memoryBitSize])
        assert helper.check(add, [None, self.memoryBitSize])
        assert helper.check(M, [None, self.memorylength, self.memoryBitSize])
        assert helper.check(w, [None, self.memorylength])

        M = tf.multiply(M, 1 - tf.matmul(tf.expand_dims(w, axis=-1),tf.expand_dims(erase, axis=-2)))
        result = M + tf.matmul(tf.expand_dims(w, axis=-1),tf.expand_dims(add, axis=-2))

        print(result.get_shape())

        assert helper.check(result, [None, self.memorylength, self.memoryBitSize])
        return result