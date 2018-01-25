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

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.setup(input)

            LSTMOuput = self.LSTM.buildTimeLayer(tf.concat([input, self.prevRead], axis=-1), first)

            self.prevRead, self.wRead = self.read(self.M, self.wRead, LSTMOuput, 0)
            self.M, self.wWrite = self.write(self.M, self.wRead, LSTMOuput, 0)

            #Just used for plotting
            w = tf.concat([self.wWrite, self.wRead], axis=-1)

            return helper.map("output", LSTMOuput, self.outputSize), w

    def setup(self, firstInput):
        if(len(firstInput.get_shape())==2):
            batchSize = tf.shape(firstInput)[0]
            self.batchCheck = True
        else:
            batchSize = None
            self.batchCheck = False

        self.LSTM = LSTMCell("controller", self.controllerSize)

        with tf.variable_scope("init"):
            self.prevRead = self.getTrainableConstant("PrevRead", self.memoryBitSize, batchSize)
            self.M = tf.reshape(self.getTrainableConstant("M", self.memorylength * self.memoryBitSize, batchSize), [-1, self.memorylength, self.memoryBitSize])
            self.wRead = tf.sigmoid(self.getTrainableConstant("wRead", self.memorylength, batchSize)) #Added sigmoid
            self.wWrite = tf.sigmoid(self.getTrainableConstant("wWrite", self.memorylength, batchSize)) #Added sigmoid

    def processHead(self, O, M, w_):
        k = tf.nn.softplus(helper.map("map_k", O, self.memoryBitSize))
        b = tf.nn.softplus(helper.map("map_b", O, 1))
        g = tf.sigmoid(helper.map("map_g", O, 1))
        s = tf.nn.softmax(tf.sigmoid(helper.map("map_s", O, 5))) #Added sigmoid
        y = tf.nn.softplus(helper.map("map_y", O, 1)) + 1

        wc = self.getWc(k, M, b)
        wg = self.getWg(wc, g, w_)
        wm = self.getWmFast(wg, s)
        w = self.getW(wm, y)
        return w

    def getWc(self, k, M, b):
        assert helper.check(k, [self.memoryBitSize], self.batchCheck)
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchCheck)
        assert helper.check(b, [1], self.batchCheck)

        dot = tf.squeeze(tf.matmul(M, tf.expand_dims(k, axis=-1)), axis=-1)
        l1 = tf.sqrt(tf.reduce_sum(tf.pow(k, 2), axis=-1, keep_dims=True))
        l2 = tf.sqrt(tf.reduce_sum(tf.pow(M, 2), axis=-1))
        cosSim = tf.divide(dot, l1 * l2 + 0.001)    
        result = tf.nn.softmax((b * cosSim) + 0.001)

        assert helper.check(result, [self.memorylength], self.batchCheck)
        return result

    def getWg(self, wc, g, w_):
        assert helper.check(wc, [self.memorylength], self.batchCheck)
        assert helper.check(g, [1], self.batchCheck)
        assert helper.check(w_, [self.memorylength], self.batchCheck)

        result = g*wc + (1-g)*w_

        assert helper.check(result, [self.memorylength], self.batchCheck)
        return result

    def getWm(self, wg, s):
        assert helper.check(wg, [self.memorylength], self.batchCheck)
        assert helper.check(s, [5], self.batchCheck)

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

        result = tf.stack([indices(i) for i in range(0,size)], axis=-1)

        assert helper.check(result, [self.memorylength], self.batchCheck)
        return result

    def getWmFast(self, wg, s):
        #Amount of concat operations is proportional to the shift size, instead of memory length (Only significantly faster on a big memory)
        assert helper.check(wg, [self.memorylength], self.batchCheck)
        assert helper.check(s, [5], self.batchCheck)

        w1 = tf.concat([wg[:,-2:], wg[:,:-2]], axis=-1)
        w2 = tf.concat([wg[:,-1:], wg[:,:-1]], axis=-1)
        w4 = tf.concat([wg[:,1:], wg[:,:1]], axis=-1)
        w5 = tf.concat([wg[:,2:], wg[:,:2]], axis=-1)

        w = tf.stack([w1,w2,wg,w4,w5], axis=-1)
        result = tf.squeeze(tf.matmul(w, tf.expand_dims(s, axis=-1)), axis=-1)

        assert helper.check(result, [self.memorylength], self.batchCheck)
        return result

    def getW(self, wm, y):
        assert helper.check(wm, [self.memorylength], self.batchCheck)
        assert helper.check(y, [1], self.batchCheck)

        #wm can be negtive -> power will push it into the complex domain
        pow = tf.pow(wm, y)
        result =  pow / (tf.reduce_sum(pow, axis=-1, keep_dims=True)+0.001)

        assert helper.check(result, [self.memorylength], self.batchCheck)
        return result

    def read(self, M, w, O, i):
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchCheck)
        assert helper.check(w, [self.memorylength], self.batchCheck)
        assert helper.check(O, [self.controllerSize], self.batchCheck)

        with tf.variable_scope("read"+str(i), reuse=True):
            w = self.processHead(O, M, w)
            result = tf.squeeze(tf.matmul(tf.expand_dims(w,axis=-2),M),axis=-2)

        assert helper.check(result, [self.memoryBitSize], self.batchCheck)
        assert helper.check(w, [self.memorylength], self.batchCheck)
        return result, w

    def write(self, M, w, O, i):
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchCheck)
        assert helper.check(w, [self.memorylength], self.batchCheck)
        assert helper.check(O, [self.controllerSize], self.batchCheck)

        with tf.variable_scope("write"+str(i), reuse=True):
            w = self.processHead(O, M, w)
            erase = tf.sigmoid(helper.map("map_erase", O, self.memoryBitSize))
            add = tf.tanh(helper.map("map_add", O, self.memoryBitSize))

            M = tf.multiply(M, 1 - tf.matmul(tf.expand_dims(w, axis=-1),tf.expand_dims(erase, axis=-2)))
            result = M + tf.matmul(tf.expand_dims(w, axis=-1),tf.expand_dims(add, axis=-2))

        assert helper.check(result, [self.memorylength, self.memoryBitSize], self.batchCheck)
        assert helper.check(w, [self.memorylength], self.batchCheck)
        return result, w