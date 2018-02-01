import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from MANN.Head.HeadBase import *

class NTMHead(HeadBase):
    def setupStartVariables(self):
        self.wFirst = tf.sigmoid(helper.getTrainableConstant("wRead", self.memorylength, self.batchSize)) #Added sigmoid

        if self.mode == "Read":
            self.readFirst = helper.getTrainableConstant("PrevRead", self.memoryBitSize, self.batchSize)
            self.readList = []

    def buildRead(self, memory, O):
        assert helper.check(memory.getLast(), [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(self.getLastW(), [self.memorylength], self.batchSize)
        assert helper.check(O, [self.controllerSize], self.batchSize)

        self.wList.append(self.processHead(O, memory.getLast(), self.getLastW()))
        assert helper.check(self.getLastW(), [self.memorylength], self.batchSize)

        self.readList.append(tf.squeeze(tf.matmul(tf.expand_dims(self.getLastW(),axis=-2), memory.getLast()),axis=-2))
        assert helper.check(self.getLastRead(), [self.memoryBitSize], self.batchSize)

    def buildWrite(self, memory, O):
        assert helper.check(memory.getLast(), [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(self.getLastW(), [self.memorylength], self.batchSize)
        assert helper.check(O, [self.controllerSize], self.batchSize)

        self.wList.append(self.processHead(O, memory.getLast(), self.getLastW()))
        assert helper.check(self.getLastW(), [self.memorylength], self.batchSize)

        erase = tf.sigmoid(helper.map("map_erase", O, self.memoryBitSize))
        add = tf.tanh(helper.map("map_add", O, self.memoryBitSize))

        m = tf.multiply(memory.getLast(), 1 - tf.matmul(tf.expand_dims(self.getLastW(), axis=-1),tf.expand_dims(erase, axis=-2)))
        memory.new(m + tf.matmul(tf.expand_dims(self.getLastW(), axis=-1),tf.expand_dims(add, axis=-2)))

        assert helper.check(memory.getLast(), [self.memorylength, self.memoryBitSize], self.batchSize)

    def processHead(self, O, M, w_):
        k = tf.nn.softplus(helper.map("map_k", O, self.memoryBitSize))
        b = tf.nn.softplus(helper.map("map_b", O, 1))
        g = tf.sigmoid(helper.map("map_g", O, 1))
        s = tf.nn.softmax(tf.sigmoid(helper.map("map_s", O, 5))) #Added sigmoid
        y = tf.nn.softplus(helper.map("map_y", O, 1)) + 1

        wc = self.getCosSimSoftMax(k, M, b)
        wg = self.getWg(wc, g, w_)
        wm = self.getWmFast(wg, s)
        w = self.getW(wm, y)
        return w

    def getWg(self, wc, g, w_):
        assert helper.check(wc, [self.memorylength], self.batchSize)
        assert helper.check(g, [1], self.batchSize)
        assert helper.check(w_, [self.memorylength], self.batchSize)

        result = g*wc + (1-g)*w_

        assert helper.check(result, [self.memorylength], self.batchSize)
        return result

    def getWm(self, wg, s):
        assert helper.check(wg, [self.memorylength], self.batchSize)
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

        assert helper.check(result, [self.memorylength], self.batchSize)
        return result

    def getWmFast(self, wg, s):
        #Amount of concat operations is proportional to the shift size, instead of memory length (Only significantly faster on a big memory)
        assert helper.check(wg, [self.memorylength], self.batchSize)
        assert helper.check(s, [5], self.batchSize)

        w1 = tf.concat([wg[:,-2:], wg[:,:-2]], axis=-1)
        w2 = tf.concat([wg[:,-1:], wg[:,:-1]], axis=-1)
        w4 = tf.concat([wg[:,1:], wg[:,:1]], axis=-1)
        w5 = tf.concat([wg[:,2:], wg[:,:2]], axis=-1)

        w = tf.stack([w1,w2,wg,w4,w5], axis=-1)
        result = tf.squeeze(tf.matmul(w, tf.expand_dims(s, axis=-1)), axis=-1)

        assert helper.check(result, [self.memorylength], self.batchSize)
        return result

    def getW(self, wm, y):
        assert helper.check(wm, [self.memorylength], self.batchSize)
        assert helper.check(y, [1], self.batchSize)

        #wm can be negtive -> power will push it into the complex domain
        pow = tf.pow(wm, y)
        result =  pow / (tf.reduce_sum(pow, axis=-1, keep_dims=True)+0.001)

        assert helper.check(result, [self.memorylength], self.batchSize)
        return result