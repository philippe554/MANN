import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from MANN.Head.HeadBase import *

class NTMHead(HeadBase):
    def setupStartVariables(self):
        self.wWriteList = [tf.zeros([self.batchSize, self.memory.length])]
        self.wReadList = [tf.zeros([self.batchSize, self.memory.length])]

    def getWW(self, O):
        w_ = self.wWriteList[-1]
        return self.getW(O, w_)

    def getWR(self, O):
        w_ = self.wReadList[-1]
        return self.getW(O, w_)

    def getW(self, O, w_):
        assert helper.check(w_, [self.memory.length], self.batchSize)

        k = tf.nn.softplus(helper.map("map_k", O, self.memory.bitDepth))
        b = tf.nn.softplus(helper.map("map_b", O, 1))
        g = tf.sigmoid(helper.map("map_g", O, 1))
        s = tf.nn.softmax(tf.sigmoid(helper.map("map_s", O, 5))) #Added sigmoid
        y = tf.nn.softplus(helper.map("map_y", O, 1)) + 1

        wc = self.getCosSimSoftMax(k, b)
        wg = self.getWg(wc, g, w_)
        wm = self.getWmFast(wg, s)

        #wm can be negtive -> power will push it into the complex domain
        pow = tf.pow(wm, y)
        w =  pow / (tf.reduce_sum(pow, axis=-1, keep_dims=True)+0.001)

        assert helper.check(w, [self.memory.length], self.batchSize)
        return w

    def getWg(self, wc, g, w_):
        assert helper.check(wc, [self.memory.length], self.batchSize)
        assert helper.check(g, [1], self.batchSize)
        assert helper.check(w_, [self.memory.length], self.batchSize)

        result = g*wc + (1-g)*w_

        assert helper.check(result, [self.memory.length], self.batchSize)
        return result

    def getWm(self, wg, s):
        assert helper.check(wg, [self.memory.length], self.batchSize)
        assert helper.check(s, [5], self.batchCheck)

        size = self.memory.length
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

        assert helper.check(result, [self.memory.length], self.batchSize)
        return result

    def getWmFast(self, wg, s):
        #Amount of concat operations is proportional to the shift size, instead of memory length (Only significantly faster on a big memory)
        assert helper.check(wg, [self.memory.length], self.batchSize)
        assert helper.check(s, [5], self.batchSize)

        w1 = tf.concat([wg[:,-2:], wg[:,:-2]], axis=-1)
        w2 = tf.concat([wg[:,-1:], wg[:,:-1]], axis=-1)
        w4 = tf.concat([wg[:,1:], wg[:,:1]], axis=-1)
        w5 = tf.concat([wg[:,2:], wg[:,:2]], axis=-1)

        w = tf.stack([w1,w2,wg,w4,w5], axis=-1)
        result = tf.squeeze(tf.matmul(w, tf.expand_dims(s, axis=-1)), axis=-1)

        assert helper.check(result, [self.memory.length], self.batchSize)
        return result