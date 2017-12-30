import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from LSTMCell import *

class NTMCell:
    def __init__(self, name, inputSize, outputSize, memoryBitSize, memoryLength):
        self.name = name;
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.memoryBitSize = memoryBitSize
        self.memorylength = memoryLength

        self.prevRead = helper.makeStartState("pr", [self.memoryBitSize])
        self.M = helper.makeStartState("m", [self.memorylength, self.memoryBitSize])
        self.wRead = helper.makeStartState("wr", [self.memorylength])
        self.wWrite = helper.makeStartState("ww", [self.memorylength])

        self.LSTM = LSTMCell("controller", self.inputSize + self.memoryBitSize, self.memoryBitSize, 4)        

    def buildTimeLayer(self, input):
        assert(len(input.get_shape()) == 1 and input.get_shape()[0] == self.inputSize)

        with tf.variable_scope(self.name, reuse=True):
            I = tf.concat([input, self.prevRead], axis=0)
            LSTMOuput = self.LSTM.buildTimeLayer(I)

            self.wRead = self.processHead(LSTMOuput, self.M, self.wRead, "read")
            self.wWrite = self.processHead(LSTMOuput, self.M, self.wWrite, "write")

            R = self.read(self.M, self.wRead)
            self.M = self.write(LSTMOuput, self.M, self.wWrite, "write")

            OR = tf.concat([LSTMOuput,R], 0)

            output = helper.map("combine", OR, self.outputSize)

            self.prevRead = R
            return output

    def processHead(self, O, M, w_, name):
        with tf.variable_scope(name, reuse=True):
            k = tf.tanh(helper.map("map_k", O, M.get_shape()[1]))
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
        dot = tf.matmul(M, tf.reshape(k, [-1, 1]))
        l1 = tf.norm(k,axis=0)
        l2 = tf.norm(M,axis=1)
        cosSim = tf.divide(tf.reshape(dot,[-1]), l1 * l2 + 0.001)
        e = tf.exp(b * cosSim)
        return tf.divide(e, tf.reduce_sum(e))

    def getWg(self, wc, g, w_):
        gs = tf.squeeze(g)
        return tf.scalar_mul(gs, wc) + tf.scalar_mul(1-gs, w_)

    def getWm(self, wg, s):
        size = int(wg.get_shape()[0])
        shiftSize = int(int(s.get_shape()[0])/2)

        def shift(i):
            if(i<0):
                return size+i
            if(i>=size):
                return i-size
            return i

        def indices(i):
            indices = [shift(i+j) for j in range(shiftSize,-shiftSize-1,-1)]
            return tf.reduce_sum(tf.gather(wg, indices) * s,0)

        return tf.dynamic_stitch(list(range(0,size)), [indices(i) for i in range(0,size)])

    def getW(self, wm,y):
        pow = tf.pow(wm, y)
        return  pow / tf.reduce_sum(pow)

    def read(self, M, w):
        return tf.reshape(tf.matmul(tf.reshape(w,[1,-1]),M),[-1])

    def write(self, O, M, w, name):
        with tf.variable_scope(name, reuse=True):
            erase = tf.sigmoid(helper.map("map_erase", O, M.get_shape()[1]))
            add = tf.tanh(helper.map("map_add", O, M.get_shape()[1]))

            M = tf.multiply(M, 1 - tf.matmul(tf.reshape(w,[-1,1]),tf.reshape(erase, [1,-1])))
            return M + tf.matmul(tf.reshape(w,[-1,1]),tf.reshape(add, [1,-1]))