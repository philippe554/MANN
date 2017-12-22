import tensorflow as tf
import pandas as pd
import numpy as np

import helper
import LSTMCell

class NTMCell:
    def __init__(self, name):
        self.name = name;
        self.LSTM = LSTMCell("controller")

    def buildTimeLayer(self, input, prevLSTMOutput, prevLSTMState, prevRead, M, wRead, wWrite):
        with tf.variable_scope(self.name, reuse=true):
            I = tf.concat(input, prevRead)
            LSTMOuput, LSTMState = self.LSTM.buildTimeLayer(I, prevLSTMOutput, prevLSTMState)

            wRead = processHead(LSTMOuput, M, wRead, "read")
            wWrite = processHead(LSTMOuput, M, wWrite, "write")

            R = read(M, wRead)
            M = write(LSTMOuput, M, wWrite, "write")

            OR = tf.concat([LSTMOuput,R], 0)

            ouput = map("combine", OR, 10)

            return LSTMOuput, LSTMState, output, R, M, wRead, wWrite

    def processHead(self, O, M, w_, name):
        with tf.variable_scope(name, reuse=true):
            k = tf.tanh(map("map_k", O, M.get_shape()[1]))
            b = tf.nn.softplus(map("map_b", O, 1))
            g = tf.sigmoid(map("map_g", O, 1))
            s = tf.nn.softmax(map("map_s", O, 5))
            y = tf.nn.softplus(map("map_y", O, 1)) + 1

            wc = getWc(k, M, b)
            wg = getWg(wc, g, w_)
            wm = getWm(wg, s)
            w = getW(wm, y)

            return w

    def getWc(k, M, b):
        dot = tf.matmul(M, tf.reshape(k, [-1, 1]))
        l1 = tf.norm(k,axis=0)
        l2 = tf.norm(M,axis=1)
        cosSim = tf.divide(tf.reshape(dot,[-1]), l1 * l2 + 0.001)
        e = tf.exp(b * cosSim)
        return tf.divide(e, tf.reduce_sum(e))

    def getWg(wc, g, w_):
        gs = tf.squeeze(g)
        return tf.scalar_mul(gs, wc) + tf.scalar_mul(1-gs, w_)

    def getWm(wg, s):
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

    def getW(wm,y):
        pow = tf.pow(wm, y)
        return  pow / tf.reduce_sum(pow)

    def read(M, w):
        return tf.reshape(tf.matmul(tf.reshape(w,[1,-1]),M),[-1])

    def write(O, M, w, name):
        with tf.variable_scope(name, reuse=true):
            erase = tf.sigmoid(map("map_erase", O, M.get_shape()[1]))
            add = tf.tanh(map("map_add", O, M.get_shape()[1]))

            M = tf.multiply(M, 1 - tf.matmul(tf.reshape(w,[-1,1]),tf.reshape(erase, [1,-1])))
            return M + tf.matmul(tf.reshape(w,[-1,1]),tf.reshape(add, [1,-1]))