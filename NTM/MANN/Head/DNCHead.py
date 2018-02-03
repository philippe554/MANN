import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from MANN.Head.HeadBase import *

class DNCHead(HeadBase):
    def setupStartVariables(self):
        self.amountReadHeads = 2

        self.readFirst = tf.zeros([self.batchSize, self.memoryBitSize * self.amountReadHeads])
        self.readList = []

        self.u = tf.zeros([self.batchSize, self.memorylength])
        self.wW = tf.zeros([self.batchSize, self.memorylength])
        self.wR = tf.zeros([self.batchSize, self.amountReadHeads, self.memorylength])
        self.p = tf.zeros([self.batchSize, self.memorylength])
        self.l = tf.zeros([self.batchSize, self.memorylength, self.memorylength])

        self.lMask = tf.ones([self.batchSize, self.memorylength, self.memorylength]) - tf.eye(self.memorylength, batch_shape=[self.batchSize])
        assert helper.check(self.lMask, [self.memorylength, self.memorylength], self.batchSize)

    def buildHead(self, memory, O):
        kR = tf.reshape(helper.map("map_kR", O, self.amountReadHeads * self.memoryBitSize), [-1, self.amountReadHeads, self.memoryBitSize])
        bR = tf.nn.softplus(helper.map("map_bR", O, self.amountReadHeads)) + 1

        kW = helper.map("map_kW", O, self.memoryBitSize)
        bW = tf.nn.softplus(helper.map("map_bW", O, 1)) + 1

        erase = tf.sigmoid(helper.map("map_erase", O, self.memoryBitSize))
        write = helper.map("map_write", O, self.memoryBitSize)

        f = tf.sigmoid(helper.map("map_f", O, self.amountReadHeads))

        gw = tf.sigmoid(helper.map("map_gw", O, 1))
        ga = tf.sigmoid(helper.map("map_ga", O, 1))

        pi = tf.nn.softmax(tf.reshape(helper.map("map_pi", O, self.amountReadHeads * 3), [-1, self.amountReadHeads, 3]))

        self.u = self.getU(self.u, self.wW, self.wR, f)
        a = self.getA(self.u)
        cW = self.getCosSimSoftMax(kW, memory.getLast(), bW)
        self.wW = self.getWW(gw, ga, a, cW)

        self.writeToMemory(memory, erase, write, self.wW)

        self.l = self.getL(self.l, self.wW, self.p)
        cR = self.getCosSimSoftMaxExtra(kR, memory.getLast(), bR, self.amountReadHeads)
        self.wR = self.getWR(self.wR, self.l, cR, pi)

        self.readFromMemory(memory, self.wR, self.amountReadHeads)

        #Calc p after calc l
        self.p = self.getP(self.p, self.wW)

    def getU(self, _u, _wW, _wR, f):
        assert helper.check(_u, [self.memorylength], self.batchSize)
        assert helper.check(_wW, [self.memorylength], self.batchSize)
        assert helper.check(_wR, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(f, [self.amountReadHeads], self.batchSize)

        v = tf.reduce_prod(1-(tf.expand_dims(f, axis=-1)*_wR), axis=-2)
        assert helper.check(v, [self.memorylength], self.batchSize)

        u = (_u + _wW - (_u*_wW)) * v
        assert helper.check(u, [self.memorylength], self.batchSize)

        return u

    def getA(self, u):
        assert helper.check(u, [self.memorylength], self.batchSize)

        uSorted, uIndices = tf.nn.top_k(-1 * u, k=self.memorylength)
        uSorted *= -1
        assert helper.check(uSorted, [self.memorylength], self.batchSize)
        assert helper.check(uIndices, [self.memorylength], self.batchSize)

        cumProd = tf.cumprod(uSorted + 0.0001, axis=-1, exclusive=True)
        assert helper.check(cumProd, [self.memorylength], self.batchSize)

        aSorted = (1 - uSorted) * cumProd
        assert helper.check(aSorted, [self.memorylength], self.batchSize)

        #Far from sure this works, but seems faster/cleaner than implementation of Siraj Raval
        a = tf.reshape(tf.gather(tf.reshape(aSorted, [self.batchSize * self.memorylength]), tf.reshape(uIndices, [self.batchSize * self.memorylength])), [self.batchSize, self.memorylength])
        assert helper.check(a, [self.memorylength], self.batchSize)

        return a

    def getWW(self, gw, ga, a, c):
        assert helper.check(gw, [1], self.batchSize)
        assert helper.check(ga, [1], self.batchSize)
        assert helper.check(a, [self.memorylength], self.batchSize)
        assert helper.check(c, [self.memorylength], self.batchSize)

        w = gw * (ga*a + (1-ga)*c)
        assert helper.check(w, [self.memorylength], self.batchSize)

        return w

    def getP(self, _p, w):
        assert helper.check(_p, [self.memorylength], self.batchSize)
        assert helper.check(w, [self.memorylength], self.batchSize)

        p = (1 - tf.reduce_sum(w, axis=-1, keep_dims=True))*_p + w
        assert helper.check(p, [self.memorylength], self.batchSize)

        return p
        
    def getL(self, _l, w, _p):
        assert helper.check(_l, [self.memorylength, self.memorylength], self.batchSize)
        assert helper.check(w, [self.memorylength], self.batchSize)
        assert helper.check(_p, [self.memorylength], self.batchSize)

        o = tf.ones([self.batchSize, self.memorylength, self.memorylength])
        o_w = o - tf.expand_dims(w, axis=-2)
        assert helper.check(o_w, [self.memorylength, self.memorylength], self.batchSize)

        o_ww = o_w - tf.transpose(tf.expand_dims(w, axis=-2), perm=[0,2,1])
        assert helper.check(o_ww, [self.memorylength, self.memorylength], self.batchSize)

        w_l = o_ww * _l
        assert helper.check(w_l, [self.memorylength, self.memorylength], self.batchSize)

        w_p = tf.matmul(tf.expand_dims(w, axis=-1), tf.expand_dims(_p, axis=-2))
        assert helper.check(w_p, [self.memorylength, self.memorylength], self.batchSize)

        l = (w_l + w_p) * self.lMask
        assert helper.check(l, [self.memorylength, self.memorylength], self.batchSize)

        return l

    def getWR(self, _w, l, c, pi):
        assert helper.check(_w, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(l, [self.memorylength, self.memorylength], self.batchSize)
        assert helper.check(c, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(pi, [self.amountReadHeads, 3], self.batchSize)

        f = tf.matmul(_w, l)
        b = tf.matmul(_w, l, transpose_b=True)
        assert helper.check(f, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(b, [self.amountReadHeads, self.memorylength], self.batchSize)

        w = tf.expand_dims(pi[:,:,0], axis=-1)*b + tf.expand_dims(pi[:,:,1], axis=-1)*c + tf.expand_dims(pi[:,:,2], axis=-1)*f
        assert helper.check(w, [self.amountReadHeads, self.memorylength], self.batchSize)

        return w
