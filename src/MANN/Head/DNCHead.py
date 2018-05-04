import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from MANN.Head.HeadBase import *

class DNCHead(HeadBase):
    def setupStartVariables(self):
        self.wWriteList = [tf.zeros([self.batchSize, self.memory.length])]
        self.wReadList = [tf.zeros([self.batchSize, self.amountReadHeads, self.memory.length])]

        self.u = tf.zeros([self.batchSize, self.memory.length])
        self.p = tf.zeros([self.batchSize, self.memory.length])
        self.l = tf.zeros([self.batchSize, self.memory.length, self.memory.length])

        self.lMask = tf.ones([self.batchSize, self.memory.length, self.memory.length]) - tf.eye(self.memory.length, batch_shape=[self.batchSize])
        assert helper.check(self.lMask, [self.memory.length, self.memory.length], self.batchSize)

    def getWW(self, O):
        mapping = [self.amountReadHeads, self.memory.bitDepth, 1, 1, 1]
        o = helper.map("map_wwo", O, np.sum(mapping))
        o1, o2, o3, o4, o5 = tf.split(o, mapping, -1)

        f = tf.sigmoid(o1)
        self.u = self.getU(f, self.u, self.wWriteList[-1], self.wReadList[-1])
        a = self.getA(self.u)

        kW = o2
        bW = tf.nn.softplus(o3) + 1
        c = self.getCosSimSoftMax(kW, bW)

        gw = tf.sigmoid(o4)
        ga = tf.sigmoid(o5)

        w = gw * (ga*a + (1-ga)*c)
        assert helper.check(w, [self.memory.length], self.batchSize)

        return w

    def getWR(self, O):
        mapping = [self.amountReadHeads * self.memory.bitDepth, self.amountReadHeads, self.amountReadHeads * 3]
        o = helper.map("map_wro", O, np.sum(mapping))
        o1, o2, o3 = tf.split(o, mapping, -1)

        self.l = self.getL(self.l, self.wWriteList[-1], self.p)

        _w = self.wReadList[-1]
        assert helper.check(_w, [self.amountReadHeads, self.memory.length], self.batchSize)

        f = tf.matmul(_w, self.l)
        b = tf.matmul(_w, self.l, transpose_b=True)
        assert helper.check(f, [self.amountReadHeads, self.memory.length], self.batchSize)
        assert helper.check(b, [self.amountReadHeads, self.memory.length], self.batchSize)

        kR = tf.reshape(o1, [-1, self.amountReadHeads, self.memory.bitDepth])
        bR = tf.nn.softplus(o2) + 1
        c = self.getCosSimSoftMaxExtra(kR, bR, self.amountReadHeads)

        pi = tf.nn.softmax(tf.reshape(o3, [-1, self.amountReadHeads, 3]))
        w = tf.expand_dims(pi[:, :, 0], axis=-1)*b + tf.expand_dims(pi[:, :, 1], axis=-1)*c + tf.expand_dims(pi[:, :, 2], axis=-1)*f
        assert helper.check(w, [self.amountReadHeads, self.memory.length], self.batchSize)

        self.p = self.getP(self.p, self.wWriteList[-1])

        return w        

    def getU(self, f, _u, _wW, _wR):
        assert helper.check(f, [self.amountReadHeads], self.batchSize)
        assert helper.check(_u, [self.memory.length], self.batchSize)
        assert helper.check(_wW, [self.memory.length], self.batchSize)
        assert helper.check(_wR, [self.amountReadHeads, self.memory.length], self.batchSize)

        #If a reading head reads a memory adress in t-1, and the free gate is activated, release the memory
        v = tf.reduce_prod(1-(tf.expand_dims(f, axis=-1)*_wR), axis=-2)
        assert helper.check(v, [self.memory.length], self.batchSize)

        #If you write to a memory adress, reserve it
        u = (_u + _wW - (_u*_wW)) * v
        assert helper.check(u, [self.memory.length], self.batchSize)
        
        return u

    def getA(self, u):
        assert helper.check(u, [self.memory.length], self.batchSize)

        uSorted, uIndices = tf.nn.top_k(-1 * u, k=self.memory.length)
        uSorted *= -1
        assert helper.check(uSorted, [self.memory.length], self.batchSize)
        assert helper.check(uIndices, [self.memory.length], self.batchSize)

        cumProd = tf.cumprod(uSorted + 0.0001, axis=-1, exclusive=True)
        assert helper.check(cumProd, [self.memory.length], self.batchSize)

        aSorted = (1 - uSorted) * cumProd
        assert helper.check(aSorted, [self.memory.length], self.batchSize)

        a = tf.reshape(tf.gather(tf.reshape(aSorted, [self.batchSize * self.memory.length]), tf.reshape(uIndices, [self.batchSize * self.memory.length])), [self.batchSize, self.memory.length])
        assert helper.check(a, [self.memory.length], self.batchSize)

        return a

    def getP(self, _p, w):
        assert helper.check(_p, [self.memory.length], self.batchSize)
        assert helper.check(w, [self.memory.length], self.batchSize)

        p = (1 - tf.reduce_sum(w, axis=-1, keep_dims=True))*_p + w
        assert helper.check(p, [self.memory.length], self.batchSize)

        return p
        
    def getL(self, _l, w, _p):
        assert helper.check(_l, [self.memory.length, self.memory.length], self.batchSize)
        assert helper.check(w, [self.memory.length], self.batchSize)
        assert helper.check(_p, [self.memory.length], self.batchSize)

        o = tf.ones([self.batchSize, self.memory.length, self.memory.length])
        o_w = o - tf.expand_dims(w, axis=-2)
        assert helper.check(o_w, [self.memory.length, self.memory.length], self.batchSize)

        o_ww = o_w - tf.transpose(tf.expand_dims(w, axis=-2), perm=[0,2,1])
        assert helper.check(o_ww, [self.memory.length, self.memory.length], self.batchSize)

        w_l = o_ww * _l
        assert helper.check(w_l, [self.memory.length, self.memory.length], self.batchSize)

        w_p = tf.matmul(tf.expand_dims(w, axis=-1), tf.expand_dims(_p, axis=-2))
        assert helper.check(w_p, [self.memory.length, self.memory.length], self.batchSize)

        l = (w_l + w_p) * self.lMask
        assert helper.check(l, [self.memory.length, self.memory.length], self.batchSize)

        return l
