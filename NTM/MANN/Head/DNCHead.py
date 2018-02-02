import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from MANN.Head.HeadBase import *

class DNCHead(HeadBase):
    def setupStartVariables(self):
        self.wFirst = tf.sigmoid(helper.getTrainableConstant("w", self.memorylength, self.batchSize)) #Added sigmoid

        if self.mode == "Read":
            self.readFirst = helper.getTrainableConstant("firstRead", self.memoryBitSize, self.batchSize)
            self.readList = []

        self.amountReadHeads = 2

    def run(self, O, memory):
        f = 0
        gw = 0
        ga = 0
        pi = 0
        kW = 0
        kR = 0
        bW = 0
        bR = 0

        self.u = self.getU(self.u, self.wW, self.wR, f)
        a = self.getA(u)
        cW = self.getCosSimSoftMax(kW, memory.getLast(), bW)
        self.wW = self.getWW(gw, ga, a, cW)

        M = memory.getLast()
        memory.new(M)

        self.l = self.getL(self.l, self.wW, self.p)
        cR = self.getCosSimSoftMax(kR, memory.getLast(), bR)
        self.wR = self.getWR(self.wR, self.l, cR, pi)

        #Calc p after calc l
        self.p = self.getP(self.p, self.wW)

    def getU(self, _u, _wW, _wR, f):
        assert helper.check(_u, [self.memorylength], self.batchSize)
        assert helper.check(_wW, [self.memorylength], self.batchSize)
        assert helper.check(_wR, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(f, [self.amountReadHeads], self.batchSize)

        v = tf.reduce_prod(1-tf.exapnd_dim(f, axis=-1)*_wR, axis=-2)
        assert helper.check(v, [self.memorylength], self.batchSize)

        u = (_u + _wW - _u*_wW) * v
        assert helper.check(u, [self.memorylength], self.batchSize)

        return u

    def getA(self, u):
        assert helper.check(u, [self.memorylength], self.batchSize)

        uSorted, uIndices = tf.nn.top_k(-1 * u, k=self.memorylength)
        uSorted *= -1
        assert helper.check(uSorted, [self.memorylength], self.batchSize)
        assert helper.check(uIndices, [self.memorylength], self.batchSize)

        cumProd = tf.cumprod(uSorted, axis=-1, exclusive=True)
        assert helper.check(cumProd, [self.memorylength], self.batchSize)

        aSorted = (1 - uSorted) * cumProd
        assert helper.check(aSorted, [self.memorylength], self.batchSize)

        #Far from sure this works, but seems faster/cleaner than implementation of Siraj Raval
        a = tf.dynamic_stitch(uIndices, aSorted)
        assert helper.check(a, [self.memorylength], self.batchSize)

        return a

    def getWW(self, gw, ga, a, c):
        assert helper.check(gw, [1], self.batchSize)
        assert helper.check(ga, [1], self.batchSize)
        assert helper.check(a, [self.memorylength], self.batchSize)
        assert helper.check(c, [self.memorylength], self.batchSize)

        w = gw * (ga*a + (1-ga)*c)
        assert helper.check(wW, [self.memorylength], self.batchSize)

        return w

    def getP(self, _p, w):
        assert helper.check(_p, [self.memorylength], self.batchSize)
        assert helper.check(w, [self.memorylength], self.batchSize)

        p = (1 - tf.reduce_sum(w, axis=-1))*_p + w
        assert helper.check(p, [self.memorylength], self.batchSize)

        return p
        
    def getL(self, _l, w, _p):
        assert helper.check(_l, [self.memorylength, self.memorylength], self.batchSize)
        assert helper.check(w, [self.memorylength], self.batchSize)
        assert helper.check(_p, [self.memorylength], self.batchSize)

        w_l = (1 - w - tf.transpose(w)) * _l
        assert helper.check(w_p, [self.memorylength, self.memorylength], self.batchSize)

        w_p = tf.matmul(w, _p, transpose_b=True)
        assert helper.check(w_p, [self.memorylength, self.memorylength], self.batchSize)

        l = w_l + w_p
        assert helper.check(l, [self.memorylength, self.memorylength], self.batchSize)

        return l

    def getWR(self, _w, l, c, pi):
        assert helper.check(_w, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(l, [self.memorylength, self.memorylength], self.batchSize)
        assert helper.check(c, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(pi, [self.amountReadHeads, 3], self.batchSize)

        #Does not work due to multiple read vectors
        f = tf.mat_mul(l, _w)
        b = tf.matmul(l, _w, transpose_a=True)
        assert helper.check(f, [self.amountReadHeads, self.memorylength], self.batchSize)
        assert helper.check(b, [self.amountReadHeads, self.memorylength], self.batchSize)

        w = pi[0]*b + pi[1]*c + pi[2]*f
        assert helper.check(w, [self.amountReadHeads, self.memorylength], self.batchSize)

        return w
