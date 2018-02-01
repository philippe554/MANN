import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from heads.MANNHeadPrototype import *

class LRUAHead(MANNHeadPrototype):
    def setupStartVariables(self):
        self.wFirst = tf.sigmoid(helper.getTrainableConstant("w", self.memorylength, self.batchSize)) #Added sigmoid

        if self.mode == "Read":
            self.readFirst = helper.getTrainableConstant("firstRead", self.memoryBitSize, self.batchSize)
            self.readList = []

        #self.u = tf.sigmoid(helper.getTrainableConstant("u", self.memorylength, self.batchSize)) #Added sigmoid

    def buildRead(self, M, O):
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(self.wRead, [self.memorylength], self.batchSize)
        assert helper.check(O, [self.controllerSize], self.batchSize)

        with tf.variable_scope(self.name, reuse=True):
            with tf.variable_scope("read", reuse=True):
                k = tf.nn.softplus(helper.map("map_k", O, self.memoryBitSize))
                b = tf.nn.softplus(helper.map("map_b", O, 1))

                self.oldWRead = self.wRead
                self.wRead = self.getCosSimSoftMax(k, M, b)
                result = tf.squeeze(tf.matmul(tf.expand_dims(self.wRead,axis=-2),M),axis=-2)

        assert helper.check(result, [self.memoryBitSize], self.batchSize)
        assert helper.check(self.wRead, [self.memorylength], self.batchSize)
        return result, self.wRead

    def buildWrite(self, M, O):
        assert helper.check(M, [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(self.wWrite, [self.memorylength], self.batchSize)
        assert helper.check(O, [self.controllerSize], self.batchSize)

        with tf.variable_scope(self.name, reuse=True):
            with tf.variable_scope("write", reuse=True):
                g = tf.sigmoid(helper.map("map_g", O, 1))
                b = tf.nn.softplus(helper.map("map_b", O, 1))
                erase = tf.sigmoid(helper.map("map_erase", O, self.memoryBitSize))
                add = tf.tanh(helper.map("map_add", O, self.memoryBitSize))

                #differentiable approximation of lu
                lu = tf.nn.softmax((1-tf.sigmoid(self.u))*b)
                self.wWrite = g*self.oldWRead + (1-g)*lu
                self.u = 0.95*self.u + self.wRead + self.wWrite

                M = tf.multiply(M, 1 - tf.matmul(tf.expand_dims(self.wWrite, axis=-1),tf.expand_dims(erase, axis=-2)))
                result = M + tf.matmul(tf.expand_dims(self.wWrite, axis=-1),tf.expand_dims(add, axis=-2))

        assert helper.check(result, [self.memorylength, self.memoryBitSize], self.batchSize)
        assert helper.check(self.wWrite, [self.memorylength], self.batchSize)
        return result, self.wWrite
