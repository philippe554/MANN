import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from MANN.Head.HeadBase import *

#This head is not in line with paper, needs revision

class LRUAHead(HeadBase):
    def setupStartVariables(self):
        self.wWriteList = [tf.zeros([self.batchSize, self.memory.length])]
        self.wReadList = [tf.zeros([self.batchSize, self.memory.length])]
        self.u = tf.zeros([self.batchSize, self.memory.length])

    def getWW(self, O):
        g = tf.sigmoid(helper.map("map_g", O, 1))
        b = tf.nn.softplus(helper.map("map_b", O, 1))

        #differentiable approximation of lu
        lu = tf.nn.softmax((1-tf.sigmoid(self.u))*b)
        w = g*self.wReadList[-1] + (1-g)*lu
        self.u = 0.95*self.u + self.wReadList[-1] + w

        assert helper.check(w, [self.memory.length], self.batchSize)
        return w

    def getWR(self, O):
        k = tf.nn.softplus(helper.map("map_k", O, self.memory.bitDepth))
        b = tf.nn.softplus(helper.map("map_b", O, 1))

        w = self.getCosSimSoftMax(k, b)

        assert helper.check(w, [self.memory.length], self.batchSize)
        return w

    
