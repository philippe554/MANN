import numpy as np
import random

from DataGen.Data import *

import tensorflow as tf

class DataGenBase:
    def makeDataset(self, amount):
        x=[]
        y=[]
        c={}
        for i in range(0, amount):
            X,Y,C = self.getEntry()
            x.append(X)
            y.append(Y)
            if C in c:
                c[C]+=1
            else:
                c[C]=0
        return Data(x,y,c)

    def postBuild(self, _y, y, optimizer):
        if  self.postBuildMode == "softmax":
            crossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=y)
            loss = tf.reduce_sum(crossEntropy)

            grads_and_vars = optimizer.compute_gradients(loss)
            trainStep = optimizer.apply_gradients(grads_and_vars)

            p = tf.nn.softmax(y)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(_y, -1), tf.argmax(p, -1)), tf.float32))
        
            return trainStep, p, accuracy, loss

        elif self.postBuildMode == "sigmoid":
            crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
            loss = tf.reduce_sum(crossEntropy)

            grads_and_vars = optimizer.compute_gradients(loss)
            trainStep = optimizer.apply_gradients(grads_and_vars)

            p = tf.round(tf.nn.sigmoid(y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))
        
            return trainStep, p, accuracy, loss

        else:
            return self.customPostBuild(_y, y, optimizer)