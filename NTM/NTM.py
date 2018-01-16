import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
import helper
import random

length = 8
bitDepth = 6
inputMask = length * [0] + [0] + length * [0]
outputMask = (length) * [0] + [0] + (length) * [1]

X,Y = helper.getNewxy(length, bitDepth)

print("IM: " + str(inputMask))
print("OM: " + str(outputMask))
print("Example data point: ")
print(X)
print(Y)

x = tf.placeholder(tf.float32, shape=(None, inputMask.count(0), bitDepth+1))
_y = tf.placeholder(tf.float32, shape=(None, outputMask.count(1), bitDepth))
y = NTMCell("ntm", bitDepth, 10, 16, 24).build(x, inputMask=inputMask, outputMask=outputMask, outputSize=bitDepth)

helper.printStats(tf.trainable_variables())

crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
loss = tf.reduce_sum(crossEntropy)
optimizer = tf.train.RMSPropOptimizer(0.0001)
trainStep = optimizer.minimize(crossEntropy)

p = tf.round(tf.sigmoid(y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        trainLoss=0
        for j in range(20):
            X,Y = helper.getNewxyBatch(length, bitDepth, 50)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss+=l

        trainLoss = trainLoss/10

        X,Y = helper.getNewxyBatch(length, bitDepth, 100)
        acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})

        print("#" + str(i+1) + "\tacc: " + "{0:.4f}".format(acc) + "\tLoss: " + str(int(trainLoss)) + "-" + str(int(testLoss)))
