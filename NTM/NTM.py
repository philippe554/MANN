import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
import helper
import random

length = 8
bitDepth = 6
inputMask = length * [1,0] + (length*2) * [2]
outputMask = (length*2) * [0] + (length) * [0,1]

x = tf.placeholder(tf.float32, shape=(None, inputMask.count(0), bitDepth))
_y = tf.placeholder(tf.float32, shape=(None, outputMask.count(1), bitDepth))
y = NTMCell("ntm", bitDepth, 20, 128, 100).build(x, inputMask=inputMask, outputMask=outputMask, outputSize=bitDepth)

crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
trainStep = tf.train.AdamOptimizer().minimize(crossEntropy)

p = tf.round(tf.sigmoid(y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))

helper.printStats(tf.trainable_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        for j in range(10):
            X,Y = helper.getNewxyBatch(length, bitDepth, 100)
            trainStep.run(feed_dict={x: X, _y: Y})

        X,Y = helper.getNewxyBatch(length, bitDepth, 100)
        acc = sess.run(accuracy, feed_dict={x: X, _y: Y})

        print("Training batch: " + str(i+1) + " | AVG accuracy: " + str(acc))
