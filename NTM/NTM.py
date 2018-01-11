import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
import helper

length = 5
bitDepth = 3

def makeNTM(x, mask):  
    ntmCell = NTMCell("ntm", bitDepth+1, bitDepth, bitDepth*2, length*2, 12)
    output = []

    for i in range(0,x.get_shape()[0]):
        print("Building step: "+str(i+1))
        input = tf.squeeze(tf.slice(x,[i,0],[1,bitDepth+1]))
        O = ntmCell.buildTimeLayer(input)
        
        if(mask[i]==1):
            output.append(tf.expand_dims(O, 0))

    return tf.concat(output, axis=0)

x = tf.placeholder(tf.float32, shape=(length * 2, bitDepth+1))
_y = tf.placeholder(tf.float32, shape=(length, bitDepth))

#mask = np.concatenate((np.zeros((length*2)), np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])), axis=0)
mask = np.concatenate((np.zeros((length)), np.ones((length))), axis=0)
y = makeNTM(x, mask)

crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
trainStep = tf.train.AdamOptimizer().minimize(crossEntropy)

p = tf.round(tf.sigmoid(y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))

helper.printStats(tf.trainable_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        #X,Y = helper.getNewxyBatch(length, bitDepth,100)
        #trainStep.run(feed_dict={x: X, _y: Y})

        for j in range(1000):
            X,Y = helper.getNewxy(length, bitDepth)
            trainStep.run(feed_dict={x: X, _y: Y})

        sum = 0.0
        for j in range(100):
            X,Y = helper.getNewxy(length, bitDepth)
            sum += sess.run(accuracy, feed_dict={x: X, _y: Y})

        print("Training batch: " + str(i+1) + " | AVG accuracy: " + str(sum/100))