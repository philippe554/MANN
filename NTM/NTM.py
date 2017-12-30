import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
import helper

def makeNTM(x, mask):  
    ntmCell = NTMCell("ntm", 5, 4, 4, 8)
    output = []

    for i in range(0,x.get_shape()[1]):
        print("Building step: "+str(i+1))
        input = tf.squeeze(tf.slice(x,[0,i],[x.get_shape()[0],1]))
        O = ntmCell.buildTimeLayer(input)
        
        if(mask[i]==1):
            output.append(tf.expand_dims(O, 1))

    return tf.concat(output, axis=1)

length = 8
bitDepth = 4

x = tf.placeholder(tf.float32, shape=(bitDepth + 1, length * 2))
_y = tf.placeholder(tf.float32, shape=(bitDepth, length))

mask = np.concatenate((np.zeros((length)),np.ones((length))), axis=0)
y = makeNTM(x, mask)

crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
trainStep = tf.train.AdamOptimizer().minimize(crossEntropy)

p = tf.round(tf.sigmoid(y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        X,Y = helper.getNewxy(length, bitDepth)
        trainStep.run(feed_dict={x: X, _y: Y})

        a = sess.run(accuracy, feed_dict={x: X, _y: Y})

        print("Training step: " + str(i+1) + " A: " + str(a))