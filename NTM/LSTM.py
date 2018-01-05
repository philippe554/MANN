import tensorflow as tf
import pandas as pd
import numpy as np
from LSTMCell import *
import helper

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def makeLSTM(x):  
    ntmCell = LSTMCell("ntm", 28, 10, 10)
    output = []

    for i in range(0,28):
        print("Building step: "+str(i+1))
        input = tf.squeeze(tf.slice(x,[0,i,0],[-1,1,28]), axis=1)
        O = ntmCell.buildTimeLayerBatch(input, i==0)
        
        output.append(tf.expand_dims(O, 0))

    #return tf.concat(output, axis=0)
    return tf.squeeze(output[-1])

x = tf.placeholder(tf.float32, shape=(None, 28, 28))
_y = tf.placeholder(tf.float32, shape=(None, 10))

y = makeLSTM(x)

crossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=y)
trainStep = tf.train.AdamOptimizer().minimize(crossEntropy)

p = tf.equal(tf.argmax(y, 0), tf.argmax(_y, 0))
accuracy = tf.reduce_mean(tf.cast(p, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sum=0.0

    for step in range(0, 1000000):
        batch_x, batch_y = mnist.train.next_batch(1)
        batch_x = batch_x.reshape((1, 28, 28))
        batch_y = batch_y.reshape((1, 10))
        sess.run(trainStep, feed_dict={x: batch_x, _y: batch_y})

        sum += sess.run(accuracy, feed_dict={x: batch_x, _y: batch_y})
        if(step%1000 == 0):
            print(str(step)+": "+str(sum/1000.0))
            sum=0
            
