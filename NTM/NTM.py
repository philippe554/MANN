import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
from heads.NTMHead import *
from heads.LRUAHead import *
import helper
import random
import time
import matplotlib.pyplot as plt

length = 3
bitDepth = 6
inputMask = length * [0] + [0] + length * [0]
outputMask = (length) * [0] + [0] + (length) * [1]

X,Y = helper.getNewxy(length, bitDepth)

print("IM: " + str(inputMask))
print("OM: " + str(outputMask))
print("Example data point: ")
print(X)
print(Y)

Xfull,Yfull= helper.getNewxyBatch(length, bitDepth, 50000)

x = tf.placeholder(tf.float32, shape=(None, inputMask.count(0), bitDepth+1))
_y = tf.placeholder(tf.float32, shape=(None, outputMask.count(1), bitDepth))

#head = NTMHead("head1")
head = LRUAHead("head1")
cell = NTMCell("ntm", bitDepth, 10, 24, 25, head)


y,W = cell.build(x, inputMask=inputMask, outputMask=outputMask, outputSize=bitDepth)
W=tf.squeeze(W)

crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
loss = tf.reduce_sum(crossEntropy)
#optimizer = tf.train.AdamOptimizer()
optimizer = tf.train.RMSPropOptimizer(0.001)
#trainStep = optimizer.minimize(crossEntropy)
grads_and_vars = optimizer.compute_gradients(crossEntropy)
trainStep = optimizer.apply_gradients(grads_and_vars)


p = tf.round(tf.sigmoid(y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))

helper.printStats(tf.trainable_variables())

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

for grad, var in grads_and_vars:
    if grad is not None:
        tf.summary.histogram("grad/"+var.name, grad)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:/temp/tf_log/", sess.graph)
    # python -m tensorboard.main --logdir="C:/temp/tf_log/"
    # localhost:6006

    plt.ion()

    for i in range(100000):
        trainLoss=0
        start_time = time.time()
        for j in range(100):
            X,Y = helper.getSampleOf(Xfull, Yfull, 100)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss+=l
        duration = time.time() - start_time
        trainLoss = trainLoss/100

        X,Y = helper.getNewxyBatch(length, bitDepth, 100)
        acc, testLoss, summary = sess.run([accuracy, loss, merged], feed_dict={x: X, _y: Y})
        writer.add_summary(summary, i)

        print("#" + str(i+1) + "\tacc: " + "{0:.4f}".format(acc) + "\tLoss: " + str(int(trainLoss)) + "-" + str(int(testLoss)) + "\tTime: " + "{0:.4f}".format(duration) + "s")

        if(i%1==0):
            X,Y = helper.getNewxyBatch(length, bitDepth, 1)
            acc, w = sess.run([accuracy, W], feed_dict={x: X, _y: Y})
            plt.imshow(w, vmin=0, vmax=1);
            plt.show()
            plt.pause(0.05)
            