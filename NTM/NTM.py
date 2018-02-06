import tensorflow as tf
import pandas as pd
import numpy as np
from MANN.MANNUnit import *
from MANN.Head.NTMHead import *
from MANN.Head.DNCHead import *
from MANN.Head.LRUAHead import *
from MANN.Memory.BasicMemory import *
from RNN.GRUCell import *
from RNN.FFCell import *
from RNN.LSTMCell import *
import helper
from DataGen import MinPath
import random
import time
import matplotlib.pyplot as plt

length = 10
bitDepth = 6
#inputMask = length * [0] + [0] + length * [0]
#outputMask = (length) * [0] + [0] + (length) * [1]

#X,Y = helper.getNewxy(length, bitDepth)

#print("IM: " + str(inputMask))
#print("OM: " + str(outputMask))
#print("Example data point: ")
#print(X)
#print(Y)

Xfull,Yfull= MinPath.getNewBatch(9, 13, 5, 4, 10000)
X,Y = helper.getSampleOf(Xfull, Yfull, 1)
print(X,Y)

x = tf.placeholder(tf.float32, shape=(None, 13+4+1, 9+1))
_y = tf.placeholder(tf.float32, shape=(None, 1, 5+1))

cell = MANNUnit("L1MANN")
cell.addMemory(BasicMemory("M1", 24, 10, "Trainable"))
cell.addController(LSTMCell("Controller1", 25))
cell.addHead(DNCHead("Head1", 2))
#cell.addHead(NTMHead("Head2"))
#cell.addHead(LRUAHead("Head3"))

inputMask = (13+4+1) * [0]
outputMask = (13+4) * [0] + 1 * [1]

y = cell.build(x, inputMask=inputMask, outputMask=outputMask, outputSize=5+1)

#y = helper.map("L2", y, bitDepth)
#W=tf.squeeze(W)

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

#tf.summary.scalar('loss', loss)
#tf.summary.scalar('accuracy', accuracy)

#for grad, var in grads_and_vars:
#    if grad is not None:
#        tf.summary.histogram("grad/"+var.name, grad)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("C:/temp/tf_log/", sess.graph)
    # python -m tensorboard.main --logdir="C:/temp/tf_log/"
    # localhost:6006

    #plt.ion()

    X,Y = MinPath.getNewBatch(9, 13, 5, 4, 100)
    acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})
    #writer.add_summary(summary, i)

    print("Start:" + "\tacc: " + str(acc) + "\tLoss: " + str(testLoss))

    for i in range(100000):
        trainLoss=0
        start_time = time.time()
        for j in range(100):
            X,Y = helper.getSampleOf(Xfull, Yfull, 100)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss+=l
        duration = time.time() - start_time
        trainLoss = trainLoss/100

        X,Y = MinPath.getNewBatch(9, 13, 5, 4, 100)
        acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})
        #writer.add_summary(summary, i)

        print("#" + str(i+1) + "\tacc: " + str(acc) + "\tLoss: " + str(trainLoss) + "-" + str(testLoss) + "\tTime: " + "{0:.4f}".format(duration) + "s")
        #print("#" + str(i+1) + "\tacc: " + "{0:.4f}".format(acc) + "\tLoss: " + str(int(trainLoss)) + "-" + str(int(testLoss)) + "\tTime: " + "{0:.4f}".format(duration) + "s")

        #if(i%1==0):
        #    X,Y = helper.getNewxyBatch(length, bitDepth, 1)
        #    acc, w = sess.run([accuracy, W], feed_dict={x: X, _y: Y})
        #    plt.imshow(w, vmin=0, vmax=1);
        #    plt.show()
        #    plt.pause(0.05)
            