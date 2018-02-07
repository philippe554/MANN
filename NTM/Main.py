import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

import mann
import helper

#Define the MANN
cell = mann.MANNUnit("L1MANN")
cell.addMemory(mann.BasicMemory("M1", 24, 10))
cell.addController(mann.FFCell("Controller1", 25))
cell.addHead(mann.DNCHead("Head1", 2))

#Define the test data
#generator = mann.MinPath(15, 20, 7, 4)
generator = mann.Copy(10,8)

#Define optimizer
optimizer = tf.train.RMSPropOptimizer(0.001)

#### End of configuration ####

#Build the network
x = tf.placeholder(tf.float32, shape=(None, generator.inputLength, generator.inputSize))
_y = tf.placeholder(tf.float32, shape=(None, generator.outputLength, generator.outputSize))
y = cell.build(x, generator.outputMask, generator.outputSize)

#Build optimizer
trainStep, p, accuracy, loss = generator.postBuild(_y, y, optimizer)

#Visualize parameters
helper.printStats(tf.trainable_variables())

#Generate the data
trainData = generator.makeDataset(10000)
testData = generator.makeDataset(1000)

#Train network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    X,Y = testData.getBatch(100)
    acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})
    print("Start:" + "\tacc: " + str(acc) + "\tLoss: " + str(testLoss))

    for i in range(100000):
        trainLoss=0
        start_time = time.time()
        for j in range(100):
            X,Y = trainData.getBatch(100)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss+=l
        duration = time.time() - start_time
        trainLoss = trainLoss/100

        X,Y = testData.getBatch(100)
        acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})

        print("#" + str(i+1) + "\tacc: " + str(acc) + "\tLoss: " + str(trainLoss) + "-" + str(testLoss) + "\tTime: " + "{0:.4f}".format(duration) + "s")
        
