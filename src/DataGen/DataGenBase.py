import numpy as np
import random
import datetime
import os
import pickle

from DataGen.Data import *
import helper

import tensorflow as tf

class DataGenBase:
    saver = None

    def makeDataset(self, amount, token):
        x=[]
        y=[]
        c={}
        for i in range(0, amount):
            helper.progress(i+1, amount, status="Creating dataset of size "+str(amount))
            X,Y,C = self.getEntry()
            x.append(X)
            y.append(Y)
            if C in c:
                c[C]+=1
            else:
                c[C]=0
        return Data(x,y,c)

    def makeAndSaveDataset(self, amount, token):
        dataPath = os.path.join(os.getcwd(), os.pardir, "data", self.name)

        if not os.path.exists(dataPath):
           os.makedirs(dataPath)

        file = os.path.join(dataPath, str(token)+"-"+str(amount)+".p")

        try:
            return pickle.load(open(os.path.abspath(file),"rb"))
        except:
            data = self.makeDataset(amount, token)
            pickle.dump(data, open(os.path.abspath(file), "wb"))
            return data

    def getInput(self):
        return tf.placeholder(tf.float32, shape=(None, self.inputLength, self.inputSize))

    def getLabel(self):
        return tf.placeholder(tf.float32, shape=(None, self.outputLength, self.outputSize))

    def postBuild(self, _y, y, optimizer):
        if self.postBuildMode == "softmax":
            crossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=y)
            loss = tf.reduce_mean(crossEntropy)

            grads_and_vars = optimizer.compute_gradients(loss)
            trainStep = optimizer.apply_gradients(grads_and_vars)

            p = tf.nn.softmax(y)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(_y, -1), tf.argmax(p, -1)), tf.float32))
        
            return trainStep, p, accuracy, loss

        elif self.postBuildMode == "sigmoid":
            crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)
            loss = tf.reduce_mean(crossEntropy)

            grads_and_vars = optimizer.compute_gradients(loss)
            trainStep = optimizer.apply_gradients(grads_and_vars)

            p = tf.round(tf.nn.sigmoid(y))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(_y,p), tf.float32))
        
            return trainStep, p, accuracy, loss

        else:
            return self.customPostBuild(_y, y, optimizer)

    def process(self, X, Y, R):
        pass

    def save(self, sess, epoch, loss):
        modelPath = os.path.join(os.getcwd(), os.pardir, "models", self.name)

        if not os.path.exists(modelPath):
           os.makedirs(modelPath)

        if self.saver is None:
            self.saver = tf.train.Saver()

        file = os.path.join(modelPath, datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + " Epoch-" + str(epoch ) + " Loss-" +str(loss*100))

        self.saver.save(sess, os.path.abspath(file))
        print("Model saved to path: " + os.path.abspath(file))

    def restore(self, sess, file):
        if self.saver is None:
            self.saver = tf.train.Saver()

        modelPath = os.path.join(os.getcwd(), os.pardir, "models", self.name)
        file = os.path.join(modelPath, file)

        self.saver.restore(sess, os.path.abspath(file))
        print("Model restores from path: " + os.path.abspath(file))