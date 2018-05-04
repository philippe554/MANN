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

    modelPath = os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\models\\"
    dataPath = os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\data\\"

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
        if not os.path.exists(self.dataPath + self.name + "\\"):
            os.makedirs(self.dataPath + self.name + "\\")

        file = self.dataPath + self.name + "\\" + str(token)+"-"+str(amount)+".p"

        try:
            return pickle.load(open(file,"rb"))
        except:
            data = self.makeDataset(amount, token)
            pickle.dump(data, open(file, "wb"))
            return data

    def getInput(self):
        return tf.placeholder(tf.float32, shape=(None, self.inputLength, self.inputSize))

    #def getLabel(self):
    #    return tf.placeholder(tf.float32, shape=(None, self.outputLength, self.outputSize))

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
        if not os.path.exists(self.modelPath + self.name + "\\"):
            os.makedirs(self.modelPath + self.name + "\\")

        if self.saver is None:
            self.saver = tf.train.Saver()

        file = self.modelPath + self.name + "\\" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + " Epoch-" + str(epoch) + " Loss-" + str(loss) + ".ckpt"
       
        self.saver.save(sess, file)
        print("Model saved to path: " + file)

    def restore(self, sess, file):
        if self.saver is None:
            self.saver = tf.train.Saver()

        file = self.modelPath + self.name + "\\" + file

        self.saver.restore(sess, file)
        print("Model restores from path: " + file)