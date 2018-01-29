import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class RNN:
    def __init__(self, name):
        self.name = name;

    def build(self, x, inputMask=None, outputMask=None, outputSize=None):
        output = []
        states = {}
        if(len(x.get_shape())==3):
            batchSize = tf.shape(x)[0]
        else:
            batchSize = None
        W = []

        inputCounter = 0
        for i in range(0,len(inputMask)):
            print("Building step: "+str(i+1))
            if(inputMask[i] == 0):
                input = tf.squeeze(tf.slice(x, [0,inputCounter,0], [-1,1,-1]),[1])
                inputCounter+=1
            else:
                if inputMask[i] not in states:
                    with tf.variable_scope(self.name):
                        states[inputMask[i]] = self.getTrainableConstant("dummyInput"+str(inputMask[i]), x.get_shape()[-1], batchSize)
                input = states[inputMask[i]]

            O, w = self.buildTimeLayer(input, bool(i==0))

            W.append(tf.expand_dims(w, -2))
        
            if(outputMask[i]==1):
                if(outputSize is not None):
                    with tf.variable_scope(self.name):
                        O = helper.map("outputMap", O, outputSize)
                output.append(tf.expand_dims(O, -2))

        return tf.concat(output, axis=-2), tf.concat(W, axis=-2)
