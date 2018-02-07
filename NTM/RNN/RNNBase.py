import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class RNNBase:
    def __init__(self, name):
        self.name = name;

    def build(self, x, outputMask=None, outputSize=None):
        output = []

        for i in range(0,x.get_shape()[-2]):
            print("Building step: "+str(i+1))
            input = tf.squeeze(tf.slice(x, [0,i,0], [-1,1,-1]),[1])
            O = self.buildTimeLayer(input, bool(i==0))
        
            if(outputMask[i]==1):
                if(outputSize is not None):
                    with tf.variable_scope(self.name):
                        O = helper.map("outputMap", O, outputSize)
                output.append(tf.expand_dims(O, -2))

        return tf.concat(output, axis=-2)#, tf.concat(W, axis=-2)
