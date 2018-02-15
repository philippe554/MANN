import numpy as np
import random
import tensorflow as tf

from DataGen.DataGenBase import *

class Copy(DataGenBase):
    def __init__(self, length, size):
        self.length = length
        self.size = size

        self.inputLength = self.length*2 +1
        self.inputSize = self.size+1

        self.outputLength = self.length
        self.outputSize = self.size

        self.outputMask = (self.length+1) * [0] + (self.length) * [1]

        self.activationFunc = tf.nn.sigmoid
        self.crossEntropyFunc = tf.nn.sigmoid_cross_entropy_with_logits
    
    def getEntry(self):
        data = np.random.randint(2, size=(self.length, self.size))
        a1 = np.concatenate((np.zeros((self.length,1)),data), axis=1)
        a2 = np.concatenate((a1,np.ones((1,self.size+1))), axis=0)
        x = np.concatenate((a2,np.zeros((self.length,self.size+1))), axis=0)
        y = data

        return x,y,0