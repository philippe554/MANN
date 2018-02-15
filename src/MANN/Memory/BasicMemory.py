import tensorflow as tf
import helper

from MANN.Memory.MemoryBase import *

class BasicMemory(MemoryBase):
    def __init__(self, name, length, bitDepth, profile="SingleValue", data=0.0):
        super().__init__(name)

        self.length = length
        self.bitDepth = bitDepth
        self.profile = profile
        self.data = data
       
    def setup(self, batchSize):
        self.batchSize = batchSize

        if self.profile == "Trainable":
            with tf.variable_scope(self.name):
                if batchSize is not None:
                    self.M = [tf.reshape(helper.getTrainableConstant("M", self.length * self.bitDepth, batchSize), [-1, self.length, self.bitDepth])]
                else:
                    self.M = [helper.getTrainableConstant("M", self.length * self.bitDepth, None)]

        elif self.profile == "SingleValue":
            if self.batchSize is not None:
                self.M = [tf.fill([batchSize]+[self.length, self.bitDepth], self.data)]
            else:
                self.M = [tf.fill([self.length, self.bitDepth], self.data)]

        elif self.profile == "Data":
            raise NotImplementedError

        else:
            raise ValueError("Choose one of the profiles")