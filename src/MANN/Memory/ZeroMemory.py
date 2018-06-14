import tensorflow as tf
import helper

from MANN.Memory.MemoryBase import *

class ZeroMemory(MemoryBase):
    def setup(self, batchSize):
        self.batchSize = batchSize

        if self.batchSize is not None:
            self.M = [tf.zeros([self.batchSize, self.length, self.bitDepth])]
            self.u = [tf.zeros([self.batchSize, self.length])]
        else:
            self.M = [tf.zeros([self.length, self.bitDepth])]
            self.u = [tf.zeros([self.length])]