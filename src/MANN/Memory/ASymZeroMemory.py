import tensorflow as tf
import helper

from MANN.Memory.MemoryBase import *

class ASymZeroMemory(MemoryBase):
    def setup(self, batchSize):
        self.batchSize = batchSize

        if self.batchSize is not None:
            p1 = tf.ones([self.batchSize, 1, self.bitDepth])
            p2 = tf.zeros([self.batchSize, self.length-1, self.bitDepth])
        else:
            p1 = tf.ones([1, self.bitDepth])
            p2 = tf.zeros([self.length - 1, self.bitDepth])

        self.M = [tf.concat([p1, p2], axis=-2)]