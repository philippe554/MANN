import tensorflow as tf
import helper

from MANN.Memory.MemoryBase import *

class ASymZeroMemory(MemoryBase):
    def setup(self, batchSize):
        self.batchSize = batchSize

        if self.batchSize is not None:
            m1 = tf.ones([self.batchSize, 1, self.bitDepth])
            m2 = tf.zeros([self.batchSize, self.length-1, self.bitDepth])

            u1 = tf.ones([self.batchSize, 1])
            u2 = tf.zeros([self.batchSize, self.length - 1])
        else:
            m1 = tf.ones([1, self.bitDepth])
            m2 = tf.zeros([self.length - 1, self.bitDepth])

            u1 = tf.ones([1])
            u2 = tf.zeros([self.length - 1])

        self.M = [tf.concat([m1, m2], axis=-2)]
        self.u = [tf.concat([u1, u2], axis=-1)]