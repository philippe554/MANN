import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class HeadBase:
    '''
        Base class of all read and write heads

        Functions to expand in child classes:
            setupStartVariables(self): setup all start variables
            getWW(self, O): calculate the w of the write operation
            getWR(self, O): calculate the w of the read operation
    '''

    def __init__(self, name, amountReadHeads=1):
        self.name = name
        self.amountReadHeads = amountReadHeads

    def setup(self, batchSize, memory):
        '''
            Setup all start variables of this head, can not be called in the __init__ functiom because batch size is still unknown
            The MANNUnit Calls it after it receives the first input
        '''

        self.batchSize = batchSize
        self.memory = memory

        self.readList = [tf.zeros([self.batchSize, self.memory.bitDepth * self.amountReadHeads])]

        with tf.variable_scope(self.name):
            with tf.variable_scope("init"):
                self.setupStartVariables()

    def buildWriteHead(self, O):
        '''
            Build the write head: get the W from the child class and add the operation to the memory queue
            This queue is nececairy if there are multiple write operations/heads. Otherwise the second write is based on the first write 
        '''

        with tf.variable_scope(self.name):
            with tf.variable_scope("write"):
                self.wWriteList.append(self.getWW(O))

                erase = tf.sigmoid(helper.map("map_erase", O, self.memory.bitDepth))
                write = helper.map("map_write", O, self.memory.bitDepth)
                
                self.memory.queueWrite(self.wWriteList[-1], erase, write)
    
    def buildReadHead(self, O):
        '''
            Build the read head: get the W from the child class, read the value from memory and add it to the read list.
        '''

        with tf.variable_scope(self.name):
            with tf.variable_scope("read"):
                self.wReadList.append(self.getWR(O))

                self.readList.append(self.memory.read(self.wReadList[-1]))

    def getCosSimSoftMax(self, k, b):
        '''
            Calculate the cosine between a head and a memory
        '''

        assert helper.check(k, [self.memory.bitDepth], self.batchSize)
        assert helper.check(self.memory.M[-1], [self.memory.length, self.memory.bitDepth], self.batchSize)
        assert helper.check(b, [1], self.batchSize)

        dot = tf.squeeze(tf.matmul(self.memory.M[-1], tf.expand_dims(k, axis=-1)), axis=-1)
        l1 = tf.sqrt(tf.reduce_sum(tf.pow(k, 2), axis=-1, keep_dims=True))
        l2 = tf.sqrt(tf.reduce_sum(tf.pow(self.memory.M[-1], 2), axis=-1))
        cosSim = tf.divide(dot, l1 * l2 + 0.001)

        result = tf.nn.softmax((b * cosSim) + 0.001)
        assert helper.check(result, [self.memory.length], self.batchSize)

        return result

    def getCosSimSoftMaxExtra(self, k, b, extra):
        '''
            Calculate if there are multiple reading head
            TODO: merge with function above
        '''

        assert helper.check(k, [extra, self.memory.bitDepth], self.batchSize)
        assert helper.check(self.memory.M[-1], [self.memory.length, self.memory.bitDepth], self.batchSize)
        assert helper.check(b, [extra], self.batchSize)

        dot = tf.matmul(self.memory.M[-1], k, transpose_b=True)
        assert helper.check(dot, [self.memory.length, extra], self.batchSize)

        l1 = tf.sqrt(tf.reduce_sum(tf.pow(k, 2), axis=-1, keep_dims=True))
        l2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(self.memory.M[-1], 2), axis=-1)), axis=-2)
        cosSim = tf.divide(tf.transpose(dot, perm=[0,2,1]), tf.matmul(l1, l2) + 0.001)
        assert helper.check(cosSim, [extra, self.memory.length], self.batchSize)

        result = tf.nn.softmax((tf.expand_dims(b, axis=-1) * cosSim) + 0.001)
        assert helper.check(result, [extra, self.memory.length], self.batchSize)

        return result



