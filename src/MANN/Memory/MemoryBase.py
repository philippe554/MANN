import tensorflow as tf

import helper

class MemoryBase:
    '''
        This is the base class for all sorts of memory
        
        Functions to expand in child classes:
            __init__(self, ...): Setup the parameters
            setup(self, batchSize): Setup the start variables
    '''

    def __init__(self, name):
        self.name = name
        self.ops = []

    def queueWrite(self, w, erase, add):
        assert helper.check(w, [self.length], self.batchSize)
        assert helper.check(erase, [self.bitDepth], self.batchSize)
        assert helper.check(add, [self.bitDepth], self.batchSize)

        self.ops.append({'w':w,'e':erase,'a':add})

    def write(self):
        if len(self.ops) == 1:
            erase = 1 - tf.matmul(tf.expand_dims(self.ops[0]['w'], axis=-1),tf.expand_dims(self.ops[0]['e'], axis=-2))
            add = tf.matmul(tf.expand_dims(self.ops[0]['w'], axis=-1),tf.expand_dims(self.ops[0]['a'], axis=-2))

            self.M.append(self.M[-1] * erase + add)
        else:
            erase = tf.ones([self.batchSize, self.length, self.bitDepth])
            add = tf.zeros([self.batchSize, self.length, self.bitDepth])

            for op in self.ops:
                erase *= 1 - tf.matmul(tf.expand_dims(op['w'], axis=-1),tf.expand_dims(op['e'], axis=-2))
                add += tf.matmul(tf.expand_dims(op['w'], axis=-1),tf.expand_dims(op['a'], axis=-2))

            self.M.append(self.M[-1] * erase + add)

        self.ops = []

    def read(self, w):
        if len(w.get_shape())==2:
            assert helper.check(w, [self.length], self.batchSize)
            assert helper.check(self.M[-1], [self.length, self.bitDepth], self.batchSize)

            r = tf.squeeze(tf.matmul(tf.expand_dims(w,axis=-2), self.M[-1]),axis=-2)
            assert helper.check(r, [self.bitDepth], self.batchSize)

            return r
        else:
            multiple = w.get_shape()[1]

            assert helper.check(w, [multiple, self.length], self.batchSize)
            assert helper.check(self.M[-1], [self.length, self.bitDepth], self.batchSize)

            r = tf.matmul(w, self.M[-1])
            assert helper.check(r, [multiple, self.bitDepth], self.batchSize)

            r = tf.reshape(r, [self.batchSize, multiple * self.bitDepth])
            assert helper.check(r, [multiple * self.bitDepth], self.batchSize)

            return r
            

