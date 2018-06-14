import tensorflow as tf

import helper

class MemoryBase:
    '''
        This is the base class for all sorts of memory
        
        Functions to expand in child classes:
            __init__(self, ...): Setup the parameters
            setup(self, batchSize): Setup the start variables
    '''

    def __init__(self, name, length, bitDepth):
        self.name = name
        self.length = length
        self.bitDepth = bitDepth
        self.ops = []
        self.forgetQueue = []

    def queueForget(self, v):
        if len(v.get_shape()) == 2:
            assert helper.check(v, [self.length], self.batchSize)

            self.forgetQueue.append(tf.expand_dims(v, -2))

        elif len(v.get_shape()) == 3:
            assert helper.check(v, [v.get_shape()[1], self.length], self.batchSize)

            self.forgetQueue.append(v)

    def queueWrite(self, w, erase, add):
        assert helper.check(w, [self.length], self.batchSize)
        assert helper.check(erase, [self.bitDepth], self.batchSize)
        assert helper.check(add, [self.bitDepth], self.batchSize)

        self.ops.append({'w':w,'e':erase,'a':add})

    def runQueued(self):
        u = self.forget()

        self.write(u)

    def forget(self):
        if len(self.forgetQueue) == 0:
            u = self.u[-1]
        elif len(self.forgetQueue) == 1:
            if self.forgetQueue[0].get_shape()[-2] == 1:
                u = self.u[-1] * tf.squeeze(self.forgetQueue[0], axis=-2)
            else:
                u = self.u[-1] * tf.reduce_prod(self.forgetQueue[0], axis=-2)
        else:
            u = self.u[-1] * tf.reduce_prod(tf.concat(self.forgetQueue, axis=-2), axis=-2)

        self.forgetQueue = []

        assert helper.check(u, [self.length], self.batchSize)
        return u

    def write(self, u):
        assert helper.check(u, [self.length], self.batchSize)

        if len(self.ops) == 1:
            erase = 1 - tf.matmul(tf.expand_dims(self.ops[0]['w'], axis=-1),tf.expand_dims(self.ops[0]['e'], axis=-2))
            add = tf.matmul(tf.expand_dims(self.ops[0]['w'], axis=-1),tf.expand_dims(self.ops[0]['a'], axis=-2))

            self.M.append(self.M[-1] * erase + add)

            u = u + self.ops[0]['w'] - (u*self.ops[0]['w'])

            assert helper.check(u, [self.length], self.batchSize)
            self.u.append(u)

        else:
            erase = tf.ones([self.batchSize, self.length, self.bitDepth])
            add = tf.zeros([self.batchSize, self.length, self.bitDepth])

            for op in self.ops:
                erase *= 1 - tf.matmul(tf.expand_dims(op['w'], axis=-1),tf.expand_dims(op['e'], axis=-2))
                add += tf.matmul(tf.expand_dims(op['w'], axis=-1),tf.expand_dims(op['a'], axis=-2))

                u = u + op['w'] - (u * op['w'])

            self.M.append(self.M[-1] * erase + add)

            assert helper.check(u, [self.length], self.batchSize)
            self.u.append(u)

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

    def getU(self):
        return self.u[-1]




            

