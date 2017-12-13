import tensorflow as tf
import pandas as pd
import numpy as np

def map(input, outputSize):
    inputSize = input.get_shape()[0]
    m = tf.Variable(tf.random_normal([int(inputSize),int(outputSize)]))
    i1 = tf.reshape(input, [1,-1])
    i2 = tf.matmul(i1, m)
    return tf.reshape(i2, [-1])

def getWc(k, M, b):
    dot = tf.matmul(M, tf.reshape(k, [-1, 1]))
    l1 = tf.norm(k,axis=0)
    l2 = tf.norm(M,axis=1)
    cosSim = tf.divide(tf.reshape(dot,[-1]), l1 * l2 + 0.001)

    e = tf.exp(b * cosSim)
    return tf.divide(e, tf.reduce_sum(e))

def getWg(wc, g, w_):
    gs = tf.squeeze(g)
    return tf.scalar_mul(gs, wc) + tf.scalar_mul(1-gs, w_)

def getWm(wg, s):
    size = int(wg.get_shape()[0])
    shiftSize = int(int(s.get_shape()[0])/2)

    def shift(i):
        if(i<0):
            return size+i
        if(i>=size):
            return i-size
        return i

    def indices(i):
        indices = [shift(i+j) for j in range(shiftSize,-shiftSize-1,-1)]
        return tf.reduce_sum(tf.gather(wg, indices) * s,0)

    return tf.dynamic_stitch(list(range(0,size)), [indices(i) for i in range(0,size)])

def getW(wm,y):
    pow = tf.pow(wm, y)
    return  pow / tf.reduce_sum(pow)

M = tf.Variable(tf.random_normal([8,4]))
O = tf.Variable(tf.random_normal([20]))

w_ = tf.constant([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0], shape=[1, 8]) # N

k = tf.tanh(map(O, M.get_shape()[1]))
b = tf.nn.softplus(map(O, 1))
g = tf.sigmoid(map(O, 1))
s = tf.nn.softmax(map(O, 5))
y = tf.nn.softplus(map(O, 1)) + 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

wc = getWc(k, M, b)
wg = getWg(wc, g, w_)
wm = getWm(wg, s)
w = getW(wm, y)

print(sess.run(w))