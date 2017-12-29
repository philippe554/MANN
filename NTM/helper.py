import tensorflow as tf
import numpy as np

def map(name, input, outputSize, r=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=r):
        inputSize = int(input.get_shape()[0])
        m = tf.get_variable(name+"M", initializer=tf.random_normal([inputSize,int(outputSize)]))
        i1 = tf.reshape(input, [1,-1])
        i2 = tf.matmul(i1, m)
        b = tf.get_variable(name+"B", initializer=tf.random_normal(i2.get_shape()))
        i3 = i2 + b
        return tf.reshape(i3, [-1])

def makeStartState(name, shape):
    with tf.variable_scope("init"):
        product = 1
        for i in shape:
            product = product * i

        C = tf.constant([[1]], dtype=tf.float32)
        O = tf.tanh(map(name, C, product, False))
        return tf.reshape(O, shape)

def getNewxy(length, bitDepth):
    data = np.random.randint(2, size=(bitDepth, length))
    x = np.concatenate((np.concatenate((np.zeros((1,length)),data),axis=0),np.ones((bitDepth+1,length))), axis=1)
    y = data
    mask = np.concatenate((np.zeros((length)),np.ones((length))), axis=0)
    return x,y,mask