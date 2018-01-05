import tensorflow as tf
import numpy as np
import random

def map(name, input, outputSize, r=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=r):
        inputSize = int(input.get_shape()[0])
        m = tf.get_variable(name+"M", initializer=tf.random_normal([inputSize,int(outputSize)]))
        i1 = tf.reshape(input, [1,-1])
        i2 = tf.matmul(i1, m)
        b = tf.get_variable(name+"B", initializer=tf.random_normal(i2.get_shape()))
        i3 = i2 + b
        return tf.reshape(i3, [-1])

def mapBatch(name, input, outputSize, r=tf.AUTO_REUSE):
    if(len(input.get_shape())==1):
        with tf.variable_scope(name, reuse=r):
            inputSize = int(input.get_shape()[0])
            input = tf.expand_dims(input, 0)
            m = tf.get_variable(name+"M", initializer=tf.random_normal([inputSize,int(outputSize)]))
            b = tf.get_variable(name+"B", initializer=tf.random_normal([int(outputSize)]))
            return tf.squeeze(tf.matmul(input, m) + b, [0])
    else:
        with tf.variable_scope(name, reuse=r):
            inputSize = int(input.get_shape()[1].value)
            input = tf.expand_dims(input, 1)
            m = tf.get_variable(name+"M", initializer=tf.random_normal([inputSize,int(outputSize)]))
            b = tf.get_variable(name+"B", initializer=tf.random_normal([int(outputSize)]))
            return tf.squeeze(tf.matmul(input, m) + b, [1])

def makeStartState(name, shape):
    with tf.variable_scope("init"):
        product = 1
        for i in shape:
            product = product * i

        C = tf.constant([[1]], dtype=tf.float32)
        O = tf.tanh(map(name, C, product, False))
        return tf.reshape(O, shape)

def makeStartStateBatch(name, batchSize, shape):
    with tf.variable_scope("init"):
        product = 1
        for i in shape:
            product = product * i

        C = tf.constant([[1]], dtype=tf.float32)
        O = tf.tanh(map(name, C, product, False))
        return tf.expand_dims(tf.ones(shape, tf.float32), 0) * O

def getNewxy(length, bitDepth):
    data = np.random.randint(2, size=(bitDepth, length))
    x = np.concatenate((np.concatenate((np.zeros((1,length)),data),axis=0),np.ones((bitDepth+1,length))), axis=1)
    y = data
    return x,y

def getNewxyLabeled(length, bitDepth):
    x = np.zeros((length * 4, bitDepth))
    y = np.zeros((length, bitDepth))
    for i in range(0, length):
        for j in range(0, bitDepth): 
            y[i,j]=random.randint(0,1)

    for i in range(0, length*2):
        for j in range(0, bitDepth):        
            if(i%2==0):
                x[i,j]=0
            else:
                x[i,j]=y[int(i/2),j]

    for i in range(0, length*2):
        for j in range(0, bitDepth):   
                x[length*2 + i,j]=1
    return x,y

def getNewxyBatch(length, bitDebth, amount):
    x=[]
    y=[]
    for i in range(0,amount):
        X,Y = getNewxy(length, bitDebth)
        x.append(X)
        y.append(Y)
    return x,y
