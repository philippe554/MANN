import tensorflow as tf
import pandas as pd
import numpy as np
import NTMCell

def makeNTM(x):
    steps = x.get_shape()[1]
    
    prevLSTMOutput = makeStartState([8])
    prevLSTMState = makeStartState([4])
    prevRead = makeStartState([4])
    M = makeStartState([8, 4])
    wRead = makeStartState([8])
    wWrite = makeStartState([8])

    ntmCell = NTMCell("ntm")

    output = []

    for i in range(0:steps):
        input = tf.slice(x,[0,i],[x.get_shape()[0],1])
        prevLSTMOutput, prevLSTMState, O, prevRead, M, wRead, wWrite = ntmCell.buildTimeLayer(input, prevLSTMOutput, prevLSTMState, prevRead, M, wRead, wWrite)
        output.append(O)

    result = tf.concat(output, 1)

M = tf.Variable(tf.random_normal([8,4]))
O = tf.Variable(tf.random_normal([20]))
w_ = tf.constant([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0], shape=[8]) # N

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(r))
print(sess.run(M))