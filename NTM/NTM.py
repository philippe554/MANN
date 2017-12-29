import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
import helper

def makeNTM(x, mask):
    steps = x.get_shape()[1]
    
    prevLSTMOutput = helper.makeStartState("pLSTMo", [4])
    prevLSTMState = helper.makeStartState("pLSTMs", [4])
    prevRead = helper.makeStartState("pr", [4])
    M = helper.makeStartState("m", [8, 4])
    wRead = helper.makeStartState("wr", [8])
    wWrite = helper.makeStartState("ww", [8])

    ntmCell = NTMCell("ntm")

    output = []

    for i in range(0,steps):
        print("Building step: "+str(i+1))
        input = tf.squeeze(tf.slice(x,[0,i],[x.get_shape()[0],1]))
        prevLSTMOutput, prevLSTMState, O, prevRead, M, wRead, wWrite = ntmCell.buildTimeLayer(input, prevLSTMOutput, prevLSTMState, prevRead, M, wRead, wWrite, 4)
        
        if(mask[i]==1):
            output.append(O)

    return tf.concat(output, 1)

X,Y,mask = helper.getNewxy(8,4)

x = tf.placeholder(tf.float32, shape=(4, 16))
y = tf.placeholder(tf.float32, shape=(4, 16))

y = makeNTM(x, mask)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

#rand_array = np.random.rand(1024, 1024)
#print(sess.run(y, feed_dict={x: rand_array}))