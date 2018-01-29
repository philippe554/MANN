import tensorflow as tf
import pandas as pd
import numpy as np
import helper
import random
import time

c = tf.constant([0.1,0.2,0.7,0.3,0.4,0.1])

w = tf.nn.softmax((1-c)*10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W = sess.run(w)

    print(W)
    
            
