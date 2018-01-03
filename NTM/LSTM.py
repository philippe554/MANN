import tensorflow as tf
import pandas as pd
import numpy as np
from NTMCell import *
import helper

x = tf.get_variable("x", initializer=tf.random_normal([2,3]))
x = tf.reshape(tf.tile(x, [3]+[1]), [3] + [2,3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(x))
