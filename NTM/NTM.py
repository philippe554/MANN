import tensorflow as tf
import pandas as pd
import numpy as np

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))