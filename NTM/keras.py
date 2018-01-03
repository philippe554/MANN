import tensorflow as tf
from keras import backend as K
import pandas as pd
import numpy as np
from NTMCell import *
import helper

sess = tf.Session()
K.set_session(sess)
