import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class LSTMCell:
    def __init__(self, name):
        self.name = name;

    def buildTimeLayer(self, input, prevOutput, prevLSTMState):
        with tf.variable_scope(self.name, reuse=true):
            cc = tf.concat([input,prevOutput], 0)

            forgetGate = tf.sigmoid(map("forgetGate",cc,LSTMState.get_shape()[0]))
            saveGate = tf.sigmoid(map("saveGate",cc,LSTMState.get_shape()[0]))           
            outputGate = tf.sigmoid(map("outputGate",cc,LSTMState.get_shape()[0]))

            update = tf.tanh(map("update",cc,LSTMState.get_shape()[0]))

            LSTMState = (prevLSTMState * forgetGate) + (saveGate * update)
            output = outputGate * tf.tanh(LSTMState)

            return output, LSTMState
