import tensorflow as tf
import pandas as pd
import numpy as np

import helper

class LSTMCell:
    def __init__(self, name, inputSize, outputSize, stateSize):
        self.name = name;
        assert(outputSize == stateSize) #Just for now
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.stateSize = stateSize

    def buildTimeLayer(self, input, prevOutput, prevLSTMState):
        assert(len(input.get_shape()) == 1 and input.get_shape()[0] == self.inputSize)
        assert(len(prevOutput.get_shape()) == 1 and prevOutput.get_shape()[0] == self.outputSize)
        assert(len(prevLSTMState.get_shape()) == 1 and prevLSTMState.get_shape()[0] == self.stateSize)

        with tf.variable_scope(self.name):
            cc = tf.concat([input,prevOutput], axis=0)

            forgetGate = tf.sigmoid(helper.map("forgetGate", cc, self.stateSize))
            saveGate = tf.sigmoid(helper.map("saveGate", cc, self.stateSize))           
            outputGate = tf.sigmoid(helper.map("outputGate", cc, self.stateSize))

            update = tf.tanh(helper.map("update", cc, prevLSTMState.get_shape()[0]))

            LSTMState = (prevLSTMState * forgetGate) + (saveGate * update)
            output = outputGate * tf.tanh(LSTMState)

            assert(len(output.get_shape()) == 1 and output.get_shape()[0] == self.outputSize)
            assert(len(LSTMState.get_shape()) == 1 and LSTMState.get_shape()[0] == self.stateSize)

            return output, LSTMState
