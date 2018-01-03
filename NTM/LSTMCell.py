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

        helper.makeStartState("pLSTMo", [self.outputSize])
        helper.makeStartState("pLSTMs", [self.stateSize])

    def buildTimeLayer(self, input):
        assert(len(input.get_shape()) == 1 and input.get_shape()[0] == self.inputSize)

        with tf.variable_scope(self.name):
            cc = tf.concat([input,self.prevOutput], axis=0)

            forgetGate = tf.sigmoid(helper.map("forgetGate", cc, self.stateSize))
            saveGate = tf.sigmoid(helper.map("saveGate", cc, self.stateSize))           
            outputGate = tf.sigmoid(helper.map("outputGate", cc, self.stateSize))
            update = tf.tanh(helper.map("update", cc, self.stateSize))

            state = (self.prevState * forgetGate) + (saveGate * update)
            output = outputGate * tf.tanh(state)

            assert(len(output.get_shape()) == 1 and output.get_shape()[0] == self.outputSize)
            assert(len(state.get_shape()) == 1 and state.get_shape()[0] == self.stateSize)

            self.prevOutput = output
            self.prevState = state
            return output

    def buildTimeLayerBatch(self, input, first):
        assert(len(input.get_shape()) == 2 and input.get_shape()[1] == self.inputSize)

        if first:
            setupStart(input)

        with tf.variable_scope(self.name):
            cc = tf.concat([input,self.prevOutput], axis=1)

            forgetGate = tf.sigmoid(helper.mapBatch("forgetGate", cc, self.stateSize))
            saveGate = tf.sigmoid(helper.mapBatch("saveGate", cc, self.stateSize))           
            outputGate = tf.sigmoid(helper.mapBatch("outputGate", cc, self.stateSize))
            update = tf.tanh(helper.mapBatch("update", cc, self.stateSize))

            state = (self.prevState * forgetGate) + (saveGate * update)
            output = outputGate * tf.tanh(state)

            assert(len(output.get_shape()) == 2 and output.get_shape()[1] == self.outputSize)
            assert(len(state.get_shape()) == 2 and state.get_shape()[1] == self.stateSize)

            self.prevOutput = output
            self.prevState = state
            return output

    def setupStart(input):
        with tf.variable_scope(self.name):
            self.prevOutput = tf.get_variable("pLSTMo", initializer=tf.random_normal([self.outputSize]))    
            self.prevState = tf.get_variable("pLSTMs", initializer=tf.random_normal([self.stateSize]))

            self.prevOutput = tf.reshape(tf.tile(self.prevOutput, [tf.shape(input)[0]]+[1]), [tf.shape(input)[0]] + [self.outputSize]) 
            self.prevState = tf.reshape(tf.tile(self.prevState, [tf.shape(input)[0]]+[1]), [tf.shape(input)[0]] + [self.stateSize]) 
