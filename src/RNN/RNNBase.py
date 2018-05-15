import tensorflow as tf
import numpy as np

import helper

class RNNBase:
    '''
        Base class of all recurrent units. (ex. : LSTM, GRU, FF, MANN)
        Why FF? If FF is used as a controller of the MANN, there are multiple forward calls. 
            While their are independent, they do share parameters and are structured as a RNN.

        This class splits an 2D input (3D if using batches), builds the network for every time point, merges all outputs back to a 2D tensor

        Functions to expand in child classes:
            buildTimeLayer(self, input, first=False): Build the neural network for 1 time step
                   input: 1D (2D if using batches) is the input
                   first: is true if t == 0. This can be used to setup start variables
    '''

    def __init__(self, name):
        self.name = name;

    def build(self, x, outputMask=None):
        '''
            Builds the unit.
            outputMask: array (of size of the amount of time steps and consisting of 0 and 1). If 1, output is of that time step is returned
        '''

        if isinstance(x, (list,)):
            input = x
        else:
            input = tf.unstack(x, x.get_shape()[-2], -2)

        output = []

        for i in range(len(input)):
            helper.progress(i + 1, len(input), status="Building RNN")

            O = self.buildTimeLayer(input[i], bool(i == 0))

            if outputMask == None or outputMask[i] == 1:
                output.append(O)

        if isinstance(x, (list,)):
            return output
        else:
            return tf.stack(output, -2)

