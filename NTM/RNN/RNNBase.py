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

    def build(self, x, outputMask=None, outputSize=None):
        '''
            Builds the unit.
            outputMask: array (of size of the amount of time steps and consisting of 0 and 1). If 1, output is of that time step is returned
            outputSize: if not None, do a linear map to that dimention
        '''

        output = []

        #TODO: Check if unrolling can be optimized using unstack and stack

        #Loop over all the timesteps
        for i in range(0,x.get_shape()[-2]):
            print("Building step: "+str(i+1))

            #Get slice of input, and build network for this time step
            input = tf.squeeze(tf.slice(x, [0,i,0], [-1,1,-1]),[1])
            O = self.buildTimeLayer(input, bool(i==0))
        
            #Process output as defined by parameters
            if(outputMask[i]==1):
                if(outputSize is not None):
                    with tf.variable_scope(self.name):
                        O = helper.map("outputMap", O, outputSize)
                output.append(tf.expand_dims(O, -2))
                
        #Return concated output of all time steps
        return tf.concat(output, axis=-2)
