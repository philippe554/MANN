import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN.RNNBase import *

class MANNUnit(RNNBase):
    def __init__(self, name):
        super().__init__(name)
        self.controllers = []
        self.heads = []
        self.memory = None

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.setup(input)

            prevReads = [head.readList[-1] for head in self.heads]
            O = tf.concat([input]+prevReads, axis=-1)

            for controller in self.controllers:
                O = controller.buildTimeLayer(O, first)
                
            #All memory changes are queued up, so they dont inflouence each other
            for head in self.heads:
                head.buildWriteHead(O)

            #Apply all memory operations at once
            self.memory.write()

            for head in self.heads:
                head.buildReadHead(O)

            return O

    def setup(self, firstInput):
        if self.memory is None:
            raise ValueError("No memory added")

        if(len(firstInput.get_shape())==2):
            self.batchSize = tf.shape(firstInput)[0]
        else:
            self.batchSize = None

        self.memory.setup(self.batchSize)

        for head in self.heads:
            head.setup(self.batchSize, self.memory)
        
    def addMemory(self, memory):
        self.memory = memory

    def addController(self, controller):
        self.controllers.append(controller)

    def addHead(self, head):
        self.heads.append(head)
