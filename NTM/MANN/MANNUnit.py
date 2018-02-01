import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN.RNNBase import *

class MANNUnit(RNNBase):
    def __init__(self, name):
        super().__init__(name)
        self.controllers = []
        self.readHeads = []
        self.writeHeads = []
        self.memory = None

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.setup(input)

            prevReads = [head.getLastRead() for head in self.readHeads]
            O = tf.concat([input]+prevReads, axis=-1)

            for controller in self.controllers:
                O = controller.buildTimeLayer(O, first)

            for head in self.readHeads:
                head.buildHead(self.memory, O)

            for head in self.writeHeads:
                head.buildHead(self.memory, O)

            return O

    def setup(self, firstInput):
        if self.memory is None:
            raise ValueError("No memory added")

        if(len(firstInput.get_shape())==2):
            self.batchSize = tf.shape(firstInput)[0]
        else:
            self.batchSize = None

        self.memory.setup(self.batchSize)

        for head in self.readHeads:
            head.setup(self.batchSize, self.memory.bitDepth, self.memory.length)

        for head in self.writeHeads:
            head.setup(self.batchSize, self.memory.bitDepth, self.memory.length)
        
    def addMemory(self, memory):
        self.memory = memory

    def addController(self, controller):
        self.controllers.append(controller)

    def addHead(self, head):
        if head.mode == "Read":
            self.readHeads.append(head)
        else:
            self.writeHeads.append(head)
