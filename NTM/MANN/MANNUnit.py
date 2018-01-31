import tensorflow as tf
import pandas as pd
import numpy as np

import helper
from RNN.RNN import *

class MANNUnit(RNN):
    def __init__(self, name):
        super().__init__(name)
        self.readHeads = []
        self.writeHeads = []
        self.memory = None
        self.controller = None

    def buildTimeLayer(self, input, first=False):
        with tf.variable_scope(self.name):
            if first:
                self.setup(input)

            prevReads = [head.getLastRead() for head in heads]

            O = self.controller.buildTimeLayer(tf.concat([input]+prevReads, axis=-1), first)

            for head in self.readHeads:
                head.buildRead(self.memory, O)

            for head in self.writeHeads:
                head.buildWrite(self.memory, O)

            return O

    def setup(self, firstInput):
        if self.memory is None:
            raise ValueError("No memory added")
        if self.controller is None:
            raise ValueError("No controller added")

        if(len(firstInput.get_shape())==2):
            self.batchSize = tf.shape(firstInput)[0]
        else:
            self.batchSize = None

        with tf.variable_scope("init"):
            self.memory.setup(self.batchSize)
            for head in self.readHeads:
                head.setup(self.batchSize, self.memory.bitDepth, self.memory.length, self.controller.stateSize)
            for head in self.writeHeads:
                head.setup(self.batchSize, self.memory.bitDepth, self.memory.length, self.controller.stateSize)
        
    def addMemory(self, memory):
        self.memory = memory

    def addController(self, controller):
        self.controller = controller

    def addHead(self, head):
        if head.mode == "Read":
            self.readHeads.append(head)
        else:
            self.writeHeads.append(head)
