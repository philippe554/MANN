import tensorflow as tf
import helper

from MANN.Memory.MemoryBase import *

class WeightMemory(MemoryBase):
    def setup(self, batchSize):
        self.batchSize = batchSize

        self.M = [helper.getBatchWeight("MInit", [self.length, self.bitDepth], self.batchSize)]