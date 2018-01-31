import tensorflow as tf

class MemoryPrototype:
    def __init__(self, name):
        self.name = name
        self.M = []

    def getLast(self):
        if len(M) == 0:
            return self.startMemory
        else:
            return self.M[-1]

    def new(self, M):
        self.M.append(M)


