import numpy as np

from DataGen.DataGenBase import *

class VertexCover(DataGenBase):
    def __init__(self, nodes, edges, thinkTime):
        self.name = "VertexCover"

        self.nodes = nodes
        self.edges = edges
        self.thinkTime = thinkTime

        self.inputLength = self.edges + self.thinkTime + 1
        self.inputSize = self.nodes + 1

        self.outputLength = 1
        self.outputSize = self.nodes

        self.outputMask = (self.edges + self.thinkTime) * [0] + 1 * [1]

        self.possibleRightAnswer = 30

        self.postBuildMode = "sigmoid"

    def makeDataset(self, amount, token):
        file = self.dataPath + self.name + "\\" + str(token) + "-rawVertexCover.csv"
        raw = np.genfromtxt(file, delimiter=',', dtype=int)

        x = []
        y = []
        c = {}
        for i in range(raw.shape[0]):
            helper.progress(i + 1, amount, status="Creating dataset of size " + str(raw.shape[0]))
            X, Y, C = self.getEntry(raw[i, :])
            x.append(X)
            y.append(Y)
            if C in c:
                c[C] += 1
            else:
                c[C] = 0
        return Data(x, y, c)

    def getEntry(self, row):
        E = row[1:1+self.edges*2].reshape([self.edges, 2])

        X1 = np.zeros([self.edges, self.nodes+1], dtype=float)
        for i in range(self.edges):
            X1[i, -1] = 1.0
            X1[i, E[i, 0]] = 1.0
            X1[i, E[i, 1]] = 1.0

        X2 = np.zeros([self.thinkTime, self.nodes + 1], dtype=float)

        X3 = np.zeros([1, self.nodes + 1], dtype=float)
        X3[0, -1] = 1.0

        X = np.concatenate([X1, X2, X3], axis=-2)

        Y = row[1 + self.edges + 1:].reshape([self.possibleRightAnswer, 1, self.nodes])

        return X, Y, 0