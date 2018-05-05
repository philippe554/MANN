import numpy as np
import os

from DataGen.DataGenBase import *

class VertexCover(DataGenBase):
    def __init__(self, nodes, edges, thinkTime, possibleRightAnswer):
        self.name = "VertexCover"

        self.nodes = nodes
        self.edges = edges
        self.thinkTime = thinkTime

        self.inputLength = self.edges + self.thinkTime + 1
        self.inputSize = self.nodes + 1

        self.outputLength = 1
        self.outputSize = self.nodes

        self.outputMask = (self.edges + self.thinkTime) * [0] + 1 * [1]

        self.possibleRightAnswer = possibleRightAnswer

        self.postBuildMode = "sigmoid_custom"

    def makeDataset(self, amount, token):
        file = os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, os.pardir, "data", self.name, token + "RawVertexCover.csv")
        raw = np.genfromtxt(os.path.abspath(file), delimiter=',', dtype=int)

        x = []
        y = []
        c = {}
        for i in range(min(raw.shape[0], amount)):
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

        Y = row[1 + 2*self.edges + 1:].reshape([self.possibleRightAnswer, 1, self.nodes])

        return X, Y, 0

    def getLabel(self):
        return tf.placeholder(tf.float32, shape=(None, self.possibleRightAnswer, self.outputLength, self.outputSize))

    def customPostBuild(self, _y, y, optimizer):
        assert helper.check(_y, [self.possibleRightAnswer, self.outputLength, self.outputSize], 100)
        assert helper.check(y, [self.outputLength, self.outputSize], 100)

        yy = tf.expand_dims(y, axis=-2)
        assert helper.check(yy, [1, self.outputLength, self.outputSize], 100)

        sq = tf.square(tf.subtract(yy, _y))
        assert helper.check(sq, [self.possibleRightAnswer, self.outputLength, self.outputSize], 100)

        distance = tf.sqrt(tf.reduce_sum(tf.reduce_sum(sq, axis=-1), axis=-1))
        assert helper.check(distance, [self.possibleRightAnswer], 100)

        indices = tf.argmin(distance, axis=-1)
        assert helper.check(indices, [], 100)

        num_examples = tf.cast(tf.shape(y)[0], dtype=indices.dtype)
        indices = tf.stack([tf.range(num_examples), indices], axis=-1)
        _Y = tf.gather_nd(_y, indices)
        assert helper.check(_Y, [self.outputLength, self.outputSize], 100)

        _Y = tf.stop_gradient(_Y)

        crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=_Y, logits=y)
        loss = tf.reduce_mean(crossEntropy)

        grads_and_vars = optimizer.compute_gradients(loss)
        trainStep = optimizer.apply_gradients(grads_and_vars)

        p = tf.round(tf.nn.sigmoid(y))
        accuracy = tf.reduce_mean(tf.reduce_min(tf.cast(tf.equal(_Y, p), tf.float32), axis=-1))

        return trainStep, p, accuracy, loss

    def getCoveredSet(self, g, c):
        covered = np.zeros([self.nodes], dtype=np.int)

        for i in range(self.nodes):
            if c[i] == 1:
                covered[i] = 1

                for j in range(self.edges):
                    if g[j, 0] == i:
                        covered[g[j, 1]] = 1

                    if g[j, 1] == i:
                        covered[g[j, 0]] = 1

        return covered

    def getAmountUncovered(self, g, c):
        covered = self.getCoveredSet(g, c)

        return self.nodes - np.sum(covered)

    def isVertexCover(self, g, c):
        covered = self.getCoveredSet(g, c)

        return np.sum(covered) == self.nodes

    def convertToGraph(self, x):
        x = x[0:self.edges, 0:self.nodes]
        e = []
        for i in range(self.edges):
            e.append(np.argsort(x[i, :]*-1)[0:2])

        return np.array(e)

    def process(self, X, Y, R):
        X = np.rint(np.array(X))
        Y = np.rint(np.array(Y))
        R = np.rint(np.array(R))

        totalOptimal = 0
        subOptimal = 0
        noCover = np.zeros([self.nodes+1], dtype=np.int)
        coverSizeFound = 0


        for i in range(Y.shape[0]):
            x = X[i]
            y = Y[i]
            r = R[i]

            optimalFound = False
            for j in range(y.shape[0]):
                if np.allclose(r, y[j]):
                    optimalFound = True

            g = self.convertToGraph(x)

            if optimalFound:
                totalOptimal += 1
            elif self.isVertexCover(g, r[0]):
                subOptimal += 1

            uncoveredSize = self.getAmountUncovered(g, r[0])
            noCover[uncoveredSize] += 1

            if np.sum(r) == np.sum(y[0]):
                coverSizeFound += 1

        out = "Total: " + str(Y.shape[0])
        out += " ║ Optimal: " + helper.strfixed(totalOptimal, 3)
        out += " ║ Sub optimal: " + helper.strfixed(subOptimal, 3)
        out += " ║ Size found: " + helper.strfixed(coverSizeFound, 3)
        out += " ║ Uncovered dist.: " + str(noCover)

        return out


