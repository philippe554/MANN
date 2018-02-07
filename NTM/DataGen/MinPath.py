import numpy as np
import random
import tensorflow as tf

from DataGen.DataGenBase import *

def genGraph(nodes, edges):
    N = list(range(0,nodes))
    E = []

    while len(E) < edges:
        n1 = random.randint(0,nodes-1)
        n2 = random.randint(0,nodes-1)
        if(n1 != n2 and [n1, n2] not in E and [n2, n1] not in E):
            E.append([n1, n2])

    return N,E

def getPathLength(N, E, n1, n2):
    D = [-1] * len(N)
    D[n1] = 0
    goto = [n1]

    while len(goto)>0:
        for e in E:
            if(goto[0] == e[0]):
                if(D[e[1]]>D[e[0]]+1 or D[e[1]]==-1):
                    D[e[1]] = D[e[0]]+1
                    goto.append(e[1])
            if(goto[0] == e[1]):
                if(D[e[0]]>D[e[1]]+1 or D[e[0]]==-1):
                    D[e[0]] = D[e[1]]+1
                    goto.append(e[0])
        goto.pop(0)

    return D[n2]

class MinPath(DataGenBase):
    def __init__(self, nodes, edges, maxLength, thinkTime):
        self.nodes=nodes
        self.edges=edges
        self.maxLength=maxLength
        self.thinkTime=thinkTime

        self.inputLength = self.edges+self.thinkTime+1
        self.inputSize = self.nodes+1

        self.outputLength = 1
        self.outputSize = self.maxLength+1

        self.outputMask = (self.edges+self.thinkTime) * [0] + 1 * [1]

        self.activationFunc = tf.nn.softmax
        self.crossEntropyFunc = tf.nn.softmax_cross_entropy_with_logits
    
    def getEntry(self):
        d=-1
        target = random.randint(1, self.maxLength)
        #while(d<1 or d>self.maxLength):
        while d != target:
            N,E = genGraph(self.nodes, self.edges)
            n1 = random.randint(0,self.nodes-1)
            n2 = random.randint(0,self.nodes-1)
            d = getPathLength(N, E, n1, n2)

        X1 = np.zeros([self.edges, self.nodes+1], dtype=float)

        for i,e in enumerate(E):
            X1[i, -1] = 1.0
            X1[i, e] = 1.0

        X2 = np.zeros([self.thinkTime, self.nodes+1], dtype=float)

        X3 = np.zeros([1, self.nodes+1], dtype=float)
        X3[0, n1] = 1.0
        X3[0, n2] = 1.0
        X3[0, -1] = 1.0
    
        X = np.concatenate([X1, X2, X3], axis=-2)

        Y = np.zeros([1, self.maxLength+1], dtype=float)
        Y[0, d]=1.0

        return X,Y,d
