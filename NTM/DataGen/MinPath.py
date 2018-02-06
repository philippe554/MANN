import numpy as np
import random

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
    
def makePathLengthEntry(nodes, edges, maxLength, thinkTime):
    d=-1
    while(d<0 or d>maxLength):
        N,E = genGraph(nodes, edges)
        n1 = random.randint(0,nodes-1)
        n2 = random.randint(0,nodes-1)
        d = getPathLength(N,E, n1, n2)

    X1 = np.zeros([edges, nodes+1])

    for i,e in enumerate(E):
        X1[i,0]=1
        X1[i,e]=1

    X2 = np.zeros([thinkTime, nodes+1])

    X3 = np.ones([1,nodes+1])
    
    X = np.concatenate([X1, X2, X3], axis=-2)

    Y = np.zeros([1, maxLength+1])
    Y[0, d]=1

    return X,Y

def getNewBatch(nodes, edges, maxLength, thinkTime, amount):
    x=[]
    y=[]
    for i in range(0,amount):
        X,Y = makePathLengthEntry(nodes, edges, maxLength, thinkTime)
        x.append(X)
        y.append(Y)
    return x,y