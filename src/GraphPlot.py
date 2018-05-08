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

if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt
    from string import ascii_lowercase

    while True:
        N, E = genGraph(6, 6)

        G = nx.Graph()
        G.add_nodes_from(N)
        G.add_edges_from(E)

        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()