import networkx as nx
from sklearn.cluster import KMeans
import random

class ForestFire():
    def __init__(self):
        self.G1 = nx.Graph()

    def forestfire(self, G, size):
        list_nodes = list(G.nodes())
        # print(len(G))
        dictt = set()
        random_node = random.sample(set(list_nodes), 1)[0]
        q = set()   # q = set contains the distinct values
        q.add(random_node)
        while(len(self.G1.nodes()) < size):
            if(len(q) > 0):
                initial_node = q.pop()
                if(initial_node not in dictt):
                    dictt.add(initial_node)
                    neighbours = list(G.neighbors(initial_node))
                    np = random.randint(1, len(neighbours))
                    for x in neighbours[:np]:
                        if(len(self.G1.nodes()) < size):
                            self.G1.add_edge(initial_node, x)
                            q.add(x)
                        else:
                            break
                else:
                    continue
            else:
                random_node = random.sample(set(list_nodes) - dictt, 1)[0]
                q.add(random_node)
        q.clear()
        return self.G1