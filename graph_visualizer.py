from pyvis.network import Network
import networkx as nx


def add_graph(dims, file):
    count = 0
    G = nx.Graph()
    for l in f:
        words = l.split(', ')
        G.add_edge(words[0], words[1], weight=1, relation='co-exist')
        count+=1

    nt = Network(str(dims[0]) + 'px', str(dims[1]) + 'px')
    nt.from_nx(G)
    nt.repulsion(180, 0.5, 280)
    nt.show('gtsam.html')
    return count


f = open('Other/data.txt')
total = add_graph(dims=(2000, 2000), file=f)
print("Total:", total)
