import networkx as nx
from itertools import combinations
import numpy as np

def calculate_ap(G):
    for node, data in G.nodes(data=True):
        if data['weight'] == 0:
            ap = set()
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['weight'] >= 2:
                    ap.add(neighbor)
            G.nodes[node]['ap'] = ap
        else:
            G.nodes[node]['ap'] = set()
    return G

 # we just need to update ap of neighbors when we change a vertex weight
 # Because it doesn't need to be checked in other cases, since it is already protected.
def update_ap_vertex(G, vertex):
    for neighbor in G.neighbors(vertex):
        if G.nodes[neighbor]['weight'] == 0:
            G.nodes[neighbor]['ap'].append(vertex)
    return G
    


def check_roman_2_strong_v2(G):
    G = calculate_ap(G)
    for v in G.nodes():
        if G.nodes[v]['weight'] == 0:
            has_positive_neighbor = False
            for neighbor in G.neighbors(v):
                if G.nodes[neighbor]['weight'] >= 2:
                    has_positive_neighbor = True
                    break
            if not has_positive_neighbor:
                return False
        elif G.nodes[v]['weight'] == 2:
            c = 0
            for neighbor in G.neighbors(v):
                if len(G.nodes[neighbor]['ap']) == 1:
                    c += 1
                if c > 1:
                    return False
    return True

def fix_instance(G):
    G = calculate_ap(G)
    for node, data in G.nodes(data=True):
        if data['weight'] == 0:
            has_positive_neighbor = False
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['weight'] >= 2:
                    has_positive_neighbor = True
                    break
            # We have to check if there is any neighbor with weight 2 or 3
            # If not, we have to increase the weight of this vertex
            if not has_positive_neighbor:
                if len(G.nodes[node]) > 3:
                    G.nodes[node]['weight'] = 3
                    G = update_ap_vertex(G, node)
                else:
                    G.nodes[node]['weight'] = 1
                
        elif data['weight'] == 2:
            c = 0
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['weight'] == 0 and len(G.nodes[neighbor]['ap']) == 1:
                    c += 1
                if c > 1:
                    G.nodes[node]['weight'] = 3
                    break
    return G

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3),
        (3, 4), (4, 0),
        (0, 5), (1, 6),
        (2, 7), (3, 8),
        (4, 9),
        (5, 7), (7, 9),
        (6, 8), (6, 9),
        (5, 8)
    ])
    path_small = nx.petersen_graph()
    

    weights = [0, 2, 0, 0, 0, 2, 2, 0, 0, 1]
    for i in G.nodes():
        print("node: ", i)
        print("neighbors: ", list(G.neighbors(i)))
    for node, weight in enumerate(weights):
        G.nodes[node]['weight'] = weight
    print(sum(i[1]['weight'] for i in fix_instance(G).nodes(data=True)))
    print(G.nodes(data=True))

