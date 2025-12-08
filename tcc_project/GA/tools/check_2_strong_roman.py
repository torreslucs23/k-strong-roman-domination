import networkx as nx
from itertools import combinations
import numpy as np

def create_graph_from_dict(graph_dict):
    G = nx.Graph()
    for node, (neighbors, weight) in graph_dict.items():
        G.add_node(node, weight=weight)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G


def calculate_power(G):
    for node, data in G.nodes(data=True):
        weight = data["weight"]
        power_value = max(0, weight - 1)
        data["power"] = power_value
    return G

        

# mudar depois de lista para set tanto aqui quanto na função de update
def calculate_ap(G):
    for node, data in G.nodes(data=True):
        ap = []
        if data['weight'] == 0:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['power'] >= 1:
                    ap.append(neighbor)
            G.nodes[node]['ap'] = ap
            continue
        G.nodes[node]['ap'] = ap
    return G

 # we just need to update ap of neighbors when we change a vertex weight
 # Because it doesn't need to be checked in other cases, since it is already protected.
def update_ap_vertex(G, vertex):
    for neighbor in G.neighbors(vertex):
        if G.nodes[neighbor]['weight'] == 0:
            G.nodes[neighbor]['ap'].append(vertex)
    return G
    


def check_roman_2_strong(G):
    G = calculate_power(G)
    G = calculate_ap(G)
    for node, data in G.nodes(data=True):
        if data['weight'] == 0:
            has_positive_neighbor = False
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['power'] >= 1:
                    has_positive_neighbor = True
                    break
            if not has_positive_neighbor:
                return False
        elif data['weight'] == 2:
            c = 0
            for neighbor in G.neighbors(node):
                if len(G.nodes[neighbor]['ap']) == 1:
                    c += 1
                if c > 1:
                    return False
    return True

def fix_instance(G):
    G = calculate_power(G)
    G = calculate_ap(G)
    for node, data in G.nodes(data=True):
        if data['weight'] == 0:
            has_positive_neighbor = False
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['power'] >= 1:
                    has_positive_neighbor = True
                    break
            # Precisamos testar o grau do vizinho
            # Se o grau for acima de 2 ou 3, rotulamos o peso com 3 e atualizamos os aps dos vizinhos
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
    

    # graph_1_dict = {
    #     0: (np.array([1]), 0),
    #     1: (np.array([0, 2]), 2),
    #     2: (np.array([1, 3]), 0),
    #     3: (np.array([2]), 3),
    # }

    # g1 = create_graph_from_dict(graph_1_dict)
    # g1 = fix_instance(g1)

    # print(g1.nodes(data=True))
    
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
    path_small = nx.path_graph(10)
    

    weights = [0, 0, 0, 0, 0, 0, 0, 3, 3, 3]
    for i in G.nodes():
        print("node: ", i)
        print("neighbors: ", list(G.neighbors(i)))
    for node, weight in enumerate(weights):
        G.nodes[node]['weight'] = weight
    print([i[1]['weight'] for i in fix_instance(G).nodes(data=True)])
    print(G.nodes(data=True))
    # print(calculate_power(g1).nodes(data=True))
    # g1 = calculate_power(g1)
    # print(calculate_ap(g1).nodes(data=True))

    # print(check_linear(g1))

