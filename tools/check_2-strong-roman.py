from itertools import combinations
import numpy as np

def create_combinations(nodes):
    return list(combinations(nodes, 2))

def check_combination(graph, attack_a, attack_b):
    flag_a = -1
    if graph[attack_a][1] >= 1:
        flag_a = -2
    else:
        for neighbor in graph[attack_a][0]:
            if graph[neighbor][1] == 2:
                #check if the first vertex attacked has more than 2 neighbors with weight >= 2
                if flag_a != -1:
                    flag_a = -2
                    break
                flag_a = neighbor
                continue
            if graph[neighbor][1] == 3:
                flag_a = -2
                break
    if flag_a == -1:
        return False
    
    #this flag is used to check if the second vertex has more than 2 neighbors with weight >= 2
    flag_b = False
    if graph[attack_b][1] >= 1:
        return True
    for neighbor in graph[attack_b][0]:
        if graph[neighbor][1] == 2:
            if flag_a == -2 or neighbor != flag_a or flag_b == True:
                return True
            else:
                flag_b = True
                continue
        elif graph[neighbor][1] == 3:
            return True
    return False


def check_2_strong_roman(graph):
    nodes = np.array(list(graph.keys()))
    attacks_pairs = create_combinations(nodes)
    for attack_a, attack_b in attacks_pairs:
        if not check_combination(graph, attack_a, attack_b):
            return False
    return True

graph_1 = {
    0: (np.array([1]), 0),
    1: (np.array([1, 2]), 0),
    2: (np.array([1, 3]), 2),
    3: (np.array([2]), 1)
}

# graph_2 = {
#     0: ([1],1),
#     1: ([1,2], 0),
#     2: ([1,3], 2),
#     3: ([2], 1)
# }
print(check_2_strong_roman(graph_1))
        