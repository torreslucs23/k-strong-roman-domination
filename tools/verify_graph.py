from itertools import combinations




""""
g receives a graph in the form of a dictionary where each key is a node
and the value is a list containing the node's value and a list of its neighbors.
g = {0: [0, [1, 2]],
      1: [1, [0, 3]],}
"""
def verify_graph(g):
    #Create a list of all attacks in the graph: C(n/2)
    vertex = list(g.keys())
    attacks = list(combinations(vertex, 2))

    for attack in attacks:
        flag = False
        v2_counter = 0
        v2_label = -1
        v3_counter = 0
        if g[attack[0]][0] == 1 or g[attack[0]][0] == 2 or g[attack[0]][0] == 3:
            v3_counter = 1
        else:
            for n1 in g[attack[0]][1]:
                if g[n1][0] == 3:
                    v3_counter += 1
                    break
                if g[n1][0] == 2:
                    v2_counter += 1
                    v2_label = n1
                    if v2_counter > 1:
                        break
        if v2_counter == 0 and v3_counter == 0:
            return False
        

        if g[attack[1]][0] == 1 or g[attack[1]][0] == 2 or g[attack[1]][0] == 3:
            continue
        flag = False
        if g[attack[1]][0] == 1 or g[attack[1]][0] == 2 or g[attack[1]][0] == 3:
            v3_counter = 1
        for n2 in g[attack[1]][1]:
            if g[n2][0] == 3:
                v3_counter += 1
                flag = True
                break
            if g[n2][0] == 2:
                v2_counter += 1
                if n2 != v2_label:
                    flag = True
                    break
                if v2_counter > 2:
                    flag = True
                    break
        if flag == False:
            return False
    return True
        

# Example
g = {0: [2, [1]],
     1: [0, [0, 2]],
     2: [2, [1, 3]],
     3: [0, [2]],
     4: [0, [2, 5]],
     5: [2, [4]]}
print(verify_graph(g))