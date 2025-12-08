import matplotlib.pyplot as plt
import networkx as nx
from ortools.linear_solver import pywraplp
import pandas as pd
from itertools import combinations


G = nx.path_graph(10)
V = list(G.nodes())
neighbors = {i: list(G.neighbors(i)) for i in V}
print(neighbors)
pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=15)

# plt.show()

# all atacks patterns with k = 2
k = 2
attack_patterns = list(combinations(V, k))  # lista de tuplas

# solver for create a ilp model
solver = pywraplp.Solver.CreateSolver('SAT')

# contraint with a very large number
M = 3

# all vars used in the model
x = {i: solver.IntVar(0, solver.infinity(), f"x[{i}]") for i in V}
z = {i: solver.BoolVar(f"z[{i}]") for i in V}

# all patterns of movement
y = {}


for h, pattern in enumerate(attack_patterns):
    for j in V:
        for i in neighbors[j]:
            y[h, i, j] = solver.BoolVar(f"y[{h},{i},{j}]")
            print(f"y[{h},{i},{j}]")

# defines z[i] variable
for i in V:
    solver.Add(z[i] <= x[i])
    solver.Add(M * z[i] >= x[i])

# ensures that all vertex attacked can be protected by itself or by a neighbour
for h, pattern in enumerate(attack_patterns):
    for i in pattern:
        neighbors_i = neighbors[i]
        solver.Add(z[i] + solver.Sum(y[h, i, j] for j in neighbors_i) >= 1)

# the number of FAs allocated to each neighbours node cannot be greater than the number of FAs allocated to the node itself
for h, pattern in enumerate(attack_patterns):
    for j in V:
        neighbor_js = neighbors[j]
        sum_y_out = solver.Sum(y[h, i, j] for i in neighbor_js)
        solver.Add(z[j] + sum_y_out <= x[j])

solver.Minimize(solver.Sum(x[i] for i in V))

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Found Solution:")
    for i in V:
        print(f"x[{i}] = {x[i].solution_value()}  | z[{i}] = {z[i].solution_value()}")
    print("Objective value = ", solver.Objective().Value())

else:
    print("Not find solution")