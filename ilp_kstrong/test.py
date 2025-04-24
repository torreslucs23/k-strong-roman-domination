from itertools import combinations
from ortools.linear_solver import pywraplp
import networkx as nx


G = nx.cycle_graph(6)
V = list(G.nodes())
neighbors = {i: list(G.neighbors(i)) for i in V}


k = 2
attack_patterns = list(combinations(V, k))

print(attack_patterns)