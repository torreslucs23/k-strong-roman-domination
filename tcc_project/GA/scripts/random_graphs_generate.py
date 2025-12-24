import os
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "benchmarks", "random")

os.makedirs(OUT_DIR, exist_ok=True)

sizes = [10, 15, 20, 30, 45, 50, 100]
densities = [0.1, 0.2, 0.3, 0.4, 0.5]

seed = 42

def save_graph(G, path):
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

for n in sizes:
    for p in densities:
        G = nx.gnp_random_graph(n, p, seed=seed)
        G.remove_edges_from(nx.selfloop_edges(G))

        name = f"random_n{n}_p{p}.txt"
        save_graph(G, os.path.join(OUT_DIR, name))

print("Random benchmark generated.")
