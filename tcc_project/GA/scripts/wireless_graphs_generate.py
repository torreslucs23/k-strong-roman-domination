import os
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "benchmarks", "wireless")

os.makedirs(OUT_DIR, exist_ok=True)

sizes = [20, 30, 50, 100]
radii = [0.3, 0.4, 0.5, 0.6]

seed = 42

def save_graph(G, path):
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

for n in sizes:
    for r in radii:
        G = nx.random_geometric_graph(n, radius=r, seed=seed)
        G.remove_edges_from(nx.selfloop_edges(G))

        name = f"wireless_n{n}_r{r}.txt"
        save_graph(G, os.path.join(OUT_DIR, name))

print("Wireless benchmark generated.")
