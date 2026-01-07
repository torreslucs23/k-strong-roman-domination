import os
import networkx as nx

# Script base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "generated_graphs")

os.makedirs(OUT_DIR, exist_ok=True)

# Number of graphs per type
NUM_GRAPHS = 15

# Sizes (increasing)
SIZES = [
    10, 20, 30, 50, 75,
    100, 125, 150, 200,
    300, 400, 500, 600, 700, 1000
]

assert len(SIZES) == NUM_GRAPHS


def save_graph(G, path):
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")


# =========================
# 1. Path Graphs
# =========================
path_dir = os.path.join(OUT_DIR, "paths")
os.makedirs(path_dir, exist_ok=True)

for i, n in enumerate(SIZES):
    G = nx.path_graph(n)
    save_graph(G, os.path.join(path_dir, f"path_{n}.txt"))


# =========================
# 2. Cycle Graphs
# =========================
cycle_dir = os.path.join(OUT_DIR, "cycles")
os.makedirs(cycle_dir, exist_ok=True)

for i, n in enumerate(SIZES):
    G = nx.cycle_graph(n)
    save_graph(G, os.path.join(cycle_dir, f"cycle_{n}.txt"))


# =========================
# 3. Star Graphs
# =========================
star_dir = os.path.join(OUT_DIR, "stars")
os.makedirs(star_dir, exist_ok=True)

for i, n in enumerate(SIZES):
    # star_graph(n) creates n+1 vertices (1 center + n leaves)
    G = nx.star_graph(n)
    save_graph(G, os.path.join(star_dir, f"star_{n+1}.txt"))


# =========================
# 4. 3-Regular Graphs
# =========================
regular_dir = os.path.join(OUT_DIR, "regular3")
os.makedirs(regular_dir, exist_ok=True)

for i, n in enumerate(SIZES):
    # 3-regular only exists if n is even
    if n % 2 != 0:
        n += 1

    G = nx.random_regular_graph(d=3, n=n, seed=i)
    save_graph(G, os.path.join(regular_dir, f"regular3_{n}.txt"))


print("Graphs generated successfully!")
