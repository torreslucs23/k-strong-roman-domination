import networkx as nx
import sys

def analyze_graph(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0
    
    print(f"Arquivo: {filepath}")
    print(f"Vértices: {n}")
    print(f"Arestas: {m}")
    print(f"Densidade: {density:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_graph.py <caminho_do_grafo.txt>")
        sys.exit(1)
    
    analyze_graph(sys.argv[1])