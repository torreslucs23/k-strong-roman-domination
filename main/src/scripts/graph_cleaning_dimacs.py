################################################################
# dependências: networkx
#
# Script que lê cada arquivo DIMACS (extensão .clq, .col,
# ou qualquer arquivo que contenha formato DIMACS "p edge" + "e u v")
# dentro da pasta "descompactados" e converte para um grafo NetworkX.
#
# Depois salva na pasta "base_final" um arquivo .txt contendo:
#     qtd_vertices qtd_arestas
#     u v
#     u v
#
# Os vértices são renumerados para 0-based consecutivo.
#
################################################################

import os
import networkx as nx

# Diretórios
INPUT_DIR = "BASE"
OUTPUT_DIR = "base_final"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_dimacs_graph(path):
    """
    Lê um grafo no formato DIMACS (edge/clique).
    Retorna: (G, num_vertices, num_edges)
    """
    G = nx.Graph()
    num_vertices = None
    num_edges = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("c"):
                continue

            parts = line.split()

            if parts[0] == "p":
                # p edge V E
                _, _, V, E = parts
                num_vertices = int(V)
                num_edges = int(E)

            elif parts[0] == "e":
                # e u v
                _, u, v = parts
                u = int(u)
                v = int(v)

                G.add_edge(u, v)
    return G, num_vertices, num_edges


def converter_dimacs_para_txt():
    ensure_dir(OUTPUT_DIR)

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:

            # Aqui você pode colocar qualquer extensão que seus DIMACS usam
            # Muitos vêm como .clq, .col, .txt...
            if not file.endswith((".clq", ".col", ".txt", ".dimacs")):
                continue

            caminho_entrada = os.path.join(root, file)
            nome_base = os.path.splitext(file)[0]
            caminho_saida = os.path.join(OUTPUT_DIR, f"{nome_base}.txt")

            print(f"Convertendo {file} → {caminho_saida}")
            try:
                # Lê o grafo DIMACS
                G, numV, numE = read_dimacs_graph(caminho_entrada)

                # Remove laços (se houver)
                G.remove_edges_from(nx.selfloop_edges(G))

                # Garante que é simples
                if isinstance(G, nx.MultiGraph):
                    G = nx.Graph(G)

                # Relabel para 0-based consecutivo
                mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
                G = nx.relabel_nodes(G, mapping)
                
                # Salvar
                with open(caminho_saida, "w") as f_out:
                    f_out.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")

                    for u, v in G.edges():
                        f_out.write(f"{u} {v}\n")

            except Exception as e:
                print(f"Erro ao processar {file}: {e}")

    print("\nConversão concluída com sucesso!")


if __name__ == "__main__":
    converter_dimacs_para_txt()
