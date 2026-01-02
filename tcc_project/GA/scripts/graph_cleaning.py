################################################################
# dependências: networkx, scipy
#
# Script que lê cada arquivo com extensão .mtx (Matrix Market Format) que 
# está dentro da pasta "descompactados" e converte cada matriz esparsa 
# em um grafo do pacote NetworkX. Depois, ele salva na pasta "base_final" 
# um arquivo .txt correspondente, contendo a lista de adjacência do grafo.
# 
# O script assume que a matriz esparsa .mtx representa uma 
# matriz de adjacência simétrica (grafo não direcionado).
#
################################################################
import os
import networkx as nx
from scipy.io import mmread

# Diretórios
INPUT_DIR = "descompactados"
OUTPUT_DIR = "base_final"

def ensure_dir(path):
    """Garante que o diretório de saída exista."""
    if not os.path.exists(path):
        os.makedirs(path)

def converter_mtx_para_txt():
    ensure_dir(OUTPUT_DIR)

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".mtx"):
                caminho_entrada = os.path.join(root, file)
                nome_base = os.path.splitext(file)[0]
                caminho_saida = os.path.join(OUTPUT_DIR, f"{nome_base}.txt")

                print(f"Convertendo {file} → {caminho_saida}")

                try:
                    # Lê a matriz esparsa
                    matriz = mmread(caminho_entrada).tocsr()
                    
                    # Carrega apenas matrizes quadradas (com mesmo número de linhas e colunas)
                    if matriz.shape[0] != matriz.shape[1]:
                        print(f"Erro: a matriz {file} não é quadrada")
                        continue
                    
                    # Converte para grafo não direcionado
                    G = nx.from_scipy_sparse_array(matriz)

                    # Garante que é um grafo sem arestas múltiplas
                    if isinstance(G, nx.MultiGraph):
                        G = nx.Graph(G)

                    # Remove laços
                    G.remove_edges_from(nx.selfloop_edges(G))

                    # Relabel para 0-based consecutivo
                    mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(G.nodes()))}
                    G = nx.relabel_nodes(G, mapping)

                    # Salva lista de arestas
                    with open(caminho_saida, "w") as f_out:
                        for u, v in G.edges():
                            f_out.write(f"{u} {v}\n")

                except Exception as e:
                    print(f"Erro ao processar {file}: {e}")

    print("\nConversão concluída com sucesso!")


if __name__ == "__main__":
    converter_mtx_para_txt()
