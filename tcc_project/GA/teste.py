import os
import networkx as nx
import csv


def load_graph_from_txt(path):
    """
    Load an undirected graph from a TXT file.
    Supports optional first line 'n m'.
    """
    G = nx.Graph()

    with open(path, "r") as f:
        lines = f.readlines()

    if not lines:
        return G

    start_idx = 0
    first = lines[0].strip().split()
    if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
        start_idx = 1

    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        u, v = map(int, parts)
        if u != v:
            G.add_edge(u, v)

    return G


def analyze_directory(base_dir, dataset_name):
    """
    Analyze all TXT graphs inside a directory.
    """
    results = []

    print(f"\n=== Analyzing {dataset_name} ===")

    for filename in sorted(os.listdir(base_dir)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(base_dir, filename)

        try:
            G = load_graph_from_txt(path)
            n = G.number_of_nodes()
            m = G.number_of_edges()

            results.append({
                "dataset": dataset_name,
                "graph": filename,
                "vertices": n,
                "edges": m
            })

            print(f"{filename:30s} | n={n:6d} | m={m:8d}")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    return results


def save_to_csv(results, output_file="graph_summary.csv"):
    """
    Save results to CSV.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "graph", "vertices", "edges"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {output_file}")


def main():

    base_paths = {
        "DIMACS": os.path.join("graphs", "DIMACS", "base_final"),
        "Harwell-Boeing": os.path.join("graphs", "Harwell-Boeing", "base_final")
    }

    all_results = []

    for name, path in base_paths.items():
        if not os.path.isdir(path):
            print(f"[WARNING] Directory not found: {path}")
            continue

        results = analyze_directory(path, name)
        all_results.extend(results)

    # Save summary
    save_to_csv(all_results)


if __name__ == "__main__":
    main()
