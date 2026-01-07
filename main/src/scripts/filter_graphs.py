import os
import shutil
from pathlib import Path
import networkx as nx
from collections import defaultdict
import random


def read_edgelist_graph(filepath):
    """
    Reads a graph from an edge list file.

    Expected format:
        0 1
        0 4
        1 2
        ...
    """
    G = nx.Graph()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)

    return G


def read_adjacency_matrix_graph(filepath):
    """
    Reads a graph from an adjacency matrix file.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = [float(x.strip()) for x in line.split(',')]
        matrix.append(row)

    n = len(matrix)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0.5:
                G.add_edge(i, j)

    return G


def calculate_density(G):
    """
    Computes graph density.
    density = 2*m / (n*(n-1))
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n <= 1:
        return 0.0

    return (2.0 * m) / (n * (n - 1))


def classify_by_size(n):
    """
    Classifies graph by number of vertices.
    """
    if n < 100:
        return "tiny"
    elif 100 <= n < 500:
        return "small"
    elif 500 <= n < 1000:
        return "medium"
    elif 1000 <= n <= 1500:
        return "large"
    else:
        return "xlarge"


def classify_by_density(density):
    """
    Classifies graph by density.
    """
    if density < 0.4:
        return "sparse"
    elif 0.4 <= density < 0.7:
        return "average"
    else:
        return "dense"


def process_directory(dir_path, is_adjacency_matrix=False):
    """
    Processes all graph files in a directory and extracts their properties.
    """
    graphs_info = []

    dir_path = Path(dir_path)
    txt_files = list(dir_path.glob("*.txt"))

    print(f"\nProcessing directory: {dir_path.name}")
    print(f"Found {len(txt_files)} files")

    for filepath in txt_files:
        try:
            # Read graph
            if is_adjacency_matrix:
                G = read_adjacency_matrix_graph(filepath)
            else:
                G = read_edgelist_graph(filepath)

            n = G.number_of_nodes()
            m = G.number_of_edges()
            density = calculate_density(G)

            size_class = classify_by_size(n)
            density_class = classify_by_density(density)

            graphs_info.append({
                'filename': filepath.name,
                'filepath': filepath,
                'n': n,
                'm': m,
                'density': density,
                'size_class': size_class,
                'density_class': density_class
            })

        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    print(f"  Successfully processed {len(graphs_info)} graphs")
    return graphs_info


def select_diverse_graphs_fixed_count(graphs_info, target_count):
    """
    Selects exactly target_count graphs with diversity.

    Strategy:
    1. Group by (size_class, density_class)
    2. Distribute target_count proportionally across groups
    3. Uniformly sample graphs inside each group
    """
    if len(graphs_info) <= target_count:
        return graphs_info

    organized = defaultdict(list)
    for g in graphs_info:
        key = (g['size_class'], g['density_class'])
        organized[key].append(g)

    for key in organized:
        organized[key].sort(key=lambda x: x['density'])

    categories = list(organized.keys())
    graphs_per_category = target_count // len(categories)
    remainder = target_count % len(categories)

    selected = []

    for i, key in enumerate(categories):
        candidates = organized[key]

        count_to_select = graphs_per_category + (1 if i < remainder else 0)
        count_to_select = min(count_to_select, len(candidates))

        if count_to_select > 0:
            step = len(candidates) / count_to_select
            indices = [int(i * step) for i in range(count_to_select)]

            for idx in indices:
                if idx < len(candidates):
                    selected.append(candidates[idx])

    if len(selected) < target_count:
        remaining_needed = target_count - len(selected)
        all_not_selected = [g for g in graphs_info if g not in selected]

        all_not_selected.sort(key=lambda x: (x['size_class'], x['density']))
        selected.extend(all_not_selected[:remaining_needed])

    return selected[:target_count]


def copy_selected_graphs(selected_graphs, output_dir):
    """
    Copies selected graph files to the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for graph_info in selected_graphs:
        src = graph_info['filepath']
        dst = output_path / graph_info['filename']
        shutil.copy2(src, dst)


def print_selection_summary(selected_graphs, dir_name, target_count):
    """
    Prints a summary of the selected graphs.
    """
    print("\n" + "=" * 70)
    print(f"SELECTION SUMMARY: {dir_name}")
    print("=" * 70)
    print(f"Target: {target_count} graphs | Selected: {len(selected_graphs)} graphs\n")

    by_size = defaultdict(int)
    by_density = defaultdict(int)

    for g in selected_graphs:
        by_size[g['size_class']] += 1
        by_density[g['density_class']] += 1

    print("By size:")
    for size_class in ['tiny', 'small', 'medium', 'large', 'xlarge']:
        count = by_size[size_class]
        if count > 0:
            print(f"  {size_class:8s}: {count:3d} graphs")

    print("\nBy density:")
    for density_class in ['sparse', 'average', 'dense']:
        count = by_density[density_class]
        if count > 0:
            print(f"  {density_class:8s}: {count:3d} graphs")

    n_values = [g['n'] for g in selected_graphs]
    m_values = [g['m'] for g in selected_graphs]
    d_values = [g['density'] for g in selected_graphs]

    print("\nStatistics:")
    print(f"  Vertices: min={min(n_values)}, max={max(n_values)}, avg={sum(n_values)/len(n_values):.1f}")
    print(f"  Edges:    min={min(m_values)}, max={max(m_values)}, avg={sum(m_values)/len(m_values):.1f}")
    print(f"  Density:  min={min(d_values):.4f}, max={max(d_values):.4f}, avg={sum(d_values)/len(d_values):.4f}")


def main():
    """
    Processes directories and filters graphs to reach a total of 330.

    Target distribution:
    - 16 wireless-networks
    - 35 random
    - 168 geoinstances
    - 70 Harwell-Boeing
    - 41 DIMACS
    """
    print("=" * 70)
    print("GRAPH FILTERING - TARGET: 330 GRAPHS")
    print("=" * 70)

    directories_config = [
        ("wireless-networks", False, 16),
        ("random", False, 35),
        ("geoinstances", False, 168),
        ("Harwell-Boeing", False, 70),
        ("DIMACS", False, 41),
    ]

    total_selected = 0
    results = []

    for dir_name, is_matrix, target_count in directories_config:
        if not os.path.exists(dir_name):
            print(f"\nDirectory '{dir_name}' not found. Skipping.")
            continue

        graphs_info = process_directory(dir_name, is_adjacency_matrix=is_matrix)

        if not graphs_info:
            print(f"No valid graphs found in {dir_name}")
            continue

        available = len(graphs_info)
        actual_target = min(available, target_count)

        if available < target_count:
            print(f"Warning: only {available} graphs available, target was {target_count}")

        selected = select_diverse_graphs_fixed_count(graphs_info, actual_target)

        output_dir = f"{dir_name}_filtered"
        copy_selected_graphs(selected, output_dir)

        print_selection_summary(selected, dir_name, target_count)
        print(f"\n{len(selected)} graphs copied to: {output_dir}/")

        total_selected += len(selected)
        results.append({
            'dir': dir_name,
            'target': target_count,
            'selected': len(selected)
        })

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for r in results:
        status = "OK" if r['selected'] == r['target'] else "WARNING"
        print(f"{status:7s} {r['dir']:20s}: {r['selected']:3d} / {r['target']:3d} graphs")

    print("-" * 70)
    print(f"TOTAL SELECTED: {total_selected} / 330 graphs")
    print("=" * 70)

    if total_selected == 330:
        print("Target reached: 330 graphs selected")
    elif total_selected < 330:
        print(f"{330 - total_selected} graphs missing to reach the target")
    else:
        print(f"{total_selected - 330} graphs above the target")

    print("\nEstimated processing time:")
    print(f"  {total_selected} graphs x 15 min = {total_selected * 15} min")
    print(f"  = {total_selected * 15 / 60:.1f} hours")
    print(f"  = {total_selected * 15 / 60 / 24:.2f} days")


if __name__ == "__main__":
    main()
