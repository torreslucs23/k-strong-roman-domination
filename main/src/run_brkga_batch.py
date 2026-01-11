import os
import csv
import time
import traceback
import networkx as nx
import numpy as np
from datetime import datetime
from shutil import copy2
from pathlib import Path

from brkga import Roman2StrongDominationProblem, RomanDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize


BASE_DIR = "graphs/test_graphs"

GRAPH_DIRS = [
    "DIMACS_filtered",
]

OUTPUT_CSV = "brkga_results.csv"
CHECKPOINT_FILE = ".checkpoint_brkga.txt"
BACKUP_DIR = "brkga_backups"
BACKUP_INTERVAL = 5
N_RUNS = 20

# Best parameters from irace
BEST_PARAMS = {
    'n_elites': 39,
    'n_offsprings': 242,
    'n_mutants': 19,
    'bias': 0.7106,
    'generations': 100
}


def load_graph_from_txt(filepath):
    """Load graph and normalize vertex IDs to 0..n-1"""
    G_original = nx.Graph()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            G_original.add_edge(u, v)
    
    G = nx.Graph()
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G_original.nodes()))}
    
    for u, v in G_original.edges():
        G.add_edge(node_mapping[u], node_mapping[v])
    
    return G


def init_csv(filepath):
    header = [
        "graph_type",
        "name",
        "vertices",
        "edges",
        "density",
        "best",
        "mean",
        "median",
        "std",
        "worst",
        "time_mean_ms",
        "time_std_ms",
        "n_runs",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv(filepath, row):
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def create_backup(filepath):
    if os.path.exists(filepath):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"{Path(filepath).name}.backup_{timestamp}")
        copy2(filepath, backup_path)
        print(f"  [BACKUP] Created: {backup_path}")
        return backup_path
    return None


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed = set(line.strip() for line in f)
        return processed
    return set()


def save_checkpoint(graph_id):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{graph_id}\n")


def safe_init_csv(filepath):
    if os.path.exists(filepath):
        response = input(f"\n{filepath} already exists. Options:\n"
                        "  [a] Append to existing file (SAFE - recommended)\n"
                        "  [o] Overwrite (DANGER - will lose data)\n"
                        "  [b] Create backup and start fresh\n"
                        "  [q] Quit\n"
                        "Choose: ").lower()
        
        if response == 'a':
            print(f"  Appending to existing {filepath}")
            return 'append'
        elif response == 'o':
            confirm = input("  Are you SURE you want to overwrite? Type 'yes': ")
            if confirm == 'yes':
                create_backup(filepath)
                init_csv(filepath)
                return 'overwrite'
            else:
                print("  Cancelled. Exiting.")
                exit(0)
        elif response == 'b':
            create_backup(filepath)
            init_csv(filepath)
            return 'new'
        else:
            print("  Exiting.")
            exit(0)
    else:
        init_csv(filepath)
        return 'new'


def run_brkga_multiple_times(G, n_runs=20):
    """Run BRKGA multiple times and collect statistics"""
    results = []
    times = []
    
    for run in range(n_runs):
        np.random.seed(run)
        
        problem = Roman2StrongDominationProblem(G)
        
        algorithm = BRKGA(
            n_elites=BEST_PARAMS['n_elites'],
            n_offsprings=BEST_PARAMS['n_offsprings'],
            n_mutants=BEST_PARAMS['n_mutants'],
            bias=BEST_PARAMS['bias'],
            eliminate_duplicates=RomanDuplicateElimination()
        )
        
        start_time = time.time()
        res = minimize(
            problem,
            algorithm,
            ('n_gen', BEST_PARAMS['generations']),
            seed=run,
            verbose=False
        )
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        
        fitness = res.F[0]
        results.append(fitness)
        
        print(f"    Run {run+1}/{n_runs}: fitness={fitness:.0f}, time={elapsed_ms:.2f}ms")
    
    results = np.array(results)
    times = np.array(times)
    
    stats = {
        'best': np.min(results),
        'mean': np.mean(results),
        'median': np.median(results),
        'std': np.std(results),
        'worst': np.max(results),
        'time_mean_ms': np.mean(times),
        'time_std_ms': np.std(times),
        'n_runs': n_runs
    }
    
    return stats


def run():
    print("=" * 80)
    print("BRKGA BATCH TESTING - MULTIPLE RUNS PER GRAPH")
    print("=" * 80)
    print(f"Parameters: {BEST_PARAMS}")
    print(f"Runs per graph: {N_RUNS}")
    print("=" * 80)

    mode = safe_init_csv(OUTPUT_CSV)
    
    processed_graphs = load_checkpoint() if mode == 'append' else set()
    
    if processed_graphs:
        print(f"\n  Resuming from checkpoint: {len(processed_graphs)} graphs already processed")

    total_graphs = 0
    graphs_since_backup = 0

    for graph_type in GRAPH_DIRS:
        dir_path = os.path.join(BASE_DIR, graph_type)

        if not os.path.isdir(dir_path):
            print(f"[WARNING] Directory not found: {dir_path}")
            continue

        print(f"\nProcessing graph set: {graph_type}")
        print("-" * 80)

        files = sorted(
            f for f in os.listdir(dir_path) if f.endswith(".txt")
        )

        for filename in files:
            graph_id = f"{graph_type}/{filename}"
            
            if graph_id in processed_graphs:
                print(f"\n-> Skipping (already processed): {graph_id}")
                continue
            
            total_graphs += 1
            filepath = os.path.join(dir_path, filename)

            print(f"\n-> Processing graph [{total_graphs}]: {graph_id}")

            try:
                G = load_graph_from_txt(filepath)
                n = G.number_of_nodes()
                m = G.number_of_edges()
                density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0

                print(f"  Graph loaded | V={n}, E={m}, density={density:.4f}")
                print(f"  Running {N_RUNS} times...")

                stats = run_brkga_multiple_times(G, n_runs=N_RUNS)

                append_csv(
                    OUTPUT_CSV,
                    [
                        graph_type,
                        filename,
                        n,
                        m,
                        density,
                        stats['best'],
                        stats['mean'],
                        stats['median'],
                        stats['std'],
                        stats['worst'],
                        stats['time_mean_ms'],
                        stats['time_std_ms'],
                        stats['n_runs'],
                    ],
                )

                save_checkpoint(graph_id)
                graphs_since_backup += 1

                print(f"  Results: best={stats['best']:.0f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")
                print(f"  Time: mean={stats['time_mean_ms']:.2f}ms, std={stats['time_std_ms']:.2f}ms")

                if graphs_since_backup >= BACKUP_INTERVAL:
                    create_backup(OUTPUT_CSV)
                    graphs_since_backup = 0

            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Saving progress...")
                create_backup(OUTPUT_CSV)
                print(f"  Progress saved. Processed {total_graphs} graphs.")
                print(f"  Run again to resume from checkpoint.")
                exit(0)

            except Exception as e:
                print("  [ERROR] Failed to process graph")
                print(f"  Reason: {e}")
                traceback.print_exc()

                append_csv(
                    OUTPUT_CSV,
                    [
                        graph_type,
                        filename,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        0,
                    ],
                )
                save_checkpoint(graph_id)

    create_backup(OUTPUT_CSV)

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print("=" * 80)
    print(f"Total graphs processed: {total_graphs}")
    print(f"Results saved to: {OUTPUT_CSV}")
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Checkpoint file removed.")


if __name__ == "__main__":
    run()