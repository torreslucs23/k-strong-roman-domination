import os
import shutil
import networkx as nx
from pathlib import Path

def analyze_graph(filepath):
    """Analyze graph and return n, m, filesize"""
    try:
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
        
        n = G.number_of_nodes()
        m = G.number_of_edges()
        filesize = os.path.getsize(filepath)
        
        return n, m, filesize
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None, None, None

def filter_large_graphs(input_dir, output_dir, max_vertices=1000, max_filesize_mb=5):
    """
    Filter out large graphs
    
    Parameters:
        input_dir: directory with graphs
        output_dir: where to save filtered graphs
        max_vertices: maximum number of vertices allowed
        max_filesize_mb: maximum file size in MB
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    large_graphs_path = Path(f"{output_dir}_large")
    
    output_path.mkdir(parents=True, exist_ok=True)
    large_graphs_path.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(input_path.glob("*.txt"))
    
    print(f"Found {len(txt_files)} graphs in {input_dir}")
    print(f"Filtering: n <= {max_vertices}, filesize <= {max_filesize_mb}MB")
    print("="*70)
    
    kept = 0
    removed = 0
    max_filesize_bytes = max_filesize_mb * 1024 * 1024
    
    for filepath in sorted(txt_files):
        n, m, filesize = analyze_graph(filepath)
        
        if n is None:
            continue
        
        filesize_mb = filesize / (1024 * 1024)
        
        # Decide if keep or remove
        if n >= max_vertices or filesize >= max_filesize_bytes:
            # Too large - move to large_graphs folder
            shutil.copy2(filepath, large_graphs_path / filepath.name)
            removed += 1
            status = "REMOVED"
            reason = []
            if n >= max_vertices:
                reason.append(f"n={n}>={max_vertices}")
            if filesize >= max_filesize_bytes:
                reason.append(f"size={filesize_mb:.2f}MB")
            reason_str = ", ".join(reason)
            print(f"{filepath.name:40s} | {status:8s} | {reason_str}")
        else:
            # Keep it
            shutil.copy2(filepath, output_path / filepath.name)
            kept += 1
            print(f"✓  {filepath.name:40s} | n={n:4d} | m={m:6d} | {filesize_mb:.2f}MB")
    
    print("="*70)
    print(f"KEPT: {kept} graphs (saved to {output_dir})")
    print(f"REMOVED: {removed} graphs (saved to {output_dir}_large)")

if __name__ == "__main__":
    filter_large_graphs(
        input_dir="graphs/test_graphs/DIMACS_filtered",
        output_dir="graphs/tunning_filtered",
        max_vertices=500,      # adjust this
        max_filesize_mb=2      # adjust this
    )