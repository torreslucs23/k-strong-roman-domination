import os
import geopandas as gpd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "neighborhoods")
OUTPUT_DIR = os.path.join(BASE_DIR, "graphs_txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def geojson_to_graph(gdf):
    # Fix invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)

    G = nx.Graph()

    for idx in gdf.index:
        G.add_node(idx)

    geometries = gdf.geometry

    for i in geometries.index:
        geom_i = geometries[i]
        if geom_i is None or geom_i.is_empty:
            continue

        for j in geometries.index:
            if i >= j:
                continue

            geom_j = geometries[j]
            if geom_j is None or geom_j.is_empty:
                continue

            try:
                if geom_i.touches(geom_j):
                    G.add_edge(i, j)
            except Exception:
                # ignore problematic pairs
                continue

    return G

def save_graph(G, path):
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")


for file in os.listdir(INPUT_DIR):
    if file.endswith(".geojson"):
        path = os.path.join(INPUT_DIR, file)
        name = os.path.splitext(file)[0]

        print(f"Processing {file}...")

        gdf = gpd.read_file(path)
        G = geojson_to_graph(gdf)

        out_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        save_graph(G, out_path)

print("Conversion completed.")
