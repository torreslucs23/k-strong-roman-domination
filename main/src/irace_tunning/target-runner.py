#!/usr/bin/env python3
import sys
import subprocess
import networkx as nx

def read_graph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            u, v = map(int, line.split())
            G.add_edge(u, v)
    return G

def main():
    # === IRACE PROTOCOL === 
    # sys.argv = [script, configuration_id, instance_id, seed, instance, params...]
    if len(sys.argv) < 5:
        print("999999")
        return
        
    config_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = int(sys.argv[3])
    instance = sys.argv[4]
    params = sys.argv[5:]

    algorithm = None
    params_dict = {}
    i = 0
    while i < len(params):
        if params[i].startswith("--"):
            param_name = params[i]
            param_value = params[i + 1] if i + 1 < len(params) else None
            
            if param_name == "--algorithm":
                algorithm = param_value
            else:
                params_dict[param_name] = param_value
            i += 2
        else:
            i += 1

    if not algorithm:
        print("999999")
        return

    # === FILTER PARAMS BY ALGORITHM ===
    if algorithm == "ga":
        ga_params = ["--population_size", "--gene_mutation_rate", "--n_mutants", "--crossover_rate", "--elitism_rate"]
        filtered_params = []
        for param in ga_params:
            if param in params_dict:
                filtered_params.extend([param, params_dict[param]])
        
        cmd = [
            "python3", "../run_ga_cli.py",
            "--instance", instance,
            "--seed", str(seed)
        ] + filtered_params

    elif algorithm == "brkga":
        try:
            n_elites = float(params_dict["--n_elites"])
            n_mutants = float(params_dict["--n_mutants_brkga"])
            pop_size = int(params_dict["--population_size"])
            bias = params_dict["--bias"]
        except KeyError:
            print("999999")
            return

        n_elites_abs = int(n_elites * pop_size)
        n_mutants_abs = int(n_mutants * pop_size)
        n_offsprings = pop_size - n_elites_abs - n_mutants_abs

        if n_offsprings <= 0 or n_elites_abs + n_mutants_abs >= pop_size:
            print("999999")
            return

        cmd = [
            "python3", "../run_brkga_cli.py",
            "--instance", instance,
            "--seed", str(seed),
            "--n_elites", str(n_elites_abs),
            "--n_mutants_brkga", str(n_mutants_abs),
            "--n_offsprings", str(n_offsprings),
            "--bias", bias,
        ]

    else:
        print("999999")
        return

    # === EXECUTA ===
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        cost = float(result.stdout.strip())
    except Exception:
        cost = 999999

    print(cost)

if __name__ == "__main__":
    main()
