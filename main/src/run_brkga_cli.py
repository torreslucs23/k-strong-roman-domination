import argparse
import networkx as nx
import numpy as np
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize

from tools.check_2_strong_roman import fix_instance
from brkga import Roman2StrongDominationProblem, RomanDuplicateElimination

def read_graph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            u, v = map(int, line.split())
            G.add_edge(u, v)
    return G

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance", required=True)
    parser.add_argument("--n_elites", type=int, required=True)
    parser.add_argument("--n_offsprings", type=int, required=True)
    parser.add_argument("--n_mutants_brkga", type=int, required=True)
    parser.add_argument("--bias", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)

    G = read_graph(args.instance)

    problem = Roman2StrongDominationProblem(G)

    algorithm = BRKGA(
        n_elites=args.n_elites,
        n_offsprings=args.n_offsprings,
        n_mutants=args.n_mutants_brkga,
        bias=args.bias,
        eliminate_duplicates=RomanDuplicateElimination()
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 100),
        seed=args.seed,
        verbose=False
    )

    print(res.F[0])

if __name__ == "__main__":
    main()
