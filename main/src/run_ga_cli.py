import argparse
import networkx as nx
from ga import GeneticAlgorithm
from tools.check_2_strong_roman import fix_instance

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
    parser.add_argument("--population_size", type=int, required=True)
    parser.add_argument("--gene_mutation_rate", type=float, required=True)
    parser.add_argument("--n_mutants", type=float, required=True)
    parser.add_argument("--crossover_rate", type=float, required=True)
    parser.add_argument("--elitism_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    G = read_graph(args.instance)

    ga = GeneticAlgorithm(
        graph=G,
        population_size=args.population_size,
        gene_mutation_rate=args.gene_mutation_rate,
        n_mutants=args.n_mutants,
        crossover_rate=args.crossover_rate,
        fix_instance=fix_instance,
        elitism_rate=args.elitism_rate
    )

    best_solution = ga.run(generations=100)
    best_fitness = sum(min(best_solution, key=lambda x: sum(x)))
    print(best_fitness)

if __name__ == "__main__":
    main()
