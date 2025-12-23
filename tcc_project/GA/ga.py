import math
import numpy as np
import networkx as nx
from tools.check_2_strong_roman import fix_instance as fix_instance

from utils.decorators import simple_decorator

class GeneticAlgorithm:
    def __init__(self, graph, population_size, gene_mutation_rate, individual_mutation_rate, crossover_rate, fix_instance, k=2, denominator=2):
        self.graph = graph
        self.population_size = population_size
        self.gene_mutation_rate = gene_mutation_rate
        self.individual_mutation_rate = individual_mutation_rate
        self.crossover_rate = crossover_rate
        self.denominator = denominator
        self.fix_instance = fix_instance

    def initialize_random_population(self):
        self.population = np.random.randint(1, 4, size=(self.population_size // 2, self.graph.number_of_nodes()))
        return self.population
    
    def initialize_population_greedy(self):
        n = self.graph.number_of_nodes()
        population = []

        for _ in range(math.ceil(self.population_size / self.denominator)):
            s = np.zeros(n, dtype=int)
            covered = set()

            shuffled_nodes = list(self.graph.nodes())
            np.random.shuffle(shuffled_nodes)

            while len(covered) < n:
                g_values = {}

                for v in shuffled_nodes:
                    if v in covered:
                        continue
                    neighbors = set(self.graph.neighbors(v)) | {v}
                    g_values[v] = len(neighbors - covered)

                v_prime = max(g_values, key=g_values.get)
                
                uncov_v = (set(self.graph.neighbors(v_prime)) | {v_prime}) - covered
                
                if v_prime in uncov_v:
                    I_v = 0
                else:
                    I_v = 1

                s[v_prime] = min(3, g_values[v_prime] + I_v)

                covered |= uncov_v

            population.append(s)
            
        return np.array(population)
    
    def mutate_population(self, population):
        n_individuals, n_genes = population.shape

        n_to_mutate = int(np.ceil(n_individuals * self.individual_mutation_rate))

        indices_to_mutate = np.random.choice(
            n_individuals,
            size=n_to_mutate,
            replace=False
        )

        for i in indices_to_mutate:
            for j in range(n_genes):
                if np.random.rand() < self.gene_mutation_rate:

                    if population[i, j] == 3:
                        population[i, j] = 2

                    elif population[i, j] == 1:
                        population[i, j] = 0

        return population


    
    @staticmethod
    def crossover_single_point(parent1, parent2):
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    @staticmethod
    def crossover_uniform(parent1, parent2, prob=0.5):
        mask = np.random.rand(len(parent1)) < prob
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def crossover_population(self, pop):
        new_pop = []
        crossover_methods = [self.crossover_single_point, self.crossover_uniform]
        
        for i in range(0, len(pop), 2):
            parent1, parent2 = pop[i], pop[(i+1) % len(pop)]
            if np.random.rand() < self.crossover_rate:
                crossover_fn = np.random.choice(crossover_methods)
                child1, child2 = crossover_fn(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            new_pop.extend([child1, child2])
        return np.array(new_pop)
    
    
        
    def evaluate_population(self, population, previous_population):
        valid_individuals = []
        fitness_scores = []

        for individual in population:
            temp_graph = self.graph.copy()
            for node, weight in enumerate(individual):
                temp_graph.nodes[node]['weight'] = weight

            temp_graph = self.fix_instance(temp_graph)
            individual_fixed = np.array([temp_graph.nodes[node]['weight'] for node in temp_graph.nodes()])
            valid_individuals.append(individual_fixed)
            fitness_scores.append(np.sum(individual_fixed))


        valid_individuals = np.array(valid_individuals)
        fitness_scores = np.array(fitness_scores)

        sorted_indices = np.argsort(fitness_scores)
        valid_individuals = valid_individuals[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        elite_size = int(0.2 * self.population_size)
        rest_size = self.population_size - elite_size

        if previous_population is not None:
            previous_fitness = np.array([np.sum(ind) for ind in previous_population])
            elite_indices = np.argsort(previous_fitness)[:elite_size]
            elites = previous_population[elite_indices]
        else:
            elites = valid_individuals[:elite_size]

        if len(valid_individuals) > rest_size:
            rest_individuals = valid_individuals[:rest_size]
        else:
            rest_individuals = valid_individuals

        new_population = np.vstack([elites, rest_individuals])

        if len(new_population) > self.population_size:
            new_population = new_population[:self.population_size]

        return new_population
    
    @simple_decorator
    def run(self, generations=50):
        
        pop_greedy = self.initialize_population_greedy()
        pop_random = self.initialize_random_population()
        population = np.vstack([pop_greedy, pop_random])
        

        for _ in range(generations):
            previous_population = population.copy()
            

            population = self.crossover_population(population)
            population = self.mutate_population(population)
            
            
            population = self.evaluate_population(population, previous_population=previous_population)
            # print(population)
            # print()
            

        return population
    
    
def main():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3),
        (3, 4), (4, 0),
        (0, 5), (1, 6),
        (2, 7), (3, 8),
        (4, 9),
        (5, 7), (7, 9),
        (6, 8), (6, 9),
        (5, 8)
    ])
    path_small = nx.path_graph(100)
    dense_50_high = nx.erdos_renyi_graph(50, 0.7)
    G_fixed_50 = nx.Graph()
    edges_50 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
            (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
            (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
            (5, 6), (5, 7), (5, 8), (5, 9),
            (6, 7), (6, 8), (6, 9),
            (7, 8), (7, 9),
            (8, 9),
            # Adiciona mais conexões para tornar denso
            (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19),
            (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19),
            (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19),
            # Conecta os dois clusters
            (0, 10), (1, 11), (2, 12), (3, 13), (4, 14),
            # Adiciona algumas diagonais
            (0, 20), (1, 21), (2, 22), (3, 23), (4, 24),
            (20, 21), (21, 22), (22, 23), (23, 24), (24, 20),
            # Liga todos ao centro
            (25, 0), (25, 1), (25, 2), (25, 3), (25, 4),
            (25, 10), (25, 11), (25, 12), (25, 13), (25, 14),
            (25, 20), (25, 21), (25, 22), (25, 23), (25, 24),
            # Preenche o resto
            (26, 27), (26, 28), (26, 29), (26, 30),
            (27, 28), (27, 29), (27, 30),
            (28, 29), (28, 30),
            (29, 30),
            # Conexões aleatórias extras
            (31, 32), (31, 33), (31, 34), (32, 33), (32, 34), (33, 34),
            (35, 36), (35, 37), (35, 38), (36, 37), (36, 38), (37, 38),
            (39, 40), (39, 41), (39, 42), (40, 41), (40, 42), (41, 42),
            (43, 44), (43, 45), (43, 46), (44, 45), (44, 46), (45, 46),
            (47, 48), (47, 49), (48, 49),
            # Liga todos os clusters
            (0, 26), (10, 31), (20, 35), (25, 39), (30, 43), (34, 47)]

    G_fixed_50.add_edges_from(edges_50)


    # G = nx.petersen_graph()
    population_size = 100
    gene_mutation_rate = 0.5
    individual_mutation_rate = 0.3
    crossover_rate = 0.7
    generations = 100
    
    

    ga = GeneticAlgorithm(
        graph=nx.petersen_graph(),
        population_size=population_size,
        gene_mutation_rate=gene_mutation_rate,
        individual_mutation_rate=individual_mutation_rate,
        crossover_rate=crossover_rate,
        fix_instance=fix_instance,
        denominator=3
    )

    final_population = ga.run(generations=generations)

    if len(final_population) > 0:
        print("\nExemplo de indivíduo viável:")
        print(final_population[0])
        print(f"O valor dele é: {sum(final_population[0])}")

if __name__ == "__main__":
    main()


                

        

        
# g = nx.Graph()
# g.add_nodes_from([0, 1, 2, 3, 4, 5])
# g.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5)])
# g.nodes[0]['weight'] = 0
# g.nodes[1]['weight'] = 0
# g.nodes[2]['weight'] = 1
# g.nodes[3]['weight'] = 1
# g.nodes[4]['weight'] = 0
# ga = GeneticAlgorithm(g, population_size=10, mutation_rate=0.01, crossover_rate=0.7)
# population = ga.initialize_population_greedy()
# print(population)