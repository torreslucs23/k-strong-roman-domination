import math
import numpy as np
import networkx as nx
from tools.check_2_strong_roman import fix_instance as fix_instance
from multiprocessing import Pool, cpu_count


# from utils.decorators import simple_decorator


def evaluate_individual(args):
    individual, graph, fix_instance = args

    temp_graph = graph.copy()
    for node, weight in enumerate(individual):
        temp_graph.nodes[node]['weight'] = weight

    temp_graph = fix_instance(temp_graph)
    individual_fixed = np.array(
        [temp_graph.nodes[node]['weight'] for node in temp_graph.nodes()]
    )

    fitness = np.sum(individual_fixed)
    return individual_fixed, fitness


class GeneticAlgorithm:
    def __init__(self, graph, population_size, gene_mutation_rate, n_mutants, crossover_rate, fix_instance, elitism_rate=0.2):
        self.graph = graph
        self.population_size = population_size
        self.gene_mutation_rate = gene_mutation_rate
        self.n_mutants = n_mutants
        self.crossover_rate = crossover_rate
        self.fix_instance = fix_instance
        self.elitism_rate = elitism_rate

    def initialize_random_population(self):
        self.population = np.random.randint(1, 4, size=(self.population_size // 2, self.graph.number_of_nodes()))
        return self.population
    
    def initialize_population_greedy(self):
        n = self.graph.number_of_nodes()
        population = []

        for _ in range(math.ceil(self.population_size / 2)):
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

        n_to_mutate = int(np.ceil(n_individuals * self.n_mutants))

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
    def crossover_uniform(parent1, parent2):
        mask = np.random.rand(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def crossover_population(self, pop):
        pop = pop.copy()
        np.random.shuffle(pop)

        new_pop = []
        crossover_methods = [self.crossover_single_point, self.crossover_uniform]

        for i in range(0, len(pop) - 1, 2):
            parent1, parent2 = pop[i], pop[i + 1]

            if np.random.rand() < self.crossover_rate:
                crossover_fn = np.random.choice(crossover_methods)
                child1, child2 = crossover_fn(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            new_pop.extend([child1, child2])

        return np.array(new_pop)

    
        
    def evaluate_population(self, population, previous_population):
        with Pool(processes=cpu_count() - 2) as pool:
            results = pool.map(
                evaluate_individual,
                [(ind, self.graph, self.fix_instance) for ind in population]
            )

        valid_individuals, fitness_scores = zip(*results)

        valid_individuals = np.array(valid_individuals)
        fitness_scores = np.array(fitness_scores)

        sorted_indices = np.argsort(fitness_scores)
        valid_individuals = valid_individuals[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        elite_size = int(self.elitism_rate * self.population_size)
        rest_size = self.population_size - elite_size

        if previous_population is not None:
            previous_fitness = np.array([np.sum(ind) for ind in previous_population])
            elite_indices = np.argsort(previous_fitness)[:elite_size]
            elites = previous_population[elite_indices]
        else:
            elites = valid_individuals[:elite_size]

        rest_individuals = valid_individuals[:rest_size]
        new_population = np.vstack([elites, rest_individuals])

        return new_population[:self.population_size]

    
    def run(self, generations=100):
        
        pop_greedy = self.initialize_population_greedy()
        pop_random = self.initialize_random_population()
        population = np.vstack([pop_greedy, pop_random])
        

        for _ in range(generations):
            previous_population = population.copy()
            
            population = self.crossover_population(population)
            population = self.mutate_population(population)
            
            population = self.evaluate_population(population, previous_population=previous_population)
            

        best = min(population, key=lambda x: sum(x))
        return population
    
