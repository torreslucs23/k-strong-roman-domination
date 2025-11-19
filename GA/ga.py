import math
import numpy as np
import networkx as nx
from tools.check_2_strong_roman import fix_instance as fix_instance

class GeneticAlgorithm:
    def __init__(self, graph, population_size, mutation_rate, crossover_rate, fix_instance, k=2):
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.k = k
        self.fix_instance = fix_instance

    def initialize_random_population(self):
        # random initialization between 1 and 2
        self.population = np.random.randint(1, 3, size=(self.population_size // 2, self.graph.number_of_nodes()))
        return self.population
    
    def initialize_population_greedy(self):
        #greedy approach to initialize population from Djukanovic
        n = self.graph.number_of_nodes()
        s = np.zeros(n, dtype=int)
        covered = set()

        while len(covered) < n:
            g_values = {}

            for v in self.graph.nodes():
                if v in covered:
                    continue
                neighbors = set(self.graph.neighbors(v)) | {v}
                g_values[v] = len(neighbors - covered)

            v_prime = max(g_values, key=g_values.get)
            if v_prime in covered:
                I_v = 1
            else:
                I_v = 0

            l_v_prime = min(self.k + 1, g_values[v_prime] + I_v)
            s[v_prime] = l_v_prime

            neighbors_v_prime = set(self.graph.neighbors(v_prime)) | {v_prime}
            covered |= neighbors_v_prime

        population = np.tile(s, (math.ceil(self.population_size / 2), 1))
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
    
    def mutate_population(self, population):
        n_individuals, n_genes = population.shape

        n_to_mutate = int(np.ceil(n_individuals * 0.30))

        indices_to_mutate = np.random.choice(n_individuals, size=n_to_mutate, replace=False)
        for i in indices_to_mutate:
            for j in range(n_genes):
                
                if np.random.rand() < self.mutation_rate:
                    
                    if population[i, j] == 3:
                        population[i, j] = 2
                        
                    elif population[i, j] == 1:
                        population[i, j] = 0
                        
        return population
        
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

        elite_size = max(1, int(0.1 * len(previous_population))) if previous_population is not None else 0
        
        if previous_population is not None and elite_size > 0:
            previous_fitness = np.array([np.sum(ind) for ind in previous_population])
            elite_indices = np.argsort(previous_fitness)[:elite_size]
            elites = previous_population[elite_indices]
            
            remaining_size = self.population_size - elite_size
            if len(valid_individuals) > remaining_size:
                valid_individuals = valid_individuals[:remaining_size]
            
            population_combined = np.vstack([elites, valid_individuals])
        else:
            if len(valid_individuals) > self.population_size:
                population_combined = valid_individuals[:self.population_size]
            else:
                population_combined = valid_individuals

        final_fitness = np.array([np.sum(ind) for ind in population_combined])
        final_sorted_indices = np.argsort(final_fitness)
        return population_combined[final_sorted_indices]
    
    def run(self, generations=50):
        
        pop_greedy = self.initialize_population_greedy()
        pop_random = self.initialize_random_population()
        population = np.vstack([pop_greedy, pop_random])
        
        
        
        population = np.sort(population, axis=1)
        

        for _ in range(generations):
            previous_population = population.copy()

            population = np.vstack([population, self.crossover_population(population)])
            population = np.vstack([population, self.mutate_population(population)])
            # print(population)
            # print()
            
            population = self.evaluate_population(population, previous_population=previous_population)
            

        return population
    
    
def main():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2),
        (1, 3),
        (2, 3), (2, 4),
        (3, 4)
    ])
    G = nx.petersen_graph()
    population_size = 10
    mutation_rate = 0.1
    crossover_rate = 0.7
    k = 2
    generations = 30
    
    

    ga = GeneticAlgorithm(
        graph=G,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        k=k,
        fix_instance=fix_instance
    )

    final_population = ga.run(generations=generations)

    print("População final (válida):")
    print(len(final_population))

    if len(final_population) > 0:
        print("\nExemplo de indivíduo viável:")
        print(final_population[0])

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