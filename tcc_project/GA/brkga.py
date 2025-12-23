import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize
import networkx as nx
from utils.decorators import simple_decorator


from tools.check_2_strong_roman import fix_instance

class Roman2StrongDominationProblem(ElementwiseProblem):
    def __init__(self, graph):
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        super().__init__(n_var=self.n_nodes, n_obj=1, xl=0, xu=1)
        
    def _evaluate(self, x, out, *args, **kwargs):
        decoded_solution = self.decode(x)
        
        fitness, temp_graph = self.evaluate_solution(decoded_solution)
        
        out['F'] = fitness
        out['decoded'] = self.encode([temp_graph.nodes[i]['weight'] for i in temp_graph.nodes()])
        
    def decode(self, x):
        weights = np.zeros(self.n_nodes, dtype=int)
        
        for i, val in enumerate(x):
            if val < 0.25:
                weights[i] = 0
            elif val < 0.5:
                weights[i] = 1
            elif val < 0.75:
                weights[i] = 2
            else:
                weights[i] = 3

        return weights
    
    def encode(self, x):
        n = len(x)
        
        for i in range(n):
            if x[i] == 0:
                x[i] = 0.1
            elif x[i] == 1:
                x[i] = 0.3
            elif x[i] == 2:
                x[i] = 0.6
            elif x[i] == 3:
                x[i] = 0.9
        return x
    
    def evaluate_solution(self, weights):
        temp_graph = self.graph.copy()
        for node in temp_graph.nodes():
            temp_graph.nodes[node]['weight'] = weights[node]
        
        temp_graph = fix_instance(temp_graph)
        
        fitness = sum([data['weight'] for _, data in temp_graph.nodes(data=True)])
        return fitness, temp_graph
    
class RomanDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.array_equal(a.get("decoded"), b.get("decoded"))


@simple_decorator
def run():
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
    path_medium = nx.path_graph(50)
    path_large = nx.path_graph(100)
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
    problem = Roman2StrongDominationProblem(nx.petersen_graph())

    algorithm = BRKGA(
        n_elites=20,
        n_offsprings=70,
        n_mutants=10,
        bias=0.7,
        eliminate_duplicates=RomanDuplicateElimination()
    )

    res = minimize(problem, algorithm, ("n_gen", 100), seed=1, verbose=True)

    best_solution = res.opt.get("decoded")[0]
    best_fitness = res.F[0]
    print("Melhor solução:", best_solution)
    print("Fitness da melhor solução:", best_fitness)
    
run()