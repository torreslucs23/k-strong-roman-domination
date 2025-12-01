import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize
import networkx as nx


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
        out['decoded'] = [temp_graph.nodes[i]['weight'] for i in temp_graph.nodes()]
        
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

path_small = nx.path_graph(10)
path_medium = nx.path_graph(50)
path_large = nx.path_graph(100)
dense_50_high = nx.erdos_renyi_graph(50, 0.7)
problem = Roman2StrongDominationProblem(dense_50_high)

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