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
