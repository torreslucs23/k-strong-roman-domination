import matplotlib.pyplot as plt
import networkx as nx
from ortools.linear_solver import pywraplp
import pandas as pd
import time


def solve_two_strong_roman(G, time_limit_seconds=300):
    """
    Solves the 2-Strong Roman Domination problem using OR-Tools
    
    Parameters:
        G: NetworkX graph
        time_limit_seconds: maximum time in seconds
    
    Returns:
        dict with solution
    """
    
    V = list(G.nodes())
    neighbors = {i: list(G.neighbors(i)) for i in V}
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('SAT')
    
    if solver is None:
        print("Error: SAT solver not available")
        return None
    
    solver.SetTimeLimit(time_limit_seconds * 1000)
    print(f"Graph: n={len(V)}, m={G.number_of_edges()}")
    print(f"Building model...")
    
    # DECISION VARIABLES
    
    # z[v,k] = 1 if vertex v has label k ∈ {0,1,2,3}
    z = {}
    for v in V:
        for k in [0, 1, 2, 3]:
            z[v, k] = solver.BoolVar(f"z[{v},{k}]")
    
    # a[u,v] = 1 if f(u)=0, f(v)=2 and v is the unique protector of u
    a = {}
    for u in V:
        for v in neighbors[u]:
            a[u, v] = solver.BoolVar(f"a[{u},{v}]")
    
    # q[u,k] auxiliary variables to track number of protectors
    # q[u,0] = 1 if u has 0 protectors
    # q[u,1] = 1 if u has exactly 1 protector
    # q[u,2] = 1 if u has 2+ protectors
    q = {}
    for u in V:
        for k in [0, 1, 2]:
            q[u, k] = solver.BoolVar(f"q[{u},{k}]")
    
    print(f"Variables created: {len(z) + len(a) + len(q)}")
    
    # CONSTRAINTS
    
    print("Adding constraints...")
    
    # Constraint: each vertex receives exactly one label
    for v in V:
        solver.Add(z[v, 0] + z[v, 1] + z[v, 2] + z[v, 3] == 1)
    
    # Constraint: each vertex v with f(v)=2 must have at most one neighbor u
    # with label 0 such that v is the unique protector of u
    for v in V:
        if len(neighbors[v]) > 0:
            solver.Add(
                solver.Sum(a[u, v] for u in neighbors[v]) <= z[v, 2]
            )
    
    for u in V:
        deg_u = len(neighbors[u])
        
        if deg_u == 0:
            # Isolated vertex must have label >= 1
            solver.Add(z[u, 0] == 0)
            continue
        
        # protector_sum = sum of (z[w,2] + z[w,3]) for all neighbors w of u
        protector_sum = solver.Sum(z[w, 2] + z[w, 3] for w in neighbors[u])
        
        # Constraint: each vertex with label 0 must have at least one protector
        solver.Add(protector_sum >= z[u, 0])
        
        # Constraints to guarantee that q[u,1] = 1 iff u has exactly 1 protector
        solver.Add(protector_sum <= 1 + (1 - q[u, 1]) * deg_u)
        solver.Add(protector_sum >= 1 - (1 - q[u, 1]) * deg_u)
        
        # Constraints: q[u,0] + q[u,1] + q[u,2] = 1
        solver.Add(q[u, 0] + q[u, 1] + q[u, 2] == 1)
        
        # Additional constraints for q variables
        solver.Add(protector_sum <= 0 + deg_u * (1 - q[u, 0]))
        solver.Add(protector_sum <= 1 + deg_u * (1 - q[u, 1]))
        solver.Add(protector_sum >= 1 - deg_u * (1 - q[u, 1]))
        solver.Add(protector_sum >= 2 * q[u, 2])
        solver.Add(protector_sum <= 1 + (deg_u - 1) * q[u, 2])
    
    # Constraints: a[u,v] = 1 iff f(u)=0, f(v)=2 and v is unique protector of u
    for u in V:
        for v in neighbors[u]:
            solver.Add(a[u, v] <= z[u, 0])
            solver.Add(a[u, v] <= z[v, 2])
            solver.Add(a[u, v] <= q[u, 1])
            solver.Add(a[u, v] >= z[u, 0] + z[v, 2] + q[u, 1] - 2)
    
    print("Constraints added successfully")
    
    # OBJECTIVE FUNCTION
    
    # Minimize: sum of (1*z[v,1] + 2*z[v,2] + 3*z[v,3]) for all v
    objective = solver.Sum(
        z[v, 1] + 2*z[v, 2] + 3*z[v, 3] for v in V
    )
    solver.Minimize(objective)
    
    # SOLVE
    
    print("Solving...")
    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # milliseconds
    
    # EXTRACT SOLUTION
    
    if status == pywraplp.Solver.OPTIMAL:
        print("\n" + "="*60)
        print("OPTIMAL SOLUTION FOUND")
        print("="*60)
        
        # Extract labeling function f
        f = {}
        for v in V:
            for k in [0, 1, 2, 3]:
                if z[v, k].solution_value() > 0.5:
                    f[v] = k
                    break
        
        obj_value = solver.Objective().Value()
        
        # Count vertices by label
        labels_count = {0: 0, 1: 0, 2: 0, 3: 0}
        for v in V:
            labels_count[f[v]] += 1
        
        print(f"\nObjective value (γ_DR2F): {obj_value}")
        print(f"Time: {elapsed_time:.2f} ms")
        print(f"\nLabel distribution:")
        for k in [0, 1, 2, 3]:
            vertices_with_k = [v for v in V if f[v] == k]
            if vertices_with_k:
                print(f"  f(v) = {k}: {labels_count[k]} vertices")
                if len(vertices_with_k) <= 20:
                    print(f"    {vertices_with_k}")
        
        return {
            "status": "OPTIMAL",
            "objective": obj_value,
            "time_ms": elapsed_time,
            "f": f,
            "labels_count": labels_count
        }
    
    elif status == pywraplp.Solver.FEASIBLE:
        print("\n" + "="*60)
        print("FEASIBLE SOLUTION FOUND (not proven optimal)")
        print("="*60)
        
        f = {}
        for v in V:
            for k in [0, 1, 2, 3]:
                if z[v, k].solution_value() > 0.5:
                    f[v] = k
                    break
        
        obj_value = solver.Objective().Value()
        print(f"Objective value: {obj_value}")
        print(f"Time: {elapsed_time:.2f} ms")
        
        return {
            "status": "FEASIBLE",
            "objective": obj_value,
            "time_ms": elapsed_time,
            "f": f
        }
    
    else:
        print("\n" + "="*60)
        print("NO SOLUTION FOUND")
        print("="*60)
        print(f"Status: {status}")
        print(f"Time: {elapsed_time:.2f} ms")
        return {
            "status": "NO_SOLUTION",
            "time_ms": elapsed_time
        }


if __name__ == "__main__":
    
    solve_two_strong_roman(nx.path_graph(100), time_limit_seconds=60)
        