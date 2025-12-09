import optuna
import numpy as np
import .ga_algorithm as ga_algorithm

def objective(trial):
    return 