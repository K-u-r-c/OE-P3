import random
from typing import List

def selection_factory(kind: str):
    mapping = {
        "tournament": "tournament",
        "rws": "rws",
        "random": "random",
    }
    return mapping[kind]

def crossover_factory(kind: str):
    mapping = {
        "single_point": "single_point",
        "two_points": "two_points",
        "uniform": "uniform",
    }
    return mapping[kind]

def mutation_factory(kind: str):
    mapping = {
        "random": "random",
        "swap": "swap",
    }
    return mapping[kind]
